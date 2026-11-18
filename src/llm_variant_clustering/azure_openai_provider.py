from __future__ import annotations

import json
import random
import time
from collections.abc import Mapping
from typing import Any

from .llm_cache import GenerationParameters, ProviderGeneration
from .local_provider import split_rendered_prompt

UNSUPPORTED_STRICT_SCHEMA_KEYWORDS = {
    "$id",
    "$schema",
    "contains",
    "format",
    "maxContains",
    "maxItems",
    "maxLength",
    "maxProperties",
    "maximum",
    "minContains",
    "minItems",
    "minLength",
    "minProperties",
    "minimum",
    "multipleOf",
    "pattern",
    "patternProperties",
    "propertyNames",
    "uniqueItems",
    "unevaluatedProperties",
}


def _extract_retry_after_seconds(error: Exception) -> float | None:
    """Best-effort ``Retry-After`` (seconds) from an OpenAI APIStatusError."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers is None:
        return None
    for key in ("Retry-After", "retry-after", "retry-after-ms"):
        value = headers.get(key) if hasattr(headers, "get") else None
        if value is None:
            continue
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            continue
        return seconds / 1000.0 if key == "retry-after-ms" else seconds
    return None


def azure_strict_tool_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, Mapping):
            converted = {
                key: convert(item)
                for key, item in value.items()
                if key not in UNSUPPORTED_STRICT_SCHEMA_KEYWORDS
                and key != "const"
            }
            if "const" in value:
                constant = value["const"]
                converted["enum"] = [constant]
                if "type" not in converted:
                    converted["type"] = {
                        str: "string",
                        bool: "boolean",
                        int: "integer",
                        float: "number",
                    }.get(type(constant))
            constraints = []
            if value.get("uniqueItems") is True:
                constraints.append("Array items must be unique.")
            if "minItems" in value:
                constraints.append(
                    f"The array must contain at least {value['minItems']} items."
                )
            if "maxItems" in value:
                constraints.append(
                    f"The array must contain at most {value['maxItems']} items."
                )
            if "minimum" in value:
                constraints.append(
                    f"The numeric value must be at least {value['minimum']}."
                )
            if "maximum" in value:
                constraints.append(
                    f"The numeric value must be at most {value['maximum']}."
                )
            if "minLength" in value:
                constraints.append(
                    f"The string must contain at least {value['minLength']} characters."
                )
            if constraints:
                existing = str(converted.get("description", "")).strip()
                converted["description"] = " ".join(
                    part for part in [existing, *constraints] if part
                )
            return converted
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    converted = convert(schema)
    if not isinstance(converted, dict):
        raise ValueError("Tool schema must be a JSON object")
    return converted


class AzureOpenAIToolProvider:
    def __init__(
        self,
        *,
        endpoint: str,
        deployment: str,
        api_version: str,
        parameters: GenerationParameters,
        schema: Mapping[str, Any],
        schema_name: str,
        max_attempts: int = 3,
        backoff_seconds: float = 2.0,
        max_backoff_seconds: float | None = None,
        jitter_seconds: float = 0.0,
        honor_retry_after: bool = False,
    ) -> None:
        """Construct the provider.

        ``max_backoff_seconds``, ``jitter_seconds`` and ``honor_retry_after``
        are purely additive, opt-in knobs (default values reproduce the
        original unbounded-exponential-backoff behaviour exactly) so existing
        callers are unaffected unless they explicitly opt in.
        """
        if parameters.seed is not None:
            raise ValueError("Azure OpenAI adapter does not support a seed")
        if parameters.temperature != 0:
            raise ValueError("Azure OpenAI pilot requires temperature=0")
        if not endpoint.startswith("https://"):
            raise ValueError("Azure OpenAI endpoint must use HTTPS")
        if not deployment or not api_version:
            raise ValueError("Deployment and API version are required")
        if not schema_name or not schema_name.replace("_", "").isalnum():
            raise ValueError("schema_name must contain only letters, digits, underscores")
        if max_attempts < 1 or backoff_seconds < 0:
            raise ValueError("Invalid retry configuration")

        try:
            from azure.identity import AzureCliCredential, get_bearer_token_provider
            from openai import (
                APIConnectionError,
                APIStatusError,
                APITimeoutError,
                AzureOpenAI,
                RateLimitError,
            )
        except ImportError as error:
            raise RuntimeError(
                "Install the project 'azure' extra to use Azure OpenAI with Entra ID"
            ) from error

        token_provider = get_bearer_token_provider(
            AzureCliCredential(process_timeout=60),
            "https://cognitiveservices.azure.com/.default",
        )
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
            timeout=120.0,
            max_retries=0,
        )
        self.deployment = deployment
        self.parameters = parameters
        self.schema = azure_strict_tool_schema(schema)
        self.schema_name = schema_name
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self.jitter_seconds = jitter_seconds
        self.honor_retry_after = honor_retry_after
        self.transient_exception_types = (
            APITimeoutError,
            APIConnectionError,
            RateLimitError,
        )
        self.api_status_error_type = APIStatusError

    def _sleep_seconds(self, attempt: int, error: Exception) -> tuple[float, bool]:
        """Exponential backoff, optionally capped, jittered, and Retry-After aware.

        With all three opt-in knobs left at their defaults this reproduces
        ``self.backoff_seconds * (2 ** (attempt - 1))`` exactly. Returns
        ``(delay_seconds, honoured_retry_after)`` so callers can track how
        often a server ``Retry-After`` header actually drove the wait.
        """
        delay = self.backoff_seconds * (2 ** (attempt - 1))
        if self.max_backoff_seconds is not None:
            delay = min(delay, self.max_backoff_seconds)
        honoured_retry_after = False
        if self.honor_retry_after:
            retry_after = _extract_retry_after_seconds(error)
            if retry_after is not None:
                delay = max(delay, retry_after)
                honoured_retry_after = True
        if self.jitter_seconds:
            delay += random.uniform(0, self.jitter_seconds)
        return delay, honoured_retry_after

    def generate(self, rendered_prompt: str) -> ProviderGeneration:
        system, user = split_rendered_prompt(rendered_prompt)
        response = None
        attempt = 0
        retry_after_waits = 0
        while response is None:
            attempt += 1
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": self.schema_name,
                                "description": (
                                    "Return the requested process-variant analysis."
                                ),
                                "parameters": self.schema,
                                "strict": True,
                            },
                        }
                    ],
                    tool_choice={
                        "type": "function",
                        "function": {"name": self.schema_name},
                    },
                    max_tokens=self.parameters.max_output_tokens,
                    temperature=0,
                )
            except self.transient_exception_types as error:
                if attempt >= self.max_attempts:
                    raise
                delay, honoured = self._sleep_seconds(attempt, error)
                retry_after_waits += int(honoured)
                time.sleep(delay)
            except self.api_status_error_type as error:
                if (
                    error.status_code not in {408, 409, 429, 500, 502, 503, 504}
                    or attempt >= self.max_attempts
                ):
                    raise
                delay, honoured = self._sleep_seconds(attempt, error)
                retry_after_waits += int(honoured)
                time.sleep(delay)

        if len(response.choices) != 1:
            raise ValueError(f"Expected one completion choice, got {len(response.choices)}")
        tool_calls = response.choices[0].message.tool_calls or []
        selected_calls = [
            call for call in tool_calls if call.function.name == self.schema_name
        ]
        if len(selected_calls) != 1:
            raise ValueError(
                f"Expected one {self.schema_name!r} tool call, got "
                f"{len(selected_calls)}"
            )
        parsed = json.loads(selected_calls[0].function.arguments)
        if not isinstance(parsed, Mapping):
            raise ValueError("Azure OpenAI tool arguments are not a JSON object")
        usage = response.usage.model_dump(mode="json") if response.usage else {}
        return ProviderGeneration(
            response=dict(parsed),
            provider_response_id=response.id,
            provider_model=response.model,
            usage=usage,
            attempt_count=attempt,
            request_metadata={"retry_after_waits": retry_after_waits},
        )
