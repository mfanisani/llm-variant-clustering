from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any

from .llm_cache import GenerationParameters, ProviderGeneration


class OpenAIResponsesProvider:
    def __init__(
        self,
        *,
        model: str,
        parameters: GenerationParameters,
        schema: Mapping[str, Any],
        schema_name: str,
        api_key_environment_variable: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
        max_attempts: int = 3,
        backoff_seconds: float = 2.0,
    ) -> None:
        if parameters.seed is not None:
            raise ValueError(
                "The OpenAI Responses API adapter does not support a seed; use None"
            )
        api_key = os.environ.get(api_key_environment_variable)
        if not api_key:
            raise RuntimeError(
                f"Required environment variable {api_key_environment_variable} "
                "is not set"
            )
        if not schema_name or not schema_name.replace("_", "").isalnum():
            raise ValueError("schema_name must contain only letters, digits, underscores")
        if max_attempts < 1 or backoff_seconds < 0:
            raise ValueError("Invalid retry configuration")

        try:
            from openai import (
                APIConnectionError,
                APIStatusError,
                APITimeoutError,
                OpenAI,
                RateLimitError,
            )
        except ImportError as error:
            raise RuntimeError(
                "Install the project 'openai' extra to use this provider"
            ) from error

        self.model = model
        self.parameters = parameters
        self.schema = dict(schema)
        self.schema_name = schema_name
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.transient_exception_types = (
            APITimeoutError,
            APIConnectionError,
            RateLimitError,
        )
        self.api_status_error_type = APIStatusError
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=0,
        )

    def generate(self, rendered_prompt: str) -> ProviderGeneration:
        response = None
        attempt = 0
        while response is None:
            attempt += 1
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=rendered_prompt,
                    max_output_tokens=self.parameters.max_output_tokens,
                    temperature=self.parameters.temperature,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": self.schema_name,
                            "schema": self.schema,
                            "strict": True,
                        }
                    },
                    store=False,
                    truncation="disabled",
                )
            except self.transient_exception_types:
                if attempt >= self.max_attempts:
                    raise
                time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
            except self.api_status_error_type as error:
                if (
                    error.status_code not in {408, 409, 429, 500, 502, 503, 504}
                    or attempt >= self.max_attempts
                ):
                    raise
                time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))

        if not response.output_text:
            raise RuntimeError(
                f"OpenAI response {response.id} did not contain output text"
            )
        parsed = json.loads(response.output_text)
        if not isinstance(parsed, dict):
            raise ValueError("Structured response must be a JSON object")
        usage = response.usage.model_dump(mode="json") if response.usage else {}
        return ProviderGeneration(
            response=parsed,
            provider_response_id=response.id,
            provider_model=response.model,
            usage=usage,
            attempt_count=attempt,
        )
