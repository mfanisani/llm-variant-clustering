from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .llm_cache import GenerationParameters, ProviderGeneration


class LocalTransformersProvider:
    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        parameters: GenerationParameters,
        cache_dir: Path,
        device: str = "cpu",
    ) -> None:
        if parameters.temperature != 0:
            raise ValueError("Local deterministic provider requires temperature=0")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Install the project 'embeddings' extra for local generation"
            ) from error

        self.torch = torch
        self.parameters = parameters
        self.model_id = model_id
        self.model_revision = model_revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=model_revision,
            cache_dir=str(cache_dir),
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=model_revision,
            cache_dir=str(cache_dir),
            dtype="auto",
            trust_remote_code=False,
        )
        self.model.to(device)
        self.model.eval()

    def generate(self, rendered_prompt: str) -> ProviderGeneration:
        system, user = split_rendered_prompt(rendered_prompt)
        prompt_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.parameters.max_output_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_tokens = output[0, inputs["input_ids"].shape[1] :]
        generated_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )
        parsed = extract_json_object(generated_text)
        return ProviderGeneration(
            response=parsed,
            provider_response_id=None,
            provider_model=f"{self.model_id}@{self.model_revision}",
            usage={
                "input_tokens": int(inputs["input_ids"].shape[1]),
                "output_tokens": int(generated_tokens.shape[0]),
            },
            attempt_count=1,
        )


def split_rendered_prompt(rendered_prompt: str) -> tuple[str, str]:
    separator = "\n\nUSER\n"
    if not rendered_prompt.startswith("SYSTEM\n") or separator not in rendered_prompt:
        raise ValueError("Rendered prompt does not contain SYSTEM and USER sections")
    system, user = rendered_prompt[len("SYSTEM\n") :].split(separator, maxsplit=1)
    return system.strip(), user.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_line_end = stripped.find("\n")
        final_fence = stripped.rfind("```")
        if first_line_end == -1 or final_fence <= first_line_end:
            raise ValueError("Malformed fenced JSON response")
        stripped = stripped[first_line_end + 1 : final_fence].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end < start:
        raise ValueError("Local model response does not contain a JSON object")
    parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, Mapping):
        raise ValueError("Local model response JSON must be an object")
    return dict(parsed)
