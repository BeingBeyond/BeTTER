from __future__ import annotations

from dataclasses import dataclass
import json
import os
import sys
import traceback

from .task_draft import CompiledTaskAuthoringBundle, TaskDraftSpec, TaskTemplateSpec, compile_task_draft
from .task_draft_pipeline import task_draft_from_json
from .task_draft_prompt import TaskDraftPromptBundle, build_task_draft_prompt_bundle


@dataclass(frozen=True)
class TaskDraftGenerationResult:
    prompt_bundle: TaskDraftPromptBundle
    raw_response_text: str
    draft: TaskDraftSpec


class OpenAITaskDraftGenerator:
    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4000,
        client: object | None = None,
    ) -> None:
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.client = client if client is not None else self._init_client(api_key=api_key, base_url=base_url)

    def generate_task_draft(
        self,
        *,
        template: TaskTemplateSpec,
        guidance: str,
    ) -> TaskDraftGenerationResult:
        prompt_bundle = build_task_draft_prompt_bundle(template=template, guidance=guidance)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt_bundle.system_prompt},
                {"role": "user", "content": prompt_bundle.user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        raw_response_text = _extract_response_text(response)
        try:
            json_text = extract_json_object_text(raw_response_text)
            draft = task_draft_from_json(json_text, guidance_override=guidance)
        except Exception as exc:
            tb = traceback.format_exc().strip()
            snippet = raw_response_text[:4000]
            print("Task draft generation parse failure.", file=sys.stderr)
            print("Raw model response:\n" + snippet, file=sys.stderr)
            print("Traceback:\n" + tb, file=sys.stderr)
            raise ValueError(
                f"{exc}\n\nModel raw response:\n{snippet}\n\nTraceback:\n{tb}"
            ) from exc
        return TaskDraftGenerationResult(
            prompt_bundle=prompt_bundle,
            raw_response_text=raw_response_text,
            draft=draft,
        )

    def generate_task_authoring_bundle(
        self,
        *,
        task_id: str,
        template: TaskTemplateSpec,
        guidance: str,
    ) -> CompiledTaskAuthoringBundle:
        result = self.generate_task_draft(template=template, guidance=guidance)
        return compile_task_draft(
            task_id=task_id,
            template=template,
            draft=result.draft,
        )

    @staticmethod
    def _init_client(*, api_key: str | None, base_url: str | None) -> object:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("OpenAI library not installed in the Isaac Sim / BeTTER environment. Install with: pip install openai") from exc

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY or pass api_key.")

        client_kwargs: dict[str, object] = {"api_key": resolved_api_key}
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        return OpenAI(**client_kwargs)



def extract_json_object_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].lstrip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    if start < 0:
        raise ValueError("Model response does not contain a JSON object.")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]

    raise ValueError("Model response does not contain a complete JSON object.")



def _extract_response_text(response: object) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("Model response did not include choices.")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Model response did not include text content.")
    return content
