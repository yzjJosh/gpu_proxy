"""Request, stream, and response format for the OpenAI legacy completions API."""

from typing import override

from app.plugins.openai_api_formats.base import ChoicesBasedAPIFormat
from app.plugins.utils import assemble_non_streaming_response

__all__ = ["LegacyCompletionsAPIFormat"]


class LegacyCompletionsAPIFormat(ChoicesBasedAPIFormat):
    """The OpenAI legacy text completions API format."""

    endpoint_path: str = "/v1/completions"

    @override
    def extract_prompt_text(self, request_body: dict) -> str:
        prompt = request_body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = "".join(prompt)
        return prompt

    @override
    def assemble_non_streaming_response(self, stream_events: list[dict]) -> bytes | None:
        return assemble_non_streaming_response(stream_events, is_chat=False)
