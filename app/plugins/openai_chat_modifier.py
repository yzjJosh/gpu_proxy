"""OpenAI chat, completions and responses request modifier plugin."""

from typing import override

from app.plugins.openai_api_formats.base import (
    OpenAIAPIFormat,
    RequestOverrides,
)
from app.plugins.openai_api_formats.chat_completions import ChatCompletionsAPIFormat
from app.plugins.openai_api_formats.legacy_completions import LegacyCompletionsAPIFormat
from app.plugins.openai_api_formats.responses import ResponsesAPIFormat
from app.plugins.openai_request_modifier import OpenAIRequestModifier


_API_FORMATS: tuple[OpenAIAPIFormat, ...] = (ChatCompletionsAPIFormat(), LegacyCompletionsAPIFormat(), ResponsesAPIFormat())


__all__ = ["OpenAIChatRequestModifier"]


# The plugin manager loads this module with spec.loader.exec_module and finds the
# class by name from config, so no production file imports it.
# lint-skip: coding.exported-api-unused
class OpenAIChatRequestModifier(OpenAIRequestModifier):
    """Plugin that applies configured overrides to OpenAI chat, completions and responses request bodies."""

    target_paths: tuple[str, ...] = tuple(api_format.endpoint_path for api_format in _API_FORMATS)

    @override
    def overrides_by_path(self, overrides: dict) -> dict[str, RequestOverrides]:
        """Return the overrides to apply on each target path, using that API's field names.

        Args:
            overrides: The configured overrides.

        Returns:
            The overrides for each target path.
        """
        return {
            api_format.endpoint_path: api_format.convert_request_overrides(overrides)
            for api_format in _API_FORMATS
        }

    def name(self) -> str:
        return self.config.get("name", "openai_chat_modifier")
