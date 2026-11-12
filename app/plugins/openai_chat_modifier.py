"""OpenAI chat, completions and responses request modifier plugin."""

from typing import override

from app.plugins.openai_dialect import (
    ChatCompletionsDialect,
    CompletionsDialect,
    OpenAIDialect,
    TranslatedOverrides,
)
from app.plugins.openai_request_modifier import OpenAIRequestModifier
from app.plugins.openai_responses_dialect import ResponsesDialect


_DIALECTS: tuple[OpenAIDialect, ...] = (ChatCompletionsDialect(), CompletionsDialect(), ResponsesDialect())


__all__ = ["OpenAIChatRequestModifier"]


# The plugin manager loads this module with spec.loader.exec_module and finds the
# class by name from config, so no production file imports it.
# lint-skip: coding.exported-api-unused
class OpenAIChatRequestModifier(OpenAIRequestModifier):
    """Plugin that applies configured overrides to OpenAI chat, completions and responses request bodies."""

    target_paths: tuple[str, ...] = tuple(dialect.path for dialect in _DIALECTS)

    @override
    def overrides_by_path(self, overrides: dict) -> dict[str, TranslatedOverrides]:
        """Give the overrides to apply on each target path, named the way its API names them.

        Args:
            overrides: The configured overrides.

        Returns:
            The overrides for each target path.
        """
        return {dialect.path: dialect.translate_overrides(overrides) for dialect in _DIALECTS}

    def name(self) -> str:
        return self.config.get("name", "openai_chat_modifier")
