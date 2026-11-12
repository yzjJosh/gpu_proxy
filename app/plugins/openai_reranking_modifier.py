"""OpenAI reranking request modifier plugin."""

from app.plugins.openai_request_modifier import OpenAIRequestModifier


_RERANKING_PATHS: tuple[str, ...] = ("/v1/rerank", "/reranking")


__all__ = ["OpenAIRerankingRequestModifier"]


# The plugin manager loads this module with spec.loader.exec_module and finds the
# class by name from config, so no production file imports it.
# lint-skip: coding.exported-api-unused
class OpenAIRerankingRequestModifier(OpenAIRequestModifier):
    """Plugin that applies configured overrides to reranking API request bodies."""

    target_paths: tuple[str, ...] = _RERANKING_PATHS

    def name(self) -> str:
        return self.config.get("name", "openai_reranking_modifier")
