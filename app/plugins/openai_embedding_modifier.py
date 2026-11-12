"""OpenAI embedding request modifier plugin."""

from app.plugins.openai_request_modifier import OpenAIRequestModifier


_EMBEDDINGS_PATHS: tuple[str, ...] = ("/v1/embeddings", "/v1/embedding", "/embedding")


__all__ = ["OpenAIEmbeddingRequestModifier"]


# The plugin manager loads this module with spec.loader.exec_module and finds the
# class by name from config, so no production file imports it.
# lint-skip: coding.exported-api-unused
class OpenAIEmbeddingRequestModifier(OpenAIRequestModifier):
    """Plugin that applies configured overrides to OpenAI embeddings API request bodies."""

    target_paths: tuple[str, ...] = _EMBEDDINGS_PATHS

    def name(self) -> str:
        return self.config.get("name", "openai_embedding_modifier")
