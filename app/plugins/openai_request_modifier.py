"""Shared base class for OpenAI request modifier plugins."""

import copy
import json

from app.config import ServiceConfig
from app.plugins.base import PluginBase
from app.plugins.openai_dialect import TranslatedOverrides, find_prefix

__all__ = ["OpenAIRequestModifier"]


def _deep_merge(request_body: dict, overrides: dict) -> dict:
    """Recursively merge overrides into a copy of request_body.

    Merge rules:
    - Both dicts: recurse key by key
    - Both lists: merge element-by-element by index, with recursion
    - Otherwise: override value wins

    Args:
        request_body: Original request body dict
        overrides: Configuration overrides to apply

    Returns:
        New merged dict. It does not mutate either input
    """
    result = copy.deepcopy(request_body)
    for key, override_value in overrides.items():
        if key not in result:
            result[key] = copy.deepcopy(override_value)
            continue
        request_value = result[key]
        if isinstance(request_value, dict) and isinstance(override_value, dict):
            result[key] = _deep_merge(request_value, override_value)
        elif isinstance(request_value, list) and isinstance(override_value, list):
            result[key] = _merge_lists(request_value, override_value)
        else:
            result[key] = copy.deepcopy(override_value)
    return result


def _merge_lists(request_val: list, override_val: list) -> list:
    """Merge two lists element-by-element by index, with recursion.

    Args:
        request_val: List from the request body
        override_val: List from the overrides

    Returns:
        New merged list. It does not mutate either input
    """
    result = copy.deepcopy(request_val)
    for i, item in enumerate(override_val):
        if i >= len(result):
            result.append(copy.deepcopy(item))
            continue
        request_item = result[i]
        if isinstance(request_item, dict) and isinstance(item, dict):
            result[i] = _deep_merge(request_item, item)
        elif isinstance(request_item, list) and isinstance(item, list):
            result[i] = _merge_lists(request_item, item)
        else:
            result[i] = copy.deepcopy(item)
    return result


class OpenAIRequestModifier(PluginBase):
    """Base plugin that applies configured overrides to OpenAI API request bodies.

    Subclasses set ``target_paths`` to pick the URL paths to change, and may
    override ``overrides_by_path`` to rename override keys for each path.
    """

    target_paths: tuple[str, ...] = ()

    def __init__(
        self, config: dict, service_config: ServiceConfig | None = None
    ) -> None:
        super().__init__(config, service_config)
        self.overrides = copy.deepcopy(config.get("overrides", {}))
        self._translated = self.overrides_by_path(self.overrides)

    def overrides_by_path(self, overrides: dict) -> dict[str, TranslatedOverrides]:
        """Give the overrides to apply on each target path.

        Args:
            overrides: The configured overrides.

        Returns:
            The overrides for each target path.
        """
        return {target: TranslatedOverrides(overrides, {}) for target in self.target_paths}

    async def on_request(
        self, request_id: int, headers: dict[str, str], body: bytes, path: str
    ) -> tuple[dict[str, str], bytes, str]:
        """Apply configured overrides to a JSON request body for matching paths.

        Args:
            headers: The request headers dict
            body: The raw request body bytes
            path: The request URL path

        Returns:
            Tuple of (headers, modified body bytes, path) with Content-Length
            updated if body was modified. Returns unchanged if not valid
            JSON, not a dict, or path does not match target_paths.
        """
        headers, body, path = await super().on_request(request_id, headers, body, path)
        target = find_prefix(self.target_paths, path)
        if target is None:
            return headers, body, path

        if not self.overrides:
            return headers, body, path

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return headers, body, path

        if not isinstance(parsed, dict):
            return headers, body, path

        translated = self._translated[target]
        yielding = {key: value for key, value in translated.yielding.items() if key not in parsed}
        overrides = {**translated.forced, **yielding}
        # The request already sets every override that yields to it.
        if not overrides:
            return headers, body, path

        merged = _deep_merge(parsed, overrides)
        new_body = json.dumps(merged, ensure_ascii=False).encode("utf-8")
        headers["Content-Length"] = str(len(new_body))
        return headers, new_body, path
