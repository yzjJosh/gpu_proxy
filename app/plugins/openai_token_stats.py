"""Track OpenAI token usage and tokens-per-second (TPS) in real time."""

import asyncio
import json
import logging
import time
import math
from typing import Callable, override

from tiktoken import get_encoding

from app.config import ServiceConfig
from app.plugins.base import PluginBase
from app.plugins.openai_api_formats.base import OpenAIAPIFormat, find_api_format_by_path_prefix
from app.plugins.openai_api_formats.chat_completions import ChatCompletionsAPIFormat
from app.plugins.openai_api_formats.legacy_completions import LegacyCompletionsAPIFormat
from app.plugins.openai_api_formats.responses import ResponsesAPIFormat
from app.plugins.utils import SSEByteDecoder
from app.plugins.utils import decode_json_object
from collections import deque

_logger = logging.getLogger(__name__)

__all__ = ["OpenAITokenStats"]

_PREFILLING = 0
_GENERATING = 1
_API_FORMATS: tuple[OpenAIAPIFormat, ...] = (
    ChatCompletionsAPIFormat(),
    LegacyCompletionsAPIFormat(),
    ResponsesAPIFormat(),
)


class _RequestState:
    def __init__(self, api_format: OpenAIAPIFormat) -> None:
        self.state: int = _PREFILLING
        self.state_change_timestamp: float = 0.0
        self.is_stream: bool = False
        self.buffer: bytearray = bytearray()
        self.sse_events: list[dict] = []
        self.decoder: SSEByteDecoder = SSEByteDecoder()
        self.request_toks: int = 0
        self.api_format: OpenAIAPIFormat = api_format


class OpenAITokenStats(PluginBase):
    """Track cumulative token usage from OpenAI API responses.

    Publishes two rows via the UI publish callback:
    - Tokens/s: current TPS (from last-second token count) + peak TPS
    - Stats: total / input / output tokens (hidden until non-zero)
    """

    def __init__(
        self, config: dict, service_config: ServiceConfig | None = None
    ) -> None:
        super().__init__(config, service_config)
        self._publish: Callable[[str, str], None] | None = None
        self._input_tokens = 0
        self._output_tokens = 0
        self._total_tokens = 0
        self._gen_tps_peak = 0
        self._gen_tps_current = 0
        self._gen_tps_chars = ""
        self._prefill_stats_last_min = deque()
        self._prefill_stats_last_s: list[tuple[int, int]] = []
        self._prefill_tps_last_min = 0
        self._prefill_tps_peak = 0
        self._tps_task: asyncio.Task[None] | None = None
        try:
            self._tps_task = asyncio.create_task(self._tps_loop())
        except RuntimeError:
            pass
        self._requests: dict[int, _RequestState] = {}
        self._encoding = get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text, allowed_special="all"))

    def __del__(self) -> None:
        try:
            if self._tps_task is not None:
                self._tps_task.cancel()
        except RuntimeError:
            pass

    def name(self) -> str:
        return self.config.get("name", "openai_token_stats")

    def init_ui(self, publish: Callable[[str, str], None]) -> None:
        self._publish = publish

    async def on_request(
        self, request_id: int, headers: dict[str, str], body: bytes, path: str
    ) -> tuple[dict[str, str], bytes, str]:
        headers, body, path = await super().on_request(request_id, headers, body, path)

        api_format = find_api_format_by_path_prefix(_API_FORMATS, path)
        if api_format is None:
            return headers, body, path

        if not body:
            return headers, body, path

        parsed = decode_json_object(body)
        if parsed is None:
            return headers, body, path

        state = _RequestState(api_format)
        state.is_stream = parsed.get("stream", False)
        state.state_change_timestamp = time.time()
        try:
            state.request_toks = self._count_tokens(api_format.extract_prompt_text(parsed))
        except Exception as e:
            _logger.error("Failed to count request tokens: %s", e)
            state.request_toks = 0
        self._requests[request_id] = state
        self._publish_stats()

        api_format.enable_streaming(parsed)
        new_body = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
        headers["Content-Length"] = str(len(new_body))
        return headers, new_body, path

    @override
    def on_response_headers(
        self, request_id: int, headers: dict[str, str], status_code: int
    ) -> dict[str, str]:
        """Set the content type for clients that want a full reply."""
        state = self._requests.get(request_id)
        if state is not None and not state.is_stream and status_code == 200:
            headers = {
                key: value for key, value in headers.items()
                if key.lower() != "content-type"
            }
            headers["Content-Type"] = "application/json"
        return headers

    def _extract_usage(self, api_format: OpenAIAPIFormat, parsed_json: dict) -> None:
        usage = api_format.extract_token_usage(parsed_json)
        if usage is not None:
            self._input_tokens += usage.input_tokens
            self._output_tokens += usage.output_tokens
            self._total_tokens += usage.total_tokens
            self._publish_stats()
    
    def _update_gen_tps(self) -> None:
        """Update the generating TPS stats."""
        tps_text = self._gen_tps_chars
        self._gen_tps_chars = ""
        self._gen_tps_current = self._count_tokens(tps_text)
        if self._gen_tps_current > self._gen_tps_peak:
            self._gen_tps_peak = self._gen_tps_current
    
    def _update_prefill_tps(self) -> None:
        """Update the prefill TPS stats."""
        self._prefill_stats_last_min.append(self._prefill_stats_last_s)
        self._prefill_stats_last_s = []
        if len(self._prefill_stats_last_min) > 60:
            self._prefill_stats_last_min.popleft()
        # Compute average tps in last min
        n = len(self._prefill_stats_last_min)
        deltas = [0] * n
        for i, stats in enumerate(reversed(self._prefill_stats_last_min)):
            for request_toks, prefilling_s in stats:
                prefill_tps = request_toks / prefilling_s
                deltas[i] += prefill_tps
                end = i + prefilling_s
                if end < n:
                    deltas[end] -= prefill_tps
        cur_tokens = 0
        total_toks = 0
        total_s = 0
        for delta in deltas:
            cur_tokens += delta
            total_toks += cur_tokens
            if cur_tokens > 0:
                total_s += 1
        self._prefill_tps_last_min = int(total_toks / total_s) if total_s > 0 else 0
        self._prefill_tps_peak = max(self._prefill_tps_peak, self._prefill_tps_last_min)
        
    def update_tps_and_publish(self) -> None:
        """Flush accumulated TPS characters and publish stats immediately."""
        self._update_gen_tps()
        self._update_prefill_tps()
        self._publish_stats()

    @staticmethod
    def _fmt(n: int) -> str:
        """Format a number to a human-readable compact string (e.g., 1.2K, 3.4M)."""
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)

    def _publish_stats(self) -> None:
        if self._publish is None:
            return
        sublabel_style = "display:inline-block;font-size:0.7rem;font-weight:600;padding:2px 6px;border-radius:3px;margin-right:4px;background:#e8e8e8;color:#555;text-transform:uppercase;letter-spacing:0.04em"
        chip_style = "display:inline-block;font-size:0.8rem;font-weight:700;padding:4px 10px;border-radius:12px;margin:0 2px"
        states: list[_RequestState] = list(self._requests.values())
        prefill_count = sum(1 for s in states if s.state == _PREFILLING)
        generate_count = sum(1 for s in states if s.state == _GENERATING)
        stages_row = (
            f'<div style="display:flex;align-items:center;gap:8px">'
            f'<span style="{sublabel_style}">Stages</span>'
            f'<span style="{chip_style};background:#f3e8fd;color:#6b21a8">{prefill_count} prefilling</span>'
            f'<span style="{chip_style};background:#e0f2fe;color:#0369a1">{generate_count} generating</span>'
            f"</div>"
        )
        stats_row = ""
        if self._total_tokens or self._input_tokens or self._output_tokens:
            stats_row = (
                f'<div style="display:flex;align-items:center;gap:8px">'
                f'<span style="{sublabel_style}">Stats</span>'
                f'<span style="{chip_style};background:#e8f0fe;color:#004494">{self._fmt(self._total_tokens)} total</span>'
                f'<span style="{chip_style};background:#e6f4ea;color:#107c10">{self._fmt(self._input_tokens)} in</span>'
                f'<span style="{chip_style};background:#fef3e0;color:#b8860b">{self._fmt(self._output_tokens)} out</span>'
                f"</div>"
            )
        html = (
            f'<div style="display:flex;flex-direction:column;width:100%;gap:8px">'
            f"{stages_row}"
            f'<div style="display:flex;align-items:center;gap:8px">'
            f'<span style="{sublabel_style}">Prefilling</span>'
            f'<span style="{chip_style};background:#fefce8;color:#a16207">{self._prefill_tps_last_min} t/s last min</span>'
            f'<span style="{chip_style};background:#fef9c3;color:#854d0e">{self._prefill_tps_peak} t/s peak</span>'
            f"</div>"
            f'<div style="display:flex;align-items:center;gap:8px">'
            f'<span style="{sublabel_style}">Generating</span>'
            f'<span style="{chip_style};background:#fff3e0;color:#c2410c">{self._gen_tps_current} t/s current</span>'
            f'<span style="{chip_style};background:#fce4ec;color:#c62828">{self._gen_tps_peak} t/s peak</span>'
            f"</div>"
            f"{stats_row}"
            f"</div>"
        )
        self._publish("Token Usage", html)

    def on_cancelled(self, request_id: int) -> None:
        super().on_cancelled(request_id)
        self._requests.pop(request_id, None)
        self._publish_stats()

    async def _tps_loop(self) -> None:
        while True:
            await asyncio.sleep(1)
            self.update_tps_and_publish()

    async def on_response_chunk(
        self,
        request_id: int,
        response_headers: dict[str, str],
        chunk: bytes,
        path: str,
        status_code: int,
        is_final: bool,
    ) -> bytes:
        chunk = await super().on_response_chunk(
            request_id, response_headers, chunk, path, status_code, is_final
        )
        if find_api_format_by_path_prefix(_API_FORMATS, path) is None:
            return chunk

        state: _RequestState | None = self._requests.get(request_id)
        if state is None:
            return chunk

        events = state.decoder.feed(chunk)
        if is_final:
            events += state.decoder.close()[0]
        has_delta = False
        for _event_type, data, _event_id, _raw_lines in events:
            if data == "[DONE]":
                continue
            try:
                parsed_json = json.loads(data)
                state.sse_events.append(parsed_json)
                for generated_text in state.api_format.extract_generated_texts(parsed_json):
                    self._gen_tps_chars += generated_text
                    has_delta = len(generated_text) > 0
                self._extract_usage(state.api_format, parsed_json)
            except json.JSONDecodeError:
                continue
        
        if state.state == _PREFILLING and has_delta:
            prefilling_s = max(1, int(math.ceil(time.time() - state.state_change_timestamp)))
            self._prefill_stats_last_s.append((state.request_toks, prefilling_s))
            state.state = _GENERATING
            state.state_change_timestamp = time.time()
            self._publish_stats()

        if not state.is_stream:
            state.buffer.extend(chunk)

        if not is_final:
            return chunk if state.is_stream else b""

        # Handle final chunk
        del self._requests[request_id]
        self._publish_stats()
        if state.is_stream:
            return chunk
        else:
            result = state.api_format.assemble_non_streaming_response(state.sse_events)
            if result is not None:
                return result
            return bytes(state.buffer)
