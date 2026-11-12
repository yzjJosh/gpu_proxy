"""Responses with header and body changes."""

from __future__ import annotations

import abc
import asyncio
import json
from http import HTTPStatus
from typing import AsyncIterator, Awaitable, Callable, Optional, Union, override
from aiohttp import web
from aiohttp.abc import AbstractStreamWriter

__all__ = [
    "AbstractMutatingResponse",
    "MutatingStreamResponse",
    "MutatingWebSocketResponse",
    "CaptureResponse",
    "identity_mutator",
]

_HTTP_OK = 200


_MutatorFn = Callable[[dict[str, str], bytes, bool, int], Awaitable[bytes]]
_HeaderMutatorFn = Callable[[dict[str, str], int], dict[str, str]]


async def identity_mutator(
    _resp_headers: dict[str, str], chunk: bytes, _is_final: bool, _status_code: int
) -> bytes:
    """Return the chunk unchanged."""
    return chunk


def _chain_mutator(prev: _MutatorFn, mutator: _MutatorFn) -> _MutatorFn:
    """Join two body changes."""

    async def _chained(
        resp_headers: dict[str, str],
        chunk: bytes,
        is_final: bool,
        status_code: int,
    ) -> bytes:
        return await prev(
            resp_headers,
            await mutator(resp_headers, chunk, is_final, status_code),
            is_final,
            status_code,
        )

    return _chained


def _identity_header_mutator(
    headers: dict[str, str], _status_code: int
) -> dict[str, str]:
    """Return the headers unchanged.

    Args:
        headers: The response headers.
        _status_code: The response status code.

    Returns:
        The same headers.
    """
    return headers


def _chain_header_mutator(
    prev: _HeaderMutatorFn, mutator: _HeaderMutatorFn
) -> _HeaderMutatorFn:
    """Join two header changes.

    Args:
        prev: The earlier changes.
        mutator: The new change.

    Returns:
        The joined header change.
    """

    def _chained(headers: dict[str, str], status_code: int) -> dict[str, str]:
        return prev(mutator(headers, status_code), status_code)

    return _chained


class AbstractMutatingResponse(web.StreamResponse, metaclass=abc.ABCMeta):
    """A response that supports changes to headers and chunks."""

    @abc.abstractmethod
    def add_header_mutator(self, mutator: _HeaderMutatorFn) -> None:
        """Add a change to the response headers.

        Args:
            mutator: The change to apply.
        """

    @abc.abstractmethod
    def add_mutator(self, mutator: _MutatorFn) -> None:
        """Chain a mutator into the response pipeline."""


class MutatingStreamResponse(AbstractMutatingResponse):
    """An HTTP response with header and body changes."""

    def __init__(
        self,
        on_write: _MutatorFn,
        status: int = _HTTP_OK,
    ) -> None:
        super().__init__(status=status)
        self._on_prepare: _HeaderMutatorFn = _identity_header_mutator
        self._on_write: _MutatorFn = on_write

    @override
    async def prepare(self, request: web.Request) -> Optional[AbstractStreamWriter]:
        if not self.prepared:
            self.headers.update(self._on_prepare(dict(self.headers), self.status))
        return await super().prepare(request)

    @override
    async def write(self, chunk: Union[bytes, bytearray, memoryview]) -> None:
        await super().write(
            await self._on_write(dict(self.headers), chunk, False, self.status)
        )

    @override
    async def write_eof(self, data: bytes = b"") -> None:
        await super().write_eof(
            await self._on_write(dict(self.headers), data, True, self.status)
        )

    @override
    def add_mutator(self, mutator: _MutatorFn) -> None:
        self._on_write = _chain_mutator(self._on_write, mutator)

    @override
    def add_header_mutator(self, mutator: _HeaderMutatorFn) -> None:
        self._on_prepare = _chain_header_mutator(self._on_prepare, mutator)


class MutatingWebSocketResponse(web.WebSocketResponse, AbstractMutatingResponse):
    """A WebSocket response with header and message changes."""

    def __init__(self, on_send: _MutatorFn) -> None:
        super().__init__()
        self._on_prepare: _HeaderMutatorFn = _identity_header_mutator
        self._on_send: _MutatorFn = on_send

    @override
    async def prepare(self, request: web.Request) -> Optional[AbstractStreamWriter]:
        if not self.prepared:
            self.headers.update(self._on_prepare(dict(self.headers), self.status))
        return await super().prepare(request)

    @override
    async def send_str(self, message: str, compress: Optional[int] = None) -> None:
        encoded = message.encode("utf-8")
        mutated = await self._on_send({}, encoded, False, self.status)
        await super().send_str(mutated.decode("utf-8"), compress=compress)

    @override
    async def send_bytes(
        self, data: Union[bytes, bytearray, memoryview], compress: Optional[int] = None
    ) -> None:
        mutated = await self._on_send({}, data, False, self.status)
        await super().send_bytes(mutated, compress=compress)

    @override
    async def send_json(
        self, data: dict, compress: Optional[int] = None
    ) -> None:
        raw = json.dumps(data).encode("utf-8")
        mutated = await self._on_send({}, raw, False, self.status)
        await super().send_str(mutated.decode("utf-8"), compress=compress)

    @override
    def add_mutator(self, mutator: _MutatorFn) -> None:
        self._on_send = _chain_mutator(self._on_send, mutator)

    @override
    def add_header_mutator(self, mutator: _HeaderMutatorFn) -> None:
        self._on_prepare = _chain_header_mutator(self._on_prepare, mutator)


def _default_reason(status: int) -> str:
    """Return the standard name for a status code."""
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return ""


class CaptureResponse(AbstractMutatingResponse):
    """Hold response chunks for a caller to read."""

    def __init__(self) -> None:
        super().__init__()
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._eof: bool = False
        self._prepared: bool = False
        self._prepared_ready: asyncio.Event = asyncio.Event()
        self.headers["Content-Type"] = "text/event-stream"
        self._capture_status: int = 200
        self._capture_reason: str = "OK"
        self._on_write: Callable[[dict[str, str], bytes, bool, int], Awaitable[bytes]] = (
            identity_mutator
        )
        self._on_prepare: Callable[[dict[str, str], int], dict[str, str]] = (
            _identity_header_mutator
        )

    @property
    def status_code(self) -> int:
        """The captured response status."""
        return self._capture_status

    @property
    @override
    def prepared(self) -> bool:
        """Whether the response is ready."""
        return self._prepared

    @property
    def reason(self) -> str:
        return self._capture_reason

    @override
    def set_status(self, status: int, reason: str = "") -> None:
        super().set_status(status, reason)
        self._capture_status = status
        self._capture_reason = reason if reason else _default_reason(status)

    @override
    async def prepare(self, _request: web.Request) -> None:
        if not self._prepared:
            self.headers.update(self._on_prepare(dict(self.headers), self._capture_status))
        self._prepared = True
        self._prepared_ready.set()

    async def wait_prepared(self) -> None:
        """Wait until the response is ready."""
        await self._prepared_ready.wait()

    @override
    async def write(self, chunk: bytes) -> None:
        if not self._eof:
            mutated = await self._on_write(
                dict(self.headers), chunk, False, self._capture_status
            )
            await self._queue.put(mutated)

    @override
    async def write_eof(self, data: bytes = b"") -> None:
        if self._eof:
            return
        self._eof = True
        mutated = await self._on_write(dict(self.headers), data, True, self._capture_status)
        if mutated:
            await self._queue.put(mutated)
        await self._queue.put(None)

    async def close(self) -> None:
        """Finish the captured response."""
        await self.write_eof()

    @override
    def add_mutator(
        self, mutator: Callable[[dict[str, str], bytes, bool, int], Awaitable[bytes]]
    ) -> None:
        self._on_write = _chain_mutator(self._on_write, mutator)

    @override
    def add_header_mutator(
        self, mutator: Callable[[dict[str, str], int], dict[str, str]]
    ) -> None:
        self._on_prepare = _chain_header_mutator(self._on_prepare, mutator)

    async def iter_chunks(self) -> AsyncIterator[bytes]:
        """Yield the captured response chunks."""
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                break
            yield chunk
