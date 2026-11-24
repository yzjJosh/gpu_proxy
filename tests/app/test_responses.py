"""Tests for captured response headers, status, and chunks."""

import asyncio
from http import HTTPStatus
from typing import Awaitable, Callable
from unittest.mock import AsyncMock, Mock
import pytest
from aiohttp.test_utils import make_mocked_request
from app.responses import (
    CaptureResponse,
    MutatingStreamResponse,
    MutatingWebSocketResponse,
    identity_mutator,
)

_UNKNOWN_STATUS = 599
_WS_HEADERS = {
    "Upgrade": "websocket",
    "Connection": "Upgrade",
    "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==",
    "Sec-WebSocket-Version": "13",
}


async def _send_ws_text(response: MutatingWebSocketResponse) -> None:
    await response.send_str("hello")


async def _send_ws_bytes(response: MutatingWebSocketResponse) -> None:
    await response.send_bytes(b"hello")


async def _send_ws_json(response: MutatingWebSocketResponse) -> None:
    await response.send_json({"hello": "world"})


@pytest.mark.parametrize("status,reason,expected", [
    (HTTPStatus.OK, "", "OK"),
    (HTTPStatus.BAD_GATEWAY, "", "Bad Gateway"),
    (_UNKNOWN_STATUS, "", ""),
    (_UNKNOWN_STATUS, "Custom", "Custom"),
])
def test_capture_status_reason(status: int, reason: str, expected: str) -> None:
    """A captured status keeps its supplied or standard reason."""
    response = CaptureResponse()
    response.set_status(status, reason)
    assert response.status_code == status
    assert response.reason == expected


async def test_capture_rejected_status_change_keeps_status_and_reason() -> None:
    """A rejected status change leaves the response unchanged."""
    response = CaptureResponse()
    response.set_status(HTTPStatus.CREATED)
    await response.prepare(make_mocked_request("GET", "/"))

    with pytest.raises(AssertionError):
        response.set_status(HTTPStatus.BAD_GATEWAY, "Bad Gateway")

    assert response.status == HTTPStatus.CREATED
    assert response.status_code == HTTPStatus.CREATED
    assert response.reason == "Created"


async def test_capture_keeps_mutated_headers_and_chunks() -> None:
    """A caller can read the changed headers and captured chunks."""
    response = CaptureResponse()

    def change_headers(headers: dict[str, str], _status: int) -> dict[str, str]:
        return {**headers, "X-Captured": "yes"}

    async def change_body(
        _headers: dict[str, str], chunk: bytes, _is_final: bool, _status: int,
    ) -> bytes:
        return chunk.upper()

    response.add_header_mutator(change_headers)
    response.add_mutator(change_body)
    assert response.prepared is False
    await response.prepare(make_mocked_request("GET", "/"))
    assert response.prepared is True
    assert response.headers["X-Captured"] == "yes"
    await response.write(b"first")
    await response.write_eof(b"last")
    await response.write_eof(b"ignored")
    assert [chunk async for chunk in response.iter_chunks()] == [b"FIRST", b"LAST"]


@pytest.mark.parametrize("is_closing", [False, True])
@pytest.mark.parametrize(
    "response_cls", [MutatingStreamResponse, MutatingWebSocketResponse]
)
async def test_prepare_cancels_when_client_transport_gone_or_closing(
    response_cls: type[MutatingStreamResponse] | type[MutatingWebSocketResponse],
    is_closing: bool,
) -> None:
    """Preparing a response on a closed or missing client transport cancels."""
    if is_closing:
        transport = Mock()
        transport.is_closing.return_value = True
        request = make_mocked_request("GET", "/", transport=transport)
    else:
        protocol = Mock()
        request = make_mocked_request("GET", "/", protocol=protocol)
        protocol.transport = None
    response = response_cls(identity_mutator)
    with pytest.raises(asyncio.CancelledError):
        await response.prepare(request)


async def test_stream_write_connection_error_becomes_cancellation() -> None:
    """A client connection error while writing a stream response cancels."""
    writer = Mock()
    writer.write_headers = AsyncMock()
    writer.write = AsyncMock(side_effect=ConnectionResetError("gone"))
    writer.write_eof = AsyncMock(side_effect=ConnectionResetError("gone"))
    request = make_mocked_request("GET", "/", writer=writer)
    response = MutatingStreamResponse(identity_mutator)
    await response.prepare(request)
    with pytest.raises(asyncio.CancelledError):
        await response.write(b"chunk")
    with pytest.raises(asyncio.CancelledError):
        await response.write_eof()


@pytest.mark.parametrize(
    "response_cls,headers",
    [
        (MutatingStreamResponse, {}),
        (MutatingWebSocketResponse, _WS_HEADERS),
    ],
)
async def test_prepare_connection_error_becomes_cancellation(
    response_cls: type[MutatingStreamResponse] | type[MutatingWebSocketResponse],
    headers: dict[str, str],
) -> None:
    """A client connection error while preparing a response cancels."""
    writer = Mock()
    writer.write_headers = AsyncMock(side_effect=ConnectionResetError("gone"))
    writer.drain = AsyncMock()
    transport = Mock()
    transport.is_closing.return_value = False
    request = make_mocked_request(
        "GET", "/", headers=headers, writer=writer, transport=transport
    )
    response = response_cls(identity_mutator)
    with pytest.raises(asyncio.CancelledError):
        await response.prepare(request)


@pytest.mark.parametrize(
    "send_fn",
    [
        pytest.param(_send_ws_text, id="send_str"),
        pytest.param(_send_ws_bytes, id="send_bytes"),
        pytest.param(_send_ws_json, id="send_json"),
    ],
)
async def test_websocket_send_connection_error_becomes_cancellation(
    send_fn: Callable[[MutatingWebSocketResponse], Awaitable[None]],
) -> None:
    """A client connection error while sending a WebSocket message cancels."""
    writer = Mock()
    writer.write_headers = AsyncMock()
    writer.drain = AsyncMock()
    transport = Mock()
    transport.is_closing.return_value = False
    transport.write = Mock(side_effect=ConnectionResetError("gone"))
    request = make_mocked_request(
        "GET", "/", headers=_WS_HEADERS, writer=writer, transport=transport
    )
    response = MutatingWebSocketResponse(identity_mutator)
    await response.prepare(request)
    with pytest.raises(asyncio.CancelledError):
        await send_fn(response)
