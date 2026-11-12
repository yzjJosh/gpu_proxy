"""Tests for captured response headers, status, and chunks."""

from http import HTTPStatus
import pytest
from aiohttp.test_utils import make_mocked_request
from app.responses import CaptureResponse

_UNKNOWN_STATUS = 599


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
