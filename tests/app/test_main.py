"""Tests for the app.main entry point."""

import asyncio
import sys
from pathlib import Path
from typing import override

import pytest
from aiohttp import web

from app.main import main


_EXPECTED_PORT = 18080
_SERVER_STARTED = asyncio.Event()


class _FakeServer:
    """A stand-in asyncio.AbstractServer so TCPSite construction succeeds."""


def _write_config(tmp_path: Path) -> str:
    """Write a minimal valid config file and return its path."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"listen_port: {_EXPECTED_PORT}\n"
        "total_vram_budget: 20480\n"
        "services:\n"
        "  llamacpp-server:\n"
        "    priority: 1\n"
        "    vram_budget: 16384\n"
        "    hostname: llamacpp.local\n"
        "    target: http://127.0.0.1:49863\n"
        "    container: llamacpp-server\n"
    )
    return str(cfg)


class _FakeRunner(web.AppRunner):
    """An aiohttp AppRunner whose setup is stubbed but still validates the app.

    The real AppRunner.setup() creates a listening socket, which the test does
    not want; the stub skips socket binding but keeps the catch-all-route check
    so main()'s wiring is still verified.
    """

    def __init__(self, app: web.Application, **kwargs: bool) -> None:
        assert kwargs.get("handler_cancellation") is True
        super().__init__(app, **kwargs)

    @override
    async def setup(self) -> None:
        # main() registers a catch-all route before building the runner.
        catch_all = [
            route.resource.canonical
            for route in self._app.router.routes()
            if route.method == "*"
        ]
        assert "/{path}" in catch_all
        # BaseSite.__init__ requires runner.server to be set after setup.
        self._server = _FakeServer()

    @override
    async def cleanup(self) -> None:
        # Skip the real server teardown (no socket bound); just run the
        # application cleanup hooks and clear state. setup() always sets
        # the fake server before the site is created.
        assert isinstance(self._server, _FakeServer)
        await self._cleanup_server()
        self._server = None


class _FakeSite(web.TCPSite):
    """An aiohttp TCPSite whose start is a no-op.

    The constructor keeps the real BaseSite.__init__ path, so an out-of-order
    setup (TCPSite before runner.setup) still raises as it would in production.
    """

    def __init__(self, runner: _FakeRunner, host: str, port: int) -> None:
        super().__init__(runner, host, port)

    @override
    async def start(self) -> None:
        assert self._host == "0.0.0.0"
        assert self._port == _EXPECTED_PORT
        catch_all = [
            route.resource.canonical
            for route in self._runner._app.router.routes()
            if route.method == "*"
        ]
        assert "/{path}" in catch_all
        _SERVER_STARTED.set()


@pytest.fixture
def fake_server(monkeypatch: pytest.MonkeyPatch) -> asyncio.Event:
    """Replace the external aiohttp server with a no-op so no socket binds.

    main() awaits AppRunner.setup(), TCPSite.start(), and then an infinite
    sleep. Replacing these with no-ops lets the test reach and cancel the
    infinite loop without binding a real port.

    Returns:
        An asyncio.Event set once TCPSite.start() has been reached, so a
        test can wait until main() is inside the server loop before it
        cancels — guaranteeing the cleanup path runs.
    """
    _SERVER_STARTED.clear()
    monkeypatch.setattr(web, "AppRunner", _FakeRunner)
    monkeypatch.setattr(web, "TCPSite", _FakeSite)
    yield _SERVER_STARTED


async def test_main_reaches_server_loop_and_cleans_up(
    tmp_path: Path, fake_server: asyncio.Event, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() starts the proxy and server loop, then shuts down cleanly."""
    config_path = _write_config(tmp_path)
    monkeypatch.setattr(sys, "argv", ["app.main", config_path])

    task = asyncio.create_task(main())
    try:
        await asyncio.wait_for(fake_server.wait(), timeout=5)
        assert not task.done()
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
