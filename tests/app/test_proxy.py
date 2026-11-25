"""End-to-end tests for the proxy: routing, eviction, dashboard, health, plugins.

Every test drives a real GpuPriorityProxy that listens on a real TCP port and is
reached with a real HTTP client, so requests travel the same path they do in
production. Targets and backups are real aiohttp servers, eviction and health
check commands are real subprocesses, and only the NVIDIA library — the one
dependency a test machine cannot be asked to provide — is replaced by a fake.
"""

from __future__ import annotations

import asyncio
import json
import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable, Iterator, Optional, override

import pytest
from aiohttp import (
    ClientResponse,
    ClientSession,
    ClientTimeout,
    ClientWebSocketResponse,
    ClientWSTimeout,
    ServerDisconnectedError,
    WSMsgType,
    web,
)

from app.config import ProxyConfig, ServiceConfig
from app.gpu_monitor import RealClock
from app.plugins.base import PluginBase
from app.plugins.manager import PluginManager
from app.plugins.openai_guard_rail import OpenAIGuardRail
from app.plugins.openai_token_stats import OpenAITokenStats
from app.proxy import GpuPriorityProxy
from app.service_state import ServiceStateEnum
from tests.app.conftest import (
    BodyAppenderPlugin,
    CacheForwardPlugin,
    CancelTrackingPlugin,
    CancelledForwardPlugin,
    ChunkCapturePlugin,
    ContentLengthPlugin,
    CustomResponsePlugin,
    DelegatePlugin,
    FailingPlugin,
    FakeGpuProcess,
    FakeNvidiaHost,
    ForwardToBackupPlugin,
    GpuDelegatePlugin,
    HeaderPlugin,
    MutatorPlugin,
    PrepareTargetRecorder,
    UIPublisherPlugin,
    extract_sse_field,
    sse_delta_content,
    sse_done,
    sse_event,
    sse_usage,
)

LOCALHOST = "127.0.0.1"
DASHBOARD_HOST = "gpu-proxy.local"
DASHBOARD_ASSETS = Path(__file__).resolve().parents[2] / "dashboard" / "dist" / "assets"

HTTP_SWITCHING_PROTOCOLS = 101
HTTP_OK = 200
HTTP_CREATED = 201
HTTP_FOUND = 302
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
HTTP_METHOD_NOT_ALLOWED = 405
HTTP_CONTENT_TOO_LARGE = 413
HTTP_INTERNAL_SERVER_ERROR = 500
HTTP_BAD_GATEWAY = 502
HTTP_SERVICE_UNAVAILABLE = 503
HTTP_LOOP_DETECTED = 508

TOTAL_VRAM = 24576
FIRST_CONTAINER_PID = 4000
VRAM_BUDGET = 20480
LARGE_SERVICE_VRAM = 16384
SMALL_SERVICE_VRAM = 4096

POLL_INTERVAL = 0.02
EVICTION_TIMEOUT = 0.3
COMMAND_TIMEOUT = 2.0
QUEUE_TIMEOUT = 2.0
CONNECT_TIMEOUT = 2.0
CLIENT_TIMEOUT = 10.0
WS_RECEIVE_TIMEOUT = 5.0
SHUTDOWN_TIMEOUT = 0.5
SETTLE_TIMEOUT = 3.0
TICK = 0.01

SMALL_LOAD_MIB = 900
OTHER_LOAD_MIB = 700
EVICTABLE_LOAD_MIB = 2000
CROWDED_LOAD_MIB = 9000
BLOCKING_LOAD_MIB = 18000
HOGGED_LOAD_MIB = 22000
PROBE_LOAD_MIB = 4000
PROBE_FREE_MIB = 19000

SLOW_RESPONSE = 0.3
QUICK_RESPONSE = 0.1
STALLED_RESPONSE = 5.0
PATIENT_EVICTION = 5.0
IMPATIENT_EVICTION = 0.1
COMMAND_KILL_TIMEOUT = 0.2
BRIEF_QUEUE_WAIT = 0.3
IMPATIENT_QUEUE_WAIT = 0.1
SLOW_CHUNK_GAP = 0.2
BRIEF_CHUNK_GAP = 0.05
UPTIME_STEP = 1.1

NOOP_COMMAND = "/bin/true {container}"
WS_UPGRADE_HEADERS = {"Upgrade": "websocket", "Connection": "Upgrade"}


@pytest.fixture(autouse=True)
def mock_subprocess_exec() -> None:
    """Run eviction commands as real subprocesses instead of the shared fake.

    Overrides the autouse fixture of the same name in conftest so that these
    tests observe what a command actually did on disk.
    """
    return None


def free_port() -> int:
    """Reserve and release a loopback port, returning its number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((LOCALHOST, 0))
        return int(sock.getsockname()[1])


@dataclass
class RecordedRequest:
    """A request as the upstream server received it."""

    method: str
    path: str
    headers: dict[str, str]
    body: bytes


class UpstreamServer:
    """A real HTTP and WebSocket server standing in for a proxied upstream."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.requests: list[RecordedRequest] = []
        self.status: int = HTTP_OK
        self.body: bytes = b"upstream-ok"
        self.content_type: str = "text/plain"
        self.response_headers: dict[str, str] = {}
        self.chunks: list[bytes] = []
        self.chunk_gap: float = 0.0
        self.response_delay: float = 0.0
        self.reject_websocket: bool = False
        self.ws_greeting: Optional[str] = None
        self.ws_close_after: Optional[int] = None
        self.port: int = 0
        self._runner: Optional[web.AppRunner] = None

    @property
    def url(self) -> str:
        """Base URL a service config points at to reach this server."""
        return f"http://{LOCALHOST}:{self.port}"

    @property
    def last_request(self) -> RecordedRequest:
        """The most recently received request."""
        return self.requests[-1]

    async def start(self) -> None:
        """Bind the server to a free loopback port."""
        self.port = free_port()
        app = web.Application()
        app.router.add_route("*", "/{path:.*}", self._handle)
        self._runner = web.AppRunner(app, shutdown_timeout=SHUTDOWN_TIMEOUT)
        await self._runner.setup()
        await web.TCPSite(self._runner, LOCALHOST, self.port).start()

    async def stop(self) -> None:
        """Shut the server down."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        """Record the request and answer it as the test has configured."""
        body = await request.read()
        self.requests.append(
            RecordedRequest(request.method, request.path_qs, dict(request.headers), body)
        )
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return await self._handle_websocket(request)
        if self.response_delay:
            await asyncio.sleep(self.response_delay)
        if self.chunks:
            return await self._stream_chunks(request)
        return web.Response(
            status=self.status, body=self.body, content_type=self.content_type,
            headers=self.response_headers,
        )

    async def _stream_chunks(self, request: web.Request) -> web.StreamResponse:
        """Write the configured chunks as a streaming response."""
        response = web.StreamResponse(status=self.status)
        response.content_type = "text/event-stream"
        response.headers.update(self.response_headers)
        await response.prepare(request)
        for chunk in self.chunks:
            await response.write(chunk)
            if self.chunk_gap:
                await asyncio.sleep(self.chunk_gap)
        await response.write_eof()
        return response

    async def _handle_websocket(self, request: web.Request) -> web.StreamResponse:
        """Accept a WebSocket and echo every message back with a prefix."""
        if self.reject_websocket:
            return web.Response(status=HTTP_SERVICE_UNAVAILABLE, reason="No WebSocket")
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        if self.ws_greeting is not None:
            await ws.send_str(self.ws_greeting)
        echoed = 0
        async for msg in ws:
            if msg.type is WSMsgType.BINARY:
                await ws.send_bytes(b"echo:" + msg.data)
            elif msg.type is WSMsgType.TEXT:
                await ws.send_str(f"{self.name}:{msg.data}")
            else:
                break
            echoed += 1
            if self.ws_close_after is not None and echoed >= self.ws_close_after:
                break
        return ws


@dataclass
class Reply:
    """A completed response as the proxy's client saw it."""

    status: int
    body: bytes
    headers: dict[str, str]
    reason: str = ""

    @property
    def text(self) -> str:
        """Response body decoded as UTF-8."""
        return self.body.decode()


class FakeGpu(FakeNvidiaHost):
    """Scriptable GPU host whose containers hold VRAM until a test frees it."""

    def __init__(self) -> None:
        super().__init__(total_vram_mib=TOTAL_VRAM)
        self.available_vram_mib: int = TOTAL_VRAM
        self.containers: dict[str, int] = {}
        self.freed_by: dict[str, Path] = {}

    def occupy(
        self, containers: dict[str, int], available_vram_mib: Optional[int] = None
    ) -> None:
        """Report the given containers holding VRAM.

        Free VRAM defaults to whatever the containers leave behind, and can be
        overridden for tests that care about a specific reading.
        """
        self.containers = dict(containers)
        if available_vram_mib is None:
            available_vram_mib = TOTAL_VRAM - sum(containers.values())
        self.available_vram_mib = available_vram_mib

    def release_all(self, available_vram_mib: int = TOTAL_VRAM) -> None:
        """Report that no container holds VRAM any more."""
        self.containers = {}
        self.freed_by = {}
        self.available_vram_mib = available_vram_mib

    def release_when_touched(self, container: str, marker: Path) -> None:
        """Free a container's VRAM once the given marker file appears.

        Lets an eviction command that touches the marker have the same effect a
        real restart has: the container's VRAM comes back to the pool.
        """
        self.freed_by[container] = marker

    def _held(self) -> dict[str, int]:
        """Containers still holding VRAM, after applying released markers."""
        return {
            name: vram
            for name, vram in self.containers.items()
            if name not in self.freed_by or not self.freed_by[name].exists()
        }

    @override
    def current_used_vram_mib(self) -> int:
        """Used VRAM, counting back anything released by a marker."""
        held = self._held()
        returned = sum(v for k, v in self.containers.items() if k not in held)
        return TOTAL_VRAM - self.available_vram_mib - returned

    @override
    def current_processes(self) -> list[FakeGpuProcess]:
        """One GPU process for each container still holding VRAM."""
        return [
            FakeGpuProcess.from_mib(
                pid=FIRST_CONTAINER_PID + index,
                vram_mib=vram,
                container=name,
                sm_utilization_pct=self.gpu_util_pct,
            )
            for index, (name, vram) in enumerate(self._held().items())
        ]


class ProxyHarness:
    """A started proxy plus a client that speaks to it over the network."""

    def __init__(self, proxy: GpuPriorityProxy, port: int, gpu: FakeGpu) -> None:
        self.proxy = proxy
        self.port = port
        self.gpu = gpu
        self._runner: Optional[web.AppRunner] = None
        self._session: Optional[ClientSession] = None

    @property
    def base_url(self) -> str:
        """URL the proxy listens on."""
        return f"http://{LOCALHOST}:{self.port}"

    @property
    def session(self) -> ClientSession:
        """Client session bound to the running proxy."""
        if self._session is None:
            raise RuntimeError("the harness has not been started")
        return self._session

    async def start(self) -> None:
        """Start the GPU monitor, serve the proxy, and open a client session."""
        await self.proxy.start()
        app = web.Application()
        app.router.add_route("*", "/{path:.*}", self.proxy.proxy_handler)
        self._runner = web.AppRunner(
            app,
            shutdown_timeout=SHUTDOWN_TIMEOUT,
            handler_cancellation=True,
        )
        await self._runner.setup()
        await web.TCPSite(self._runner, LOCALHOST, self.port).start()
        self._session = ClientSession(timeout=ClientTimeout(total=CLIENT_TIMEOUT))

    async def stop_serving(self) -> None:
        """Stop accepting requests, cancelling anything still in flight."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def stop(self) -> None:
        """Close the client, stop serving, and stop the proxy."""
        if self._session is not None:
            await self._session.close()
            self._session = None
        await self.stop_serving()
        await self.proxy.stop()

    async def send(
        self,
        path: str,
        host: str,
        method: str = "GET",
        body: Optional[bytes] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> Reply:
        """Send a request through the proxy and read the whole response."""
        sent = {"Host": host}
        sent.update(headers or {})
        async with self.session.request(
            method, f"{self.base_url}{path}", headers=sent, data=body
        ) as response:
            return Reply(
                status=response.status,
                body=await response.read(),
                headers={k: v for k, v in response.headers.items()},
                reason=response.reason or "",
            )

    async def send_chunked(
        self,
        path: str,
        host: str,
        method: str = "POST",
        chunks: Optional[list[bytes]] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> Reply:
        """Send a chunked-framed request through the proxy over a raw socket."""
        head = [f"{method} {path} HTTP/1.1", f"Host: {host}"]
        head.extend(f"{key}: {value}" for key, value in (headers or {}).items())
        head.extend(("Transfer-Encoding: chunked", "Connection: close"))
        raw = b"\r\n".join(line.encode() for line in head) + b"\r\n\r\n"
        for chunk in chunks or [b""]:
            raw += f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n"
        raw += b"0\r\n\r\n"
        reader, writer = await asyncio.open_connection(LOCALHOST, self.port)
        try:
            writer.write(raw)
            await writer.drain()
            payload = await asyncio.wait_for(reader.read(), timeout=CLIENT_TIMEOUT)
        finally:
            writer.close()
            await writer.wait_closed()
        head_block, _, body = payload.partition(b"\r\n\r\n")
        status_line, _, header_block = head_block.partition(b"\r\n")
        _, status_code, reason = status_line.decode().split(" ", 2)
        response_headers = {}
        for line in header_block.split(b"\r\n"):
            key, _, value = line.decode().partition(":")
            response_headers[key.strip()] = value.strip()
        return Reply(int(status_code), body, response_headers, reason)

    @asynccontextmanager
    async def stream(
        self, path: str, host: str, method: str = "GET"
    ) -> AsyncIterator[ClientResponse]:
        """Open a response through the proxy without draining its body."""
        async with self.session.request(
            method, f"{self.base_url}{path}", headers={"Host": host}
        ) as response:
            yield response

    @asynccontextmanager
    async def websocket(self, path: str, host: str) -> AsyncIterator[ClientWebSocketResponse]:
        """Open a WebSocket through the proxy."""
        async with self.session.ws_connect(
            f"{self.base_url}{path}",
            headers={"Host": host},
            timeout=ClientWSTimeout(ws_receive=WS_RECEIVE_TIMEOUT),
        ) as ws:
            yield ws

    async def dashboard(self) -> dict:
        """Fetch the dashboard JSON snapshot."""
        reply = await self.send("/?format=json", DASHBOARD_HOST)
        return json.loads(reply.text)

    async def service(self, name: str) -> dict:
        """Fetch one service's dashboard entry."""
        data = await self.dashboard()
        return data["services"][name]

    async def wait_for_dashboard(
        self, predicate: Callable[[dict], bool], timeout: float = SETTLE_TIMEOUT
    ) -> dict:
        """Poll the dashboard until the predicate holds, returning that snapshot.

        Fails the test when the predicate is still false at the deadline, so that
        a wait which never settles cannot be mistaken for a settled one.
        """
        deadline = asyncio.get_running_loop().time() + timeout
        data = await self.dashboard()
        while not predicate(data):
            if asyncio.get_running_loop().time() > deadline:
                pytest.fail(f"dashboard never settled within {timeout}s: {data}")
            await asyncio.sleep(TICK)
            data = await self.dashboard()
        return data

    async def wait_for_poll(self, timeout: float = SETTLE_TIMEOUT) -> str:
        """Wait for the first successful GPU poll, returning the reported health."""
        data = await self.wait_for_dashboard(
            lambda snapshot: snapshot["proxy"]["last_poll"] is not None, timeout
        )
        return data["proxy"]["health"]

    async def wait_for_state(
        self, name: str, expected: str, timeout: float = SETTLE_TIMEOUT
    ) -> str:
        """Wait for a service to reach a state, returning the state actually seen."""
        data = await self.wait_for_dashboard(
            lambda snapshot: snapshot["services"][name]["state"] == expected, timeout
        )
        return data["services"][name]["state"]

    async def wait_for_in_flight(
        self, name: str, expected: int, timeout: float = SETTLE_TIMEOUT
    ) -> int:
        """Wait for a service's in-flight count, returning the count actually seen."""
        data = await self.wait_for_dashboard(
            lambda snapshot: snapshot["services"][name]["in_flight"] == expected, timeout
        )
        return data["services"][name]["in_flight"]

    async def wait_for_health(
        self, name: str, expected: bool, timeout: float = SETTLE_TIMEOUT
    ) -> bool:
        """Wait for a service's health flag, returning the value actually seen."""
        data = await self.wait_for_dashboard(
            lambda snapshot: snapshot["services"][name]["is_healthy"] == expected, timeout
        )
        return data["services"][name]["is_healthy"]

    async def events(
        self, event: Optional[str] = None, name: Optional[str] = None
    ) -> list[dict]:
        """Return eviction-log entries, optionally filtered by event and service."""
        data = await self.dashboard()
        return select_events(data["eviction_log"], event, name)

    async def wait_for_events(
        self,
        event: str,
        name: Optional[str] = None,
        count: int = 1,
        timeout: float = SETTLE_TIMEOUT,
    ) -> list[dict]:
        """Wait until the eviction log holds `count` matching entries, then return them."""
        data = await self.wait_for_dashboard(
            lambda snapshot: len(select_events(snapshot["eviction_log"], event, name))
            >= count,
            timeout,
        )
        return select_events(data["eviction_log"], event, name)

    async def routing(self, name: str) -> dict:
        """Return the routing counters for one service."""
        data = await self.dashboard()
        return data["routing_stats"][name]

    async def wait_for_routing(
        self, name: str, key: str, expected: int, timeout: float = SETTLE_TIMEOUT
    ) -> int:
        """Wait for a routing counter to reach a value, returning the value seen."""
        data = await self.wait_for_dashboard(
            lambda snapshot: snapshot["routing_stats"][name][key] == expected, timeout
        )
        return data["routing_stats"][name][key]


def select_events(
    entries: list[dict], event: Optional[str] = None, name: Optional[str] = None
) -> list[dict]:
    """Filter eviction-log entries by event type and service name."""
    chosen = entries
    if event is not None:
        chosen = [e for e in chosen if e["event"] == event]
    if name is not None:
        chosen = [e for e in chosen if e["service"] == name]
    return chosen


async def wait_until(
    predicate: Callable[[], bool], timeout: float = SETTLE_TIMEOUT
) -> None:
    """Wait for a condition on the test's own state to become true.

    Fails the test at the deadline so that a condition which never held cannot be
    mistaken for one that did.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            pytest.fail(f"condition never became true within {timeout}s")
        await asyncio.sleep(TICK)


def make_service(
    name: str,
    target: str,
    priority: int = 1,
    vram_budget: int = SMALL_SERVICE_VRAM,
) -> ServiceConfig:
    """Build a service reachable at <name>.local with a harmless eviction command."""
    return ServiceConfig(
        name=name,
        priority=priority,
        vram_budget=vram_budget,
        hostname=f"{name}.local",
        target=target,
        container=f"{name}-container",
        eviction_command=NOOP_COMMAND,
    )


def make_config(
    services: list[ServiceConfig],
    listen_port: int,
    eviction_timeout: int = EVICTION_TIMEOUT,
    eviction_command_timeout: int = COMMAND_TIMEOUT,
    wait_queue_timeout: int = QUEUE_TIMEOUT,
    plugins: Optional[list[dict]] = None,
) -> ProxyConfig:
    """Build a proxy config that serves the given services on the given port."""
    settings = {
        "listen_port": listen_port,
        "total_vram_budget": VRAM_BUDGET,
        "poll_interval": POLL_INTERVAL,
        "eviction_timeout": eviction_timeout,
        "eviction_command_timeout": eviction_command_timeout,
        "wait_queue_timeout": wait_queue_timeout,
        "connection_timeout": CONNECT_TIMEOUT,
        "dashboard_hostname": DASHBOARD_HOST,
        "services": {svc.name: svc for svc in services},
        "hostname_map": {svc.hostname: svc for svc in services},
    }
    if plugins is not None:
        settings["plugins"] = plugins
    return ProxyConfig(**settings)


StartProxy = Callable[..., Awaitable[ProxyHarness]]
StartUpstream = Callable[..., Awaitable[UpstreamServer]]


@pytest.fixture
def fake_gpu() -> Iterator[FakeGpu]:
    """Replace the NVIDIA library the GPU monitor polls with a scriptable fake."""
    gpu = FakeGpu()
    with gpu.installed():
        yield gpu


@pytest.fixture
async def start_upstream() -> AsyncIterator[StartUpstream]:
    """Factory that starts upstream servers and stops them at teardown."""
    servers: list[UpstreamServer] = []

    async def _start(name: str = "upstream") -> UpstreamServer:
        server = UpstreamServer(name)
        await server.start()
        servers.append(server)
        return server

    yield _start
    for server in servers:
        await server.stop()


@pytest.fixture
async def start_proxy(fake_gpu: FakeGpu) -> AsyncIterator[StartProxy]:
    """Factory that starts proxies against the fake GPU and stops them at teardown."""
    harnesses: list[ProxyHarness] = []

    async def _start(
        config: ProxyConfig, plugin_manager: Optional[PluginManager] = None
    ) -> ProxyHarness:
        proxy = GpuPriorityProxy(
            config, clock=RealClock(), plugin_manager=plugin_manager
        )
        harness = ProxyHarness(proxy, config.listen_port, fake_gpu)
        await harness.start()
        harnesses.append(harness)
        return harness

    yield _start
    for harness in harnesses:
        await harness.stop()


@pytest.fixture
async def target(start_upstream: StartUpstream) -> UpstreamServer:
    """Upstream server used as a service target."""
    return await start_upstream("target")


@pytest.fixture
async def backup(start_upstream: StartUpstream) -> UpstreamServer:
    """Upstream server used as a service backup."""
    return await start_upstream("backup")


@pytest.fixture
async def gpu_service(
    start_proxy: StartProxy, target: UpstreamServer, backup: UpstreamServer
) -> ProxyHarness:
    """Proxy with one service whose /v1/completions prefix is a GPU route."""
    svc = replace(
        make_service("svc", target.url, vram_budget=SMALL_SERVICE_VRAM),
        gpu_routes=["/v1/completions"],
        backup=backup.url,
    )
    harness = await start_proxy(make_config([svc], free_port()))
    await harness.wait_for_poll()
    return harness


class TestWebSocketUpgradeDetection:
    """RFC 6455 upgrade detection on raw header dicts."""

    @pytest.mark.parametrize(
        "headers,expected",
        [
            ({"Upgrade": "websocket", "Connection": "Upgrade"}, True),
            ({"upgrade": "websocket", "connection": "upgrade"}, True),
            ({"Upgrade": "WebSocket", "Connection": "UPGRADE"}, True),
            ({"Upgrade": "websocket", "Connection": ["keep-alive", "Upgrade"]}, True),
            ({"Upgrade": "h2c", "Connection": "upgrade"}, False),
            ({"Upgrade": "websocket"}, False),
            ({"Connection": "Upgrade"}, False),
            ({}, False),
            (None, False),
        ],
    )
    def test_upgrade_requires_both_headers(
        self, headers: Optional[dict], expected: bool
    ) -> None:
        """Only Upgrade: websocket together with Connection: Upgrade counts."""
        assert GpuPriorityProxy.is_websocket_upgrade(headers) is expected


class TestHostRouting:
    """Demultiplexing incoming requests by the Host header."""

    async def test_unknown_host_returns_404(self, gpu_service: ProxyHarness) -> None:
        """A host that maps to no service is refused."""
        reply = await gpu_service.send("/v1/models", "nobody.local")
        assert reply.status == HTTP_NOT_FOUND
        assert reply.reason == "Unknown host"

    async def test_empty_host_returns_404(self, gpu_service: ProxyHarness) -> None:
        """A blank Host header cannot be routed."""
        reply = await gpu_service.send("/v1/models", "")
        assert reply.status == HTTP_NOT_FOUND

    async def test_service_host_port_is_ignored(
        self, gpu_service: ProxyHarness, target: UpstreamServer
    ) -> None:
        """A Host carrying a port still matches the service hostname."""
        reply = await gpu_service.send("/v1/models", "svc.local:8080")
        assert reply.status == HTTP_OK
        assert target.last_request.path == "/v1/models"

    async def test_dashboard_host_port_is_ignored(
        self, gpu_service: ProxyHarness
    ) -> None:
        """A Host carrying a port still selects the dashboard."""
        reply = await gpu_service.send("/", f"{DASHBOARD_HOST}:8080")
        assert reply.status == HTTP_OK
        assert '<div id="app"></div>' in reply.text

    async def test_health_host_port_is_ignored(self, gpu_service: ProxyHarness) -> None:
        """The health endpoint is reachable on the dashboard host with a port."""
        reply = await gpu_service.send("/health", f"{DASHBOARD_HOST}:8080")
        assert json.loads(reply.text)["status"] == "ok"


class TestDashboardPage:
    """The dashboard hostname serves the built single-page app and its assets."""

    async def test_serves_single_page_app(self, gpu_service: ProxyHarness) -> None:
        """The dashboard root returns the Vue shell as HTML."""
        reply = await gpu_service.send("/", DASHBOARD_HOST)
        assert reply.status == HTTP_OK
        assert reply.headers["Content-Type"].startswith("text/html")
        assert '<div id="app"></div>' in reply.text

    async def test_shell_is_mobile_ready(self, gpu_service: ProxyHarness) -> None:
        """The shell declares a viewport so the page scales on phones."""
        reply = await gpu_service.send("/", DASHBOARD_HOST)
        assert 'name="viewport"' in reply.text
        assert "width=device-width" in reply.text

    @pytest.mark.parametrize("method", ["POST", "PUT", "DELETE"])
    async def test_rejects_write_methods(
        self, gpu_service: ProxyHarness, method: str
    ) -> None:
        """Only GET is allowed on the dashboard hostname."""
        reply = await gpu_service.send("/", DASHBOARD_HOST, method)
        assert reply.status == HTTP_METHOD_NOT_ALLOWED
        assert reply.reason == "Method Not Allowed"

    async def test_serves_built_asset(self, gpu_service: ProxyHarness) -> None:
        """A file under dashboard/dist/assets is served with its media type."""
        asset = sorted(DASHBOARD_ASSETS.glob("*.css"))[0]
        reply = await gpu_service.send(f"/dashboard/assets/{asset.name}", DASHBOARD_HOST)
        assert reply.status == HTTP_OK
        assert reply.headers["Content-Type"].startswith("text/css")
        assert reply.body == asset.read_bytes()

    async def test_unknown_asset_returns_404(self, gpu_service: ProxyHarness) -> None:
        """A missing asset is reported as not found."""
        reply = await gpu_service.send("/dashboard/assets/nope.css", DASHBOARD_HOST)
        assert reply.status == HTTP_NOT_FOUND
        assert reply.reason == "Not found"


class TestDashboardData:
    """The JSON snapshot the dashboard app renders."""

    async def test_reports_gpu_telemetry(
        self, gpu_service: ProxyHarness, fake_gpu: FakeGpu
    ) -> None:
        """Every GPU field polled from the library reaches the snapshot."""
        fake_gpu.gpu_util_pct = 75
        fake_gpu.power_draw_w = 248.5
        fake_gpu.power_limit_w = 250.0
        fake_gpu.temperature_c = 70
        fake_gpu.occupy({"svc-container": PROBE_LOAD_MIB}, PROBE_FREE_MIB)
        data = await gpu_service.wait_for_dashboard(
            lambda snapshot: snapshot["gpu"]["gpu_util_pct"] == 75
        )
        assert data["gpu"]["power_usage_w"] == 248.5
        assert data["gpu"]["power_limit_w"] == 250.0
        assert data["gpu"]["temperature_c"] == 70
        assert data["gpu"]["total_vram_mib"] == TOTAL_VRAM
        assert data["gpu"]["available_vram_mib"] == PROBE_FREE_MIB
        assert data["gpu"]["docker_vram_mib"] == PROBE_LOAD_MIB
        assert data["gpu"]["system_vram_mib"] == TOTAL_VRAM - PROBE_FREE_MIB - PROBE_LOAD_MIB
        assert data["gpu"]["containers"]["svc-container"]["vram_mib"] == PROBE_LOAD_MIB

    async def test_reports_budget(self, gpu_service: ProxyHarness) -> None:
        """The effective budget is capped by the configured total, not the card."""
        gpu_service.gpu.release_all(TOTAL_VRAM)
        data = await gpu_service.wait_for_dashboard(
            lambda snapshot: snapshot["gpu"]["available_vram_mib"] == TOTAL_VRAM
        )
        assert data["gpu"]["effective_budget_mib"] == VRAM_BUDGET
        assert data["gpu"]["remaining_budget_mib"] == VRAM_BUDGET

    async def test_reports_service_configuration(
        self, gpu_service: ProxyHarness
    ) -> None:
        """Each service publishes its state, priority, budget and counters."""
        entry = await gpu_service.service("svc")
        assert entry["state"] == "cold"
        assert entry["priority"] == 1
        assert entry["vram_budget_mib"] == SMALL_SERVICE_VRAM
        assert entry["vram_used_mib"] == 0
        assert entry["in_flight"] == 0
        assert entry["is_healthy"] is True
        assert entry["external"] is False
        assert entry["routing"] == {"target": 0, "backup": 0}

    async def test_reports_external_service(
        self, start_proxy: StartProxy, target: UpstreamServer
    ) -> None:
        """A service declared external is flagged as such."""
        external = replace(
            make_service("ext", target.url), external=True, vram_budget=0, container=""
        )
        harness = await start_proxy(make_config([external], free_port()))
        entry = await harness.service("ext")
        assert entry["external"] is True
        assert entry["vram_budget_mib"] == 0

    async def test_snapshot_has_every_section(self, gpu_service: ProxyHarness) -> None:
        """The snapshot carries all sections the dashboard app reads."""
        data = await gpu_service.dashboard()
        assert set(data) == {
            "proxy",
            "gpu",
            "services",
            "eviction_log",
            "quiet_eviction_log",
            "routing_stats",
        }
        assert set(data["proxy"]) == {"uptime_s", "last_poll", "health"}
        assert data["routing_stats"]["svc"] == {"target": 0, "backup": 0, "evictions": 0}

    async def test_repeated_snapshots_agree(self, gpu_service: ProxyHarness) -> None:
        """Two snapshots taken back to back describe the same state."""
        first = await gpu_service.dashboard()
        second = await gpu_service.dashboard()
        assert first["services"] == second["services"]
        assert first["eviction_log"] == second["eviction_log"]

    async def test_quiet_log_omits_routine_approvals(
        self, gpu_service: ProxyHarness
    ) -> None:
        """An approved request is kept in the full log but filtered out of the quiet one."""
        reply = await gpu_service.send("/v1/completions", "svc.local")
        assert reply.status == HTTP_OK
        data = await gpu_service.dashboard()
        assert len(select_events(data["eviction_log"], "request_approved", "svc")) == 1
        assert select_events(data["quiet_eviction_log"], "request_approved", "svc") == []
        quiet = {(e["event"], e["service"], e["details"]) for e in data["quiet_eviction_log"]}
        full = {(e["event"], e["service"], e["details"]) for e in data["eviction_log"]}
        assert quiet.issubset(full)

    async def test_state_change_is_logged_once(
        self, gpu_service: ProxyHarness, fake_gpu: FakeGpu
    ) -> None:
        """A cold to warm transition produces exactly one log entry."""
        fake_gpu.occupy({"svc-container": SMALL_LOAD_MIB})
        assert await gpu_service.wait_for_state("svc", "warm") == "warm"
        entries = await gpu_service.events("state_transition", "svc")
        assert len(entries) == 1
        assert entries[0]["details"] == "cold → warm (VRAM detected)"

    async def test_websocket_pushes_updates(
        self, gpu_service: ProxyHarness, fake_gpu: FakeGpu
    ) -> None:
        """A dashboard WebSocket receives the initial state and later changes."""
        async with gpu_service.websocket("/ws", DASHBOARD_HOST) as ws:
            initial = json.loads((await ws.receive()).data)
            assert initial["type"] == "initial"
            assert initial["data"]["services"]["svc"]["state"] == "cold"
            fake_gpu.occupy({"svc-container": OTHER_LOAD_MIB})
            update = await wait_for_push(ws, "warm")
            assert update["data"]["services"]["svc"]["state"] == "warm"
            assert update["data"]["gpu"]["containers"]["svc-container"]["vram_mib"] == OTHER_LOAD_MIB


async def wait_for_push(ws: ClientWebSocketResponse, state: str) -> dict:
    """Read dashboard pushes until one reports the given state for svc."""
    while True:
        message = json.loads((await ws.receive()).data)
        if message["data"]["services"]["svc"]["state"] == state:
            return message


class TestHealthEndpoint:
    """The /health endpoint on the dashboard hostname."""

    async def test_reports_ok_after_a_successful_poll(
        self, gpu_service: ProxyHarness
    ) -> None:
        """A proxy that just started answers 200 with an ok status and no uptime yet."""
        reply = await gpu_service.send("/health", DASHBOARD_HOST)
        body = json.loads(reply.text)
        assert reply.status == HTTP_OK
        assert body["status"] == "ok"
        assert body["uptime_s"] == 0

    async def test_reports_unhealthy_once_polling_starts_failing(
        self, gpu_service: ProxyHarness, fake_gpu: FakeGpu
    ) -> None:
        """A failing poll discards the cached reading, so health drops to unhealthy."""
        fake_gpu.poll_fails = True
        data = await gpu_service.wait_for_dashboard(
            lambda snapshot: snapshot["proxy"]["health"] == "unhealthy"
        )
        assert data["proxy"]["last_poll"] is None
        reply = await gpu_service.send("/health", DASHBOARD_HOST)
        body = json.loads(reply.text)
        assert reply.status == HTTP_SERVICE_UNAVAILABLE
        assert body["status"] == "unhealthy"
        assert body["details"] == "no_gpu_data"

    async def test_reports_unhealthy_without_any_gpu_data(
        self, start_proxy: StartProxy, target: UpstreamServer, fake_gpu: FakeGpu
    ) -> None:
        """A proxy that never polled successfully answers 503."""
        fake_gpu.poll_fails = True
        harness = await start_proxy(make_config([make_service("svc", target.url)], free_port()))
        reply = await harness.send("/health", DASHBOARD_HOST)
        body = json.loads(reply.text)
        assert reply.status == HTTP_SERVICE_UNAVAILABLE
        assert body["status"] == "unhealthy"
        assert body["details"] == "no_gpu_data"

    async def test_poll_failure_is_recorded_in_the_log(
        self, gpu_service: ProxyHarness, fake_gpu: FakeGpu
    ) -> None:
        """A failed poll appends a gpu_poll_failed entry."""
        fake_gpu.poll_fails = True
        entries = await gpu_service.wait_for_events("gpu_poll_failed")
        assert entries[0]["service"] == "GPU poll failed"

    async def test_rejects_post(self, gpu_service: ProxyHarness) -> None:
        """The health endpoint is GET only."""
        reply = await gpu_service.send("/health", DASHBOARD_HOST, "POST")
        assert reply.status == HTTP_METHOD_NOT_ALLOWED

    async def test_service_host_health_goes_to_the_service(
        self, gpu_service: ProxyHarness, target: UpstreamServer
    ) -> None:
        """/health on a service hostname is proxied, not answered locally."""
        target.body = b"service-health"
        reply = await gpu_service.send("/health", "svc.local")
        assert reply.text == "service-health"
        assert target.last_request.path == "/health"

    async def test_uptime_counts_from_start(self, gpu_service: ProxyHarness) -> None:
        """Uptime is reported as whole seconds and grows while the proxy runs."""
        first = await gpu_service.send("/health", DASHBOARD_HOST)
        await asyncio.sleep(UPTIME_STEP)
        second = await gpu_service.send("/health", DASHBOARD_HOST)
        assert json.loads(first.text)["uptime_s"] < json.loads(second.text)["uptime_s"]


class TestPassthroughForwarding:
    """Requests on non-GPU routes are forwarded to the target as they arrived."""

    @pytest.mark.parametrize(
        "path",
        ["/v1/models", "/v1/models?stream=1&n=2", "/deep/path/segment", "/"],
    )
    async def test_path_and_query_are_preserved(
        self, gpu_service: ProxyHarness, target: UpstreamServer, path: str
    ) -> None:
        """The full path and query string arrive at the target unchanged."""
        reply = await gpu_service.send(path, "svc.local")
        assert reply.status == HTTP_OK
        assert target.last_request.path == path

    @pytest.mark.parametrize("method", ["GET", "POST", "PUT", "DELETE"])
    async def test_method_and_body_are_preserved(
        self, gpu_service: ProxyHarness, target: UpstreamServer, method: str
    ) -> None:
        """The method and request body arrive at the target unchanged."""
        await gpu_service.send("/echo", "svc.local", method, b'{"q":"hi"}')
        assert target.last_request.method == method
        assert target.last_request.body == b'{"q":"hi"}'

    async def test_response_is_relayed_verbatim(
        self, gpu_service: ProxyHarness, target: UpstreamServer
    ) -> None:
        """Body and content type from the target reach the client untouched."""
        target.body = b'{"models":["a"]}'
        target.content_type = "application/json"
        reply = await gpu_service.send("/v1/models", "svc.local")
        assert reply.status == HTTP_OK
        assert reply.body == b'{"models":["a"]}'
        assert reply.headers["Content-Type"] == "application/json"

    async def test_request_headers_are_forwarded(
        self, gpu_service: ProxyHarness, target: UpstreamServer
    ) -> None:
        """Client headers are passed through to the target."""
        await gpu_service.send(
            "/v1/models", "svc.local", headers={"Authorization": "Bearer k"}
        )
        assert target.last_request.headers["Authorization"] == "Bearer k"

    async def test_chunked_multipart_body_is_relayed_intact(
        self, gpu_service: ProxyHarness, target: UpstreamServer
    ) -> None:
        """A chunked-framed multipart body reaches the target intact."""
        head_part = (
            b'--b\r\nContent-Disposition: form-data; name="image"; '
            b'filename="p.png"\r\nContent-Type: image/png\r\n\r\n'
        )
        tail_part = b"PNGDATA\r\n--b--\r\n"
        reply = await gpu_service.send_chunked(
            "/upload/image",
            "svc.local",
            headers={"Content-Type": "multipart/form-data; boundary=b"},
            chunks=[head_part, tail_part],
        )
        assert reply.status == HTTP_OK
        received = target.last_request.body
        assert received == head_part + tail_part
        image_part = received.split(b"--b")[1]
        assert image_part.endswith(b"PNGDATA\r\n")

    async def test_framing_headers_do_not_cross_the_proxy(
        self, gpu_service: ProxyHarness, target: UpstreamServer
    ) -> None:
        """Inbound framing headers from a chunked request never reach the target."""
        reply = await gpu_service.send_chunked(
            "/upload/image",
            "svc.local",
            chunks=[b"x"],
            headers={"Keep-Alive": "timeout=5", "Proxy-Connection": "close"},
        )
        assert reply.status == HTTP_OK
        received = {key.lower() for key in target.last_request.headers}
        assert "transfer-encoding" not in received
        assert "keep-alive" not in received
        assert "proxy-connection" not in received

    async def test_forwarding_adds_no_proxy_headers(
        self, gpu_service: ProxyHarness, target: UpstreamServer
    ) -> None:
        """The proxy does not announce itself with forwarding headers."""
        await gpu_service.send("/v1/models", "svc.local")
        forwarded = [
            key
            for key in target.last_request.headers
            if key.lower() in ("x-forwarded-for", "x-forwarded-host", "via", "forwarded")
        ]
        assert forwarded == []

    async def test_passthrough_does_not_count_as_gpu_traffic(
        self, gpu_service: ProxyHarness
    ) -> None:
        """A non-GPU route leaves the in-flight counter alone."""
        await gpu_service.send("/v1/models", "svc.local")
        entry = await gpu_service.service("svc")
        assert entry["in_flight"] == 0
        assert entry["state"] == "cold"

    async def test_unreachable_target_falls_back_to_backup(
        self, start_proxy: StartProxy, backup: UpstreamServer
    ) -> None:
        """A refused connection to the target is retried on the backup."""
        backup.body = b"served-by-backup"
        svc = replace(
            make_service("svc", f"http://{LOCALHOST}:{free_port()}"), backup=backup.url
        )
        harness = await start_proxy(make_config([svc], free_port()))
        reply = await harness.send("/v1/models", "svc.local")
        assert reply.status == HTTP_OK
        assert reply.text == "served-by-backup"
        assert (await harness.routing("svc")) == {
            "target": 1,
            "backup": 1,
            "evictions": 0,
        }

    async def test_fallback_true_retries_target_error_on_backup(
        self, start_proxy: StartProxy, backup: UpstreamServer
    ) -> None:
        """With the flag explicitly on, a refused target is retried on the backup."""
        backup.body = b"served-by-backup"
        svc = replace(
            make_service("svc", f"http://{LOCALHOST}:{free_port()}"),
            backup=backup.url,
            fallback_on_error=True,
        )
        harness = await start_proxy(make_config([svc], free_port()))
        reply = await harness.send("/v1/models", "svc.local")
        assert reply.status == HTTP_OK
        assert reply.text == "served-by-backup"
        assert (await harness.routing("svc"))["backup"] == 1

    async def test_fallback_true_non_200_retries_on_backup(
        self, start_proxy: StartProxy, target: UpstreamServer, backup: UpstreamServer
    ) -> None:
        """With the flag on, a non-200 from the target is retried on the backup."""
        target.status = HTTP_INTERNAL_SERVER_ERROR
        backup.body = b"backup-recovered"
        svc = replace(
            make_service("svc", target.url),
            backup=backup.url,
            fallback_on_error=True,
        )
        harness = await start_proxy(make_config([svc], free_port()))
        reply = await harness.send("/v1/models", "svc.local")
        assert reply.status == HTTP_OK
        assert reply.text == "backup-recovered"
        assert len(target.requests) == 1
        assert len(backup.requests) == 1

    async def test_fallback_false_refused_target_returns_503(
        self, start_proxy: StartProxy, backup: UpstreamServer
    ) -> None:
        """With the flag off, a refused target is not retried; the client gets 503."""
        backup.body = b"should-not-be-used"
        svc = replace(
            make_service("svc", f"http://{LOCALHOST}:{free_port()}"),
            backup=backup.url,
            fallback_on_error=False,
        )
        harness = await start_proxy(make_config([svc], free_port()))
        reply = await harness.send("/v1/models", "svc.local")
        assert reply.status == HTTP_SERVICE_UNAVAILABLE
        assert len(backup.requests) == 0
        assert (await harness.routing("svc"))["backup"] == 0

    async def test_fallback_false_non_200_reaches_client(
        self, start_proxy: StartProxy, target: UpstreamServer, backup: UpstreamServer
    ) -> None:
        """With the flag off, the target's own non-200 is relayed, not retried."""
        target.status = HTTP_INTERNAL_SERVER_ERROR
        target.body = b"upstream exploded"
        backup.body = b"should-not-be-used"
        svc = replace(
            make_service("svc", target.url),
            backup=backup.url,
            fallback_on_error=False,
        )
        harness = await start_proxy(make_config([svc], free_port()))
        reply = await harness.send("/v1/models", "svc.local")
        assert reply.status == HTTP_INTERNAL_SERVER_ERROR
        assert reply.text == "upstream exploded"
        assert len(backup.requests) == 0
        assert (await harness.routing("svc"))["backup"] == 0

    async def test_fallback_false_still_uses_backup_when_unhealthy(
        self, start_proxy: StartProxy, target: UpstreamServer, backup: UpstreamServer
    ) -> None:
        """The flag only gates runtime failover; unhealthy routing still uses backup."""
        backup.body = b"served-by-backup"
        svc = replace(
            make_service("svc", target.url),
            backup=backup.url,
            fallback_on_error=False,
        )
        harness = await start_proxy(make_config([svc], free_port()))
        state = harness.proxy.service_states["svc"]
        state.is_healthy = False
        state.transition(ServiceStateEnum.unhealthy)
        reply = await harness.send("/v1/models", "svc.local")
        assert reply.status == HTTP_OK
        assert reply.text == "served-by-backup"
        assert (await harness.routing("svc"))["backup"] == 1
        assert (await harness.routing("svc"))["target"] == 0

    @pytest.mark.parametrize(
        "status",
        [
            HTTP_INTERNAL_SERVER_ERROR,
            HTTP_BAD_GATEWAY,
            HTTP_SERVICE_UNAVAILABLE,
        ],
    )
    async def test_target_5xx_falls_back_to_backup_and_logs_target_failed(
        self,
        gpu_service: ProxyHarness,
        target: UpstreamServer,
        backup: UpstreamServer,
        status: int,
    ) -> None:
        """A 5xx response from the target retries on the backup and logs a failure event."""
        target.status = status
        backup.body = b"backup-recovered"
        reply = await gpu_service.send("/v1/models", "svc.local")
        assert reply.status == HTTP_OK
        assert reply.text == "backup-recovered"
        assert len(target.requests) == 1
        assert len(backup.requests) == 1
        events = await gpu_service.events("target_failed", "svc")
        assert [e["event"] for e in events] == ["target_failed"]

    @pytest.mark.parametrize(
        "status",
        [
            HTTP_CREATED,
            HTTP_FOUND,
            HTTP_BAD_REQUEST,
            HTTP_NOT_FOUND,
            HTTP_CONTENT_TOO_LARGE,
        ],
    )
    async def test_target_non_5xx_status_returns_directly_without_failure_event(
        self,
        gpu_service: ProxyHarness,
        target: UpstreamServer,
        backup: UpstreamServer,
        status: int,
    ) -> None:
        """A 2xx, 3xx, or 4xx response from the target is returned as normal."""
        target.status = status
        target.body = b"target-response"
        backup.body = b"should-not-be-used"
        reply = await gpu_service.send("/v1/models", "svc.local")
        assert reply.status == status
        assert reply.text == "target-response"
        assert len(target.requests) == 1
        assert len(backup.requests) == 0
        assert await gpu_service.events("target_failed", "svc") == []

    async def test_backup_error_reaches_the_client(
        self, gpu_service: ProxyHarness, target: UpstreamServer, backup: UpstreamServer
    ) -> None:
        """When both upstreams fail, the client is given the backup's answer."""
        target.status = HTTP_BAD_GATEWAY
        backup.status = HTTP_SERVICE_UNAVAILABLE
        backup.body = b"backup is down too"
        reply = await gpu_service.send("/v1/models", "svc.local")
        assert reply.status == HTTP_SERVICE_UNAVAILABLE
        assert reply.text == "backup is down too"
        assert [e["event"] for e in await gpu_service.events("target_failed", "svc")] == [
            "target_failed"
        ]
        assert [e["event"] for e in await gpu_service.events("backup_failed", "svc")] == [
            "backup_failed"
        ]

    async def test_backup_4xx_does_not_log_backup_failed(
        self, gpu_service: ProxyHarness, target: UpstreamServer, backup: UpstreamServer
    ) -> None:
        """A 4xx response from the backup is returned as normal."""
        target.status = HTTP_BAD_GATEWAY
        backup.status = HTTP_BAD_REQUEST
        backup.body = b"invalid payload"
        reply = await gpu_service.send("/v1/models", "svc.local")
        assert reply.status == HTTP_BAD_REQUEST
        assert reply.text == "invalid payload"
        assert [e["event"] for e in await gpu_service.events("target_failed", "svc")] == [
            "target_failed"
        ]
        assert await gpu_service.events("backup_failed", "svc") == []

    @pytest.mark.parametrize(
        "status",
        [
            HTTP_INTERNAL_SERVER_ERROR,
            HTTP_BAD_GATEWAY,
            HTTP_SERVICE_UNAVAILABLE,
        ],
    )
    async def test_target_5xx_without_backup_reaches_client_and_logs_target_failed(
        self, start_proxy: StartProxy, target: UpstreamServer, status: int
    ) -> None:
        """Without a backup, a 5xx response is relayed and logged as target failed."""
        target.status = status
        target.body = b"upstream exploded"
        harness = await start_proxy(
            make_config([make_service("svc", target.url)], free_port())
        )
        reply = await harness.send("/v1/models", "svc.local")
        assert reply.status == status
        assert reply.text == "upstream exploded"
        events = await harness.events("target_failed", "svc")
        assert [e["event"] for e in events] == ["target_failed"]

    async def test_both_upstreams_unreachable_returns_503(
        self, start_proxy: StartProxy
    ) -> None:
        """When neither upstream accepts a connection the client gets 503."""
        svc = replace(
            make_service("svc", f"http://{LOCALHOST}:{free_port()}"),
            backup=f"http://{LOCALHOST}:{free_port()}",
        )
        harness = await start_proxy(make_config([svc], free_port()))
        reply = await harness.send("/v1/models", "svc.local")
        assert reply.status == HTTP_SERVICE_UNAVAILABLE

    async def test_unreachable_target_without_backup_returns_503(
        self, start_proxy: StartProxy
    ) -> None:
        """A refused target and no backup leaves nothing to serve."""
        svc = make_service("svc", f"http://{LOCALHOST}:{free_port()}")
        harness = await start_proxy(make_config([svc], free_port()))
        reply = await harness.send("/v1/models", "svc.local")
        assert reply.status == HTTP_SERVICE_UNAVAILABLE

    async def test_missing_backup_is_logged_as_target_failure(
        self, start_proxy: StartProxy
    ) -> None:
        """A failed target is recorded in the eviction log for the dashboard."""
        svc = make_service("svc", f"http://{LOCALHOST}:{free_port()}")
        harness = await start_proxy(make_config([svc], free_port()))
        await harness.send("/v1/models", "svc.local")
        events = await harness.events(name="svc")
        assert [e["event"] for e in events] == ["target_failed"]


class TestGpuGate:
    """GPU routes pass the budget gate; other routes bypass it."""

    async def test_approved_request_reaches_the_target(
        self, gpu_service: ProxyHarness, target: UpstreamServer, backup: UpstreamServer
    ) -> None:
        """With budget to spare the request is routed to the target."""
        reply = await gpu_service.send("/v1/completions", "svc.local", "POST", b"{}")
        assert reply.status == HTTP_OK
        assert len(backup.requests) == 0
        assert target.last_request.path == "/v1/completions"
        assert (await gpu_service.routing("svc"))["target"] == 1

    async def test_approval_is_recorded(self, gpu_service: ProxyHarness) -> None:
        """An approved GPU request explains itself in the eviction log."""
        await gpu_service.send("/v1/completions", "svc.local", "POST", b"{}")
        events = await gpu_service.events("request_approved", "svc")
        assert events[0]["details"] == "request_approved_budget_sufficient"

    @pytest.mark.parametrize(
        "path,expected_target,expected_backup",
        [
            ("/v1/completions", 0, 1),
            ("/v1/completions/extra", 0, 1),
            ("/v1/models", 1, 0),
            ("/v1", 1, 0),
        ],
    )
    async def test_only_matching_prefixes_are_gated(
        self,
        gpu_service: ProxyHarness,
        target: UpstreamServer,
        backup: UpstreamServer,
        fake_gpu: FakeGpu,
        path: str,
        expected_target: int,
        expected_backup: int,
    ) -> None:
        """Losing GPU data diverts gated paths only; the rest still pass through."""
        fake_gpu.poll_fails = True
        await gpu_service.wait_for_dashboard(
            lambda snapshot: snapshot["proxy"]["health"] == "unhealthy"
        )
        await gpu_service.send(path, "svc.local", "POST", b"{}")
        assert len(target.requests) == expected_target
        assert len(backup.requests) == expected_backup

    async def test_lost_gpu_data_diverts_to_backup(
        self, gpu_service: ProxyHarness, backup: UpstreamServer, fake_gpu: FakeGpu
    ) -> None:
        """Without trustworthy GPU data every GPU request goes to the backup."""
        fake_gpu.poll_fails = True
        await gpu_service.wait_for_dashboard(
            lambda snapshot: snapshot["proxy"]["health"] == "unhealthy"
        )
        reply = await gpu_service.send("/v1/completions", "svc.local", "POST", b"{}")
        assert reply.status == HTTP_OK
        assert len(backup.requests) == 1
        events = await gpu_service.events("request_rejected", "svc")
        assert events[0]["details"] == "gpu_poll_failed"

    async def test_gated_request_without_a_backup_is_refused(
        self, start_proxy: StartProxy, target: UpstreamServer, fake_gpu: FakeGpu
    ) -> None:
        """A gated request with nowhere to divert to answers 503."""
        svc = replace(make_service("svc", target.url), gpu_routes=["/v1/completions"])
        harness = await start_proxy(make_config([svc], free_port()))
        await harness.wait_for_poll()
        fake_gpu.poll_fails = True
        await harness.wait_for_dashboard(
            lambda snapshot: snapshot["proxy"]["health"] == "unhealthy"
        )
        reply = await harness.send("/v1/completions", "svc.local", "POST", b"{}")
        assert reply.status == HTTP_SERVICE_UNAVAILABLE
        assert reply.reason == "Service Unavailable"
        assert target.requests == []

    async def test_service_without_gpu_routes_is_never_gated(
        self, start_proxy: StartProxy, target: UpstreamServer, fake_gpu: FakeGpu
    ) -> None:
        """An empty gpu_routes list means the gate never applies."""
        fake_gpu.poll_fails = True
        harness = await start_proxy(
            make_config([make_service("plain", target.url)], free_port())
        )
        reply = await harness.send("/v1/completions", "plain.local", "POST", b"{}")
        assert reply.status == HTTP_OK
        assert len(target.requests) == 1

    async def test_external_service_is_never_gated(
        self, start_proxy: StartProxy, target: UpstreamServer, fake_gpu: FakeGpu
    ) -> None:
        """External services stay reachable even when GPU data is gone."""
        external = replace(
            make_service("ext", target.url),
            external=True,
            vram_budget=0,
            container="",
            gpu_routes=["/v1/completions"],
        )
        fake_gpu.poll_fails = True
        harness = await start_proxy(make_config([external], free_port()))
        reply = await harness.send("/v1/completions", "ext.local", "POST", b"{}")
        assert reply.status == HTTP_OK
        assert len(target.requests) == 1

    async def test_backup_routing_leaves_in_flight_untouched(
        self, gpu_service: ProxyHarness, fake_gpu: FakeGpu
    ) -> None:
        """Work sent to the backup consumes no GPU budget on the service."""
        fake_gpu.poll_fails = True
        await gpu_service.wait_for_dashboard(
            lambda snapshot: snapshot["proxy"]["health"] == "unhealthy"
        )
        await gpu_service.send("/v1/completions", "svc.local", "POST", b"{}")
        entry = await gpu_service.service("svc")
        assert entry["in_flight"] == 0
        assert entry["state"] == "cold"
        assert (await gpu_service.routing("svc"))["backup"] == 1

    async def test_budget_is_capped_by_configuration(
        self, start_proxy: StartProxy, target: UpstreamServer, backup: UpstreamServer
    ) -> None:
        """A service larger than the configured budget can never be approved."""
        oversized = replace(
            make_service("big", target.url, vram_budget=VRAM_BUDGET + 1),
            gpu_routes=["/v1/completions"],
            backup=backup.url,
        )
        harness = await start_proxy(make_config([oversized], free_port()))
        await harness.wait_for_poll()
        reply = await harness.send("/v1/completions", "big.local", "POST", b"{}")
        assert reply.status == HTTP_OK
        assert len(backup.requests) == 1
        assert len(target.requests) == 0

    async def test_in_flight_request_keeps_the_service_hot(
        self, gpu_service: ProxyHarness, target: UpstreamServer
    ) -> None:
        """A second request arrives while the first is still running on the target."""
        target.response_delay = SLOW_RESPONSE
        first = asyncio.create_task(
            gpu_service.send("/v1/completions", "svc.local", "POST", b"{}")
        )
        assert await gpu_service.wait_for_state("svc", "hot") == "hot"
        second = await gpu_service.send("/v1/completions", "svc.local", "POST", b"{}")
        assert second.status == HTTP_OK
        assert (await first).status == HTTP_OK
        assert len(target.requests) == 2
        assert await gpu_service.wait_for_in_flight("svc", 0) == 0

    async def test_hot_pass_through_is_recorded(
        self, gpu_service: ProxyHarness, target: UpstreamServer
    ) -> None:
        """The second concurrent request is approved because the service is hot."""
        target.response_delay = SLOW_RESPONSE
        first = asyncio.create_task(
            gpu_service.send("/v1/completions", "svc.local", "POST", b"{}")
        )
        assert await gpu_service.wait_for_state("svc", "hot") == "hot"
        await gpu_service.send("/v1/completions", "svc.local", "POST", b"{}")
        await first
        reasons = [e["details"] for e in await gpu_service.events("request_approved")]
        assert "hot_service_pass_through" in reasons


class TestServiceStateMachine:
    """States a service moves through as VRAM and traffic change."""

    async def test_starts_cold(self, gpu_service: ProxyHarness) -> None:
        """A service with no VRAM and no traffic is cold."""
        assert (await gpu_service.service("svc"))["state"] == "cold"

    async def test_vram_makes_a_service_warm(
        self, gpu_service: ProxyHarness, fake_gpu: FakeGpu
    ) -> None:
        """VRAM appearing for the container promotes the service to warm."""
        fake_gpu.occupy({"svc-container": SMALL_LOAD_MIB})
        assert await gpu_service.wait_for_state("svc", "warm") == "warm"
        assert (await gpu_service.service("svc"))["vram_used_mib"] == SMALL_LOAD_MIB

    async def test_released_vram_makes_a_service_cold(
        self, gpu_service: ProxyHarness, fake_gpu: FakeGpu
    ) -> None:
        """VRAM dropping back to zero demotes the service to cold."""
        fake_gpu.occupy({"svc-container": SMALL_LOAD_MIB})
        assert await gpu_service.wait_for_state("svc", "warm") == "warm"
        fake_gpu.release_all()
        assert await gpu_service.wait_for_state("svc", "cold") == "cold"
        events = await gpu_service.events("state_transition", "svc")
        assert events[-1]["details"] == "warm → cold (VRAM dropped to 0)"

    async def test_one_poll_moves_every_service(
        self, start_proxy: StartProxy, target: UpstreamServer, fake_gpu: FakeGpu
    ) -> None:
        """A single poll re-evaluates all services, not just one."""
        first = replace(make_service("one", target.url, priority=1), gpu_routes=["/v1"])
        second = replace(make_service("two", target.url, priority=2), gpu_routes=["/v1"])
        harness = await start_proxy(make_config([first, second], free_port()))
        fake_gpu.occupy({"one-container": SMALL_LOAD_MIB, "two-container": OTHER_LOAD_MIB})
        assert await harness.wait_for_state("one", "warm") == "warm"
        assert await harness.wait_for_state("two", "warm") == "warm"

    async def test_traffic_makes_a_service_hot(
        self, gpu_service: ProxyHarness, target: UpstreamServer
    ) -> None:
        """An in-flight GPU request is reflected as hot with a counter of one."""
        target.response_delay = SLOW_RESPONSE
        pending = asyncio.create_task(
            gpu_service.send("/v1/completions", "svc.local", "POST", b"{}")
        )
        assert await gpu_service.wait_for_in_flight("svc", 1) == 1
        assert (await gpu_service.service("svc"))["state"] == "hot"
        await pending

    async def test_hot_service_ignores_missing_vram(
        self, gpu_service: ProxyHarness, target: UpstreamServer, fake_gpu: FakeGpu
    ) -> None:
        """A poll reporting no VRAM cannot cool a service that is still working."""
        target.response_delay = SLOW_RESPONSE
        pending = asyncio.create_task(
            gpu_service.send("/v1/completions", "svc.local", "POST", b"{}")
        )
        assert await gpu_service.wait_for_in_flight("svc", 1) == 1
        fake_gpu.release_all()
        await asyncio.sleep(POLL_INTERVAL * 3)
        assert (await gpu_service.service("svc"))["state"] == "hot"
        await pending

    async def test_finished_request_cools_the_service_down(
        self, gpu_service: ProxyHarness, fake_gpu: FakeGpu
    ) -> None:
        """A completed request leaves the service warm while VRAM remains."""
        fake_gpu.occupy({"svc-container": SMALL_LOAD_MIB})
        await gpu_service.send("/v1/completions", "svc.local", "POST", b"{}")
        assert await gpu_service.wait_for_state("svc", "warm") == "warm"
        details = [e["details"] for e in await gpu_service.events("state_transition", "svc")]
        assert "hot → cooling_down (request finished)" in details

    async def test_finished_request_goes_cold_without_vram(
        self, gpu_service: ProxyHarness
    ) -> None:
        """A completed request leaves the service cold when it holds no VRAM."""
        await gpu_service.send("/v1/completions", "svc.local", "POST", b"{}")
        assert await gpu_service.wait_for_state("svc", "cold") == "cold"

    async def test_startup_never_evicts(
        self, gpu_service: ProxyHarness, fake_gpu: FakeGpu
    ) -> None:
        """Finding a container already using VRAM does not trigger an eviction."""
        fake_gpu.occupy({"svc-container": SMALL_LOAD_MIB})
        assert await gpu_service.wait_for_state("svc", "warm") == "warm"
        events = await gpu_service.events(name="svc")
        assert [e["event"] for e in events] == ["state_transition"]


@dataclass
class EvictionSetup:
    """A proxy where a low-priority service holds the VRAM a big one needs."""

    harness: ProxyHarness
    marker: Path


@dataclass
class BusyEviction:
    """An eviction that is waiting for a request already on the evicted service."""

    setup: EvictionSetup
    busy: asyncio.Task
    evicting: asyncio.Task


BuildEviction = Callable[..., Awaitable[EvictionSetup]]
StartBusyEviction = Callable[[], Awaitable[BusyEviction]]


@pytest.fixture
def build_eviction(
    start_proxy: StartProxy,
    fake_gpu: FakeGpu,
    target: UpstreamServer,
    backup: UpstreamServer,
    tmp_path: Path,
) -> BuildEviction:
    """Factory for a proxy whose warm low-priority service blocks a high-priority one.

    The low service's eviction command touches a marker file by default, and the
    fake GPU frees that container's VRAM once the marker exists, so a successful
    eviction really does unblock the waiting request.
    """
    touch_marker = f"/usr/bin/touch {tmp_path}/{{container}}"

    async def _build(
        eviction_command: str = touch_marker,
        eviction_timeout: int = EVICTION_TIMEOUT,
        eviction_command_timeout: int = COMMAND_TIMEOUT,
        wait_queue_timeout: int = QUEUE_TIMEOUT,
    ) -> EvictionSetup:
        marker = tmp_path / "low-container"
        low = replace(
            make_service("low", target.url, priority=2, vram_budget=SMALL_SERVICE_VRAM),
            gpu_routes=["/api/detect"],
            backup=backup.url,
            eviction_command=eviction_command,
        )
        high = replace(
            make_service("high", target.url, priority=1, vram_budget=LARGE_SERVICE_VRAM),
            gpu_routes=["/v1/completions"],
            backup=backup.url,
        )
        harness = await start_proxy(
            make_config(
                [low, high],
                free_port(),
                eviction_timeout=eviction_timeout,
                eviction_command_timeout=eviction_command_timeout,
                wait_queue_timeout=wait_queue_timeout,
            )
        )
        fake_gpu.release_when_touched("low-container", marker)
        fake_gpu.occupy({"low-container": BLOCKING_LOAD_MIB})
        assert await harness.wait_for_state("low", "warm") == "warm"
        return EvictionSetup(harness, marker)

    return _build


@pytest.fixture
def start_busy_eviction(
    build_eviction: BuildEviction, target: UpstreamServer
) -> StartBusyEviction:
    """Factory that begins an eviction while the low-priority service still answers."""

    async def _start() -> BusyEviction:
        setup = await build_eviction(eviction_timeout=PATIENT_EVICTION)
        target.response_delay = SLOW_RESPONSE
        busy = asyncio.create_task(
            setup.harness.send("/api/detect", "low.local", "POST", b"{}")
        )
        assert await setup.harness.wait_for_in_flight("low", 1) == 1
        evicting = asyncio.create_task(
            setup.harness.send("/v1/completions", "high.local", "POST", b"{}")
        )
        await setup.harness.wait_for_events("eviction_started", "low")
        return BusyEviction(setup, busy, evicting)

    return _start


class TestEviction:
    """Freeing VRAM by restarting a lower-priority service."""

    async def test_restarting_service_is_not_evicted_again(
        self, build_eviction: BuildEviction, backup: UpstreamServer,
    ) -> None:
        """A pending restart holds the request until its queue timeout."""
        setup = await build_eviction(wait_queue_timeout=BRIEF_QUEUE_WAIT)
        state = setup.harness.proxy.service_states["low"]
        state.transition(ServiceStateEnum.evicting)
        state.transition(ServiceStateEnum.restarting)
        state.is_healthy = False
        backup.body = b"served-by-backup"

        reply = await setup.harness.send("/v1/completions", "high.local", "POST", b"{}")

        assert reply.text == "served-by-backup"
        rejected = await setup.harness.events("request_rejected", "high")
        assert rejected[0]["details"] == "timeout"
        assert setup.marker.exists() is False
        assert await setup.harness.events("eviction_started", "low") == []
        assert await setup.harness.events("eviction_command_completed", "low") == []

    async def test_eviction_unblocks_the_waiting_request(
        self,
        build_eviction: BuildEviction,
        target: UpstreamServer,
        backup: UpstreamServer,
    ) -> None:
        """The high-priority request reaches the target once VRAM is freed."""
        setup = await build_eviction()
        reply = await setup.harness.send("/v1/completions", "high.local", "POST", b"{}")
        assert reply.status == HTTP_OK
        assert setup.marker.exists()
        assert target.last_request.path == "/v1/completions"
        assert len(backup.requests) == 0
        assert (await setup.harness.routing("high"))["target"] == 1

    async def test_eviction_is_logged_from_start_to_finish(
        self, build_eviction: BuildEviction, tmp_path: Path
    ) -> None:
        """The dashboard log explains who evicted whom, and how it ended."""
        setup = await build_eviction()
        await setup.harness.send("/v1/completions", "high.local", "POST", b"{}")
        started = await setup.harness.events("eviction_started", "low")
        assert started[0]["details"] == "evicted by high (priority 1)"
        drained = await setup.harness.events("eviction_completed", "low")
        assert drained[0]["details"] == "in_flight drained"
        finished = await setup.harness.events("eviction_command_completed", "low")
        assert finished[0]["details"] == f"/usr/bin/touch {tmp_path}/low-container"

    async def test_eviction_counter_reaches_the_dashboard(
        self, build_eviction: BuildEviction
    ) -> None:
        """The evicted service's eviction counter is incremented once."""
        setup = await build_eviction()
        await setup.harness.send("/v1/completions", "high.local", "POST", b"{}")
        assert await setup.harness.wait_for_routing("low", "evictions", 1) == 1

    async def test_evicted_service_ends_up_cold(
        self, build_eviction: BuildEviction
    ) -> None:
        """A restarted container holds no VRAM, so the service settles cold."""
        setup = await build_eviction()
        await setup.harness.send("/v1/completions", "high.local", "POST", b"{}")
        assert await setup.harness.wait_for_state("low", "cold") == "cold"
        details = [
            e["details"] for e in await setup.harness.events("state_transition", "low")
        ]
        assert "warm → evicting" in details
        assert "evicting → restarting" in details

    async def test_eviction_waits_for_in_flight_work(
        self, start_busy_eviction: StartBusyEviction
    ) -> None:
        """A request already running on the evicted service is allowed to finish."""
        pending = await start_busy_eviction()
        assert pending.setup.marker.exists() is False
        assert (await pending.busy).status == HTTP_OK
        assert (await pending.evicting).status == HTTP_OK
        drained = await pending.setup.harness.events("eviction_completed", "low")
        assert drained[0]["details"] == "in_flight drained"

    async def test_eviction_gives_up_waiting_after_the_timeout(
        self, build_eviction: BuildEviction, target: UpstreamServer
    ) -> None:
        """A request that will not finish cannot hold the eviction back forever."""
        setup = await build_eviction(eviction_timeout=IMPATIENT_EVICTION)
        target.response_delay = SLOW_RESPONSE
        busy = asyncio.create_task(
            setup.harness.send("/api/detect", "low.local", "POST", b"{}")
        )
        assert await setup.harness.wait_for_in_flight("low", 1) == 1
        await setup.harness.send("/v1/completions", "high.local", "POST", b"{}")
        timed_out = await setup.harness.events("eviction_timeout", "low")
        assert timed_out[0]["details"] == "eviction_timeout (0.1s) expired"
        assert (await busy).status == HTTP_OK

    async def test_failing_command_is_reported(
        self, build_eviction: BuildEviction
    ) -> None:
        """A non-zero exit code is logged with the command and the exit code."""
        setup = await build_eviction(
            eviction_command="/bin/false {container}",
            wait_queue_timeout=BRIEF_QUEUE_WAIT,
        )
        await setup.harness.send("/v1/completions", "high.local", "POST", b"{}")
        failed = await setup.harness.events("eviction_command_failed", "low")
        assert failed[0]["details"] == "/bin/false low-container (exit_code=1)"

    async def test_hanging_command_is_killed(
        self, build_eviction: BuildEviction
    ) -> None:
        """A command that outlives its timeout is killed and logged as failed."""
        setup = await build_eviction(
            eviction_command="/bin/sleep 30",
            eviction_command_timeout=COMMAND_KILL_TIMEOUT,
            wait_queue_timeout=BRIEF_QUEUE_WAIT,
        )
        await setup.harness.send("/v1/completions", "high.local", "POST", b"{}")
        failed = await setup.harness.events("eviction_command_failed", "low")
        assert failed[0]["details"] == "/bin/sleep 30 (timeout after 0.2s)"

    async def test_command_that_never_frees_vram_sends_work_to_backup(
        self, build_eviction: BuildEviction, backup: UpstreamServer
    ) -> None:
        """When eviction does not release VRAM the request settles on the backup."""
        setup = await build_eviction(
            eviction_command="/bin/false {container}",
            wait_queue_timeout=BRIEF_QUEUE_WAIT,
        )
        backup.body = b"served-by-backup"
        reply = await setup.harness.send("/v1/completions", "high.local", "POST", b"{}")
        assert reply.text == "served-by-backup"
        assert len(backup.requests) == 1

    async def test_concurrent_requests_evict_once(
        self, build_eviction: BuildEviction
    ) -> None:
        """Two requests needing the same VRAM share a single eviction."""
        setup = await build_eviction()
        replies = await asyncio.gather(
            setup.harness.send("/v1/completions", "high.local", "POST", b"1"),
            setup.harness.send("/v1/completions", "high.local", "POST", b"2"),
        )
        assert [reply.status for reply in replies] == [HTTP_OK, HTTP_OK]
        assert len(await setup.harness.events("eviction_started", "low")) == 1
        assert len(await setup.harness.events("eviction_command_completed", "low")) == 1

    async def test_lowest_priority_is_evicted_first(
        self,
        start_proxy: StartProxy,
        fake_gpu: FakeGpu,
        target: UpstreamServer,
        backup: UpstreamServer,
        tmp_path: Path,
    ) -> None:
        """When several services must go, the least important goes first."""
        mid = replace(
            make_service("mid", target.url, priority=2, vram_budget=SMALL_SERVICE_VRAM),
            gpu_routes=["/api/detect"],
            eviction_command=f"/usr/bin/touch {tmp_path}/{{container}}",
        )
        low = replace(mid, name="low", hostname="low.local", priority=3,
                      container="low-container")
        high = replace(
            make_service("high", target.url, priority=1, vram_budget=LARGE_SERVICE_VRAM),
            gpu_routes=["/v1/completions"],
            backup=backup.url,
        )
        harness = await start_proxy(make_config([mid, low, high], free_port()))
        fake_gpu.release_when_touched("mid-container", tmp_path / "mid-container")
        fake_gpu.release_when_touched("low-container", tmp_path / "low-container")
        fake_gpu.occupy(
            {"mid-container": CROWDED_LOAD_MIB, "low-container": CROWDED_LOAD_MIB}
        )
        assert await harness.wait_for_state("low", "warm") == "warm"
        await harness.send("/v1/completions", "high.local", "POST", b"{}")
        started = await harness.wait_for_events("eviction_started", count=2)
        assert [entry["service"] for entry in started] == ["low", "mid"]

    async def test_impossible_eviction_goes_straight_to_backup(
        self, build_eviction: BuildEviction, fake_gpu: FakeGpu, backup: UpstreamServer
    ) -> None:
        """Evicting everything available still would not free enough VRAM."""
        setup = await build_eviction()
        fake_gpu.occupy(
            {"low-container": EVICTABLE_LOAD_MIB, "someone-else": BLOCKING_LOAD_MIB}
        )
        await setup.harness.wait_for_dashboard(
            lambda snapshot: snapshot["gpu"]["docker_vram_mib"]
            == BLOCKING_LOAD_MIB + EVICTABLE_LOAD_MIB
        )
        reply = await setup.harness.send("/v1/completions", "high.local", "POST", b"{}")
        assert reply.status == HTTP_OK
        assert len(backup.requests) == 1
        assert setup.marker.exists() is False
        rejected = await setup.harness.events("request_rejected", "high")
        assert rejected[0]["details"] == "impossible_eviction"

    async def test_nothing_to_evict_goes_straight_to_backup(
        self, gpu_service: ProxyHarness, backup: UpstreamServer, fake_gpu: FakeGpu
    ) -> None:
        """VRAM held outside any managed service cannot be reclaimed."""
        fake_gpu.occupy({"someone-else": HOGGED_LOAD_MIB})
        await gpu_service.wait_for_dashboard(
            lambda snapshot: snapshot["gpu"]["docker_vram_mib"] == HOGGED_LOAD_MIB
        )
        reply = await gpu_service.send("/v1/completions", "svc.local", "POST", b"{}")
        assert reply.status == HTTP_OK
        assert len(backup.requests) == 1
        rejected = await gpu_service.events("request_rejected", "svc")
        assert rejected[0]["details"] == "no_evictable_services"

    async def test_waiting_request_times_out_to_backup(
        self, build_eviction: BuildEviction, target: UpstreamServer
    ) -> None:
        """A request waiting longer than the queue timeout is sent to the backup."""
        setup = await build_eviction(
            eviction_timeout=PATIENT_EVICTION,
            wait_queue_timeout=IMPATIENT_QUEUE_WAIT,
        )
        target.response_delay = SLOW_RESPONSE
        busy = asyncio.create_task(
            setup.harness.send("/api/detect", "low.local", "POST", b"{}")
        )
        assert await setup.harness.wait_for_in_flight("low", 1) == 1
        reply = await setup.harness.send("/v1/completions", "high.local", "POST", b"{}")
        assert reply.status == HTTP_OK
        rejected = await setup.harness.events("request_rejected", "high")
        assert rejected[0]["details"] == "timeout"
        assert (await busy).status == HTTP_OK
        await setup.harness.wait_for_events("eviction_command_completed", "low")

    async def test_service_restart_leaves_budget_free_for_the_next_request(
        self, build_eviction: BuildEviction
    ) -> None:
        """After an eviction the freed budget is available to the evicted service."""
        setup = await build_eviction()
        await setup.harness.send("/v1/completions", "high.local", "POST", b"{}")
        assert await setup.harness.wait_for_state("low", "cold") == "cold"
        reply = await setup.harness.send("/api/detect", "low.local", "POST", b"{}")
        assert reply.status == HTTP_OK
        assert (await setup.harness.routing("low"))["target"] == 1
        assert (await setup.harness.routing("low"))["backup"] == 0


class TestWebSocketProxying:
    """WebSocket upgrades relayed between the client and the upstream."""

    async def test_messages_travel_both_ways(
        self, gpu_service: ProxyHarness, target: UpstreamServer
    ) -> None:
        """A message sent by the client comes back from the upstream."""
        target.ws_close_after = 2
        async with gpu_service.websocket("/socket", "svc.local") as ws:
            await ws.send_str("first")
            assert (await ws.receive()).data == "target:first"
            await ws.send_str("second")
            assert (await ws.receive()).data == "target:second"

    async def test_binary_frames_travel_both_ways(
        self, gpu_service: ProxyHarness, target: UpstreamServer
    ) -> None:
        """Binary WebSocket frames are relayed as binary in both directions."""
        target.ws_close_after = 1
        async with gpu_service.websocket("/socket", "svc.local") as ws:
            await ws.send_bytes(b"\x00\x01binary")
            message = await ws.receive()
            assert message.type is WSMsgType.BINARY
            assert message.data == b"echo:\x00\x01binary"

    async def test_upstream_can_speak_first(
        self, gpu_service: ProxyHarness, target: UpstreamServer
    ) -> None:
        """A greeting pushed by the upstream reaches the client."""
        target.ws_greeting = "hello"
        target.ws_close_after = 1
        async with gpu_service.websocket("/socket", "svc.local") as ws:
            assert (await ws.receive()).data == "hello"
            await ws.send_str("ack")
            assert (await ws.receive()).data == "target:ack"

    async def test_upgrade_reaches_the_upstream_with_the_path(
        self, gpu_service: ProxyHarness, target: UpstreamServer
    ) -> None:
        """The upgrade is forwarded to the same path on the upstream."""
        target.ws_close_after = 1
        async with gpu_service.websocket("/socket?room=7", "svc.local") as ws:
            await ws.send_str("x")
            await ws.receive()
        assert target.last_request.path == "/socket?room=7"
        assert target.last_request.headers["Upgrade"].lower() == "websocket"

    async def test_refused_upgrade_falls_back_to_backup(
        self,
        gpu_service: ProxyHarness,
        target: UpstreamServer,
        backup: UpstreamServer,
    ) -> None:
        """When the target refuses the handshake the backup is tried."""
        target.reject_websocket = True
        backup.ws_close_after = 1
        async with gpu_service.websocket("/socket", "svc.local") as ws:
            await ws.send_str("hi")
            assert (await ws.receive()).data == "backup:hi"
        assert (await gpu_service.routing("svc"))["backup"] == 1

    async def test_unreachable_target_falls_back_to_backup_over_websocket(
        self, start_proxy: StartProxy, backup: UpstreamServer
    ) -> None:
        """A refused connection on the target is retried on the backup."""
        backup.ws_close_after = 1
        svc = replace(
            make_service("svc", f"http://{LOCALHOST}:{free_port()}"), backup=backup.url
        )
        harness = await start_proxy(make_config([svc], free_port()))
        async with harness.websocket("/socket", "svc.local") as ws:
            await ws.send_str("hi")
            assert (await ws.receive()).data == "backup:hi"

    async def test_refused_upgrade_without_backup_drops_the_connection(
        self, start_proxy: StartProxy, target: UpstreamServer
    ) -> None:
        """A failed upgrade with nowhere to fall back closes the client connection.

        The response object is already a WebSocket handshake, so the proxy has no
        way left to answer with an HTTP status.
        """
        target.reject_websocket = True
        harness = await start_proxy(
            make_config([make_service("svc", target.url)], free_port())
        )
        with pytest.raises(ServerDisconnectedError):
            await harness.send("/socket", "svc.local", headers=WS_UPGRADE_HEADERS)
        assert target.requests[0].path == "/socket"

    async def test_unreachable_target_without_backup_drops_the_connection(
        self, start_proxy: StartProxy
    ) -> None:
        """A socket that cannot reach its target and has no backup is dropped."""
        svc = make_service("svc", f"http://{LOCALHOST}:{free_port()}")
        harness = await start_proxy(make_config([svc], free_port()))
        with pytest.raises(ServerDisconnectedError):
            await harness.send("/socket", "svc.local", headers=WS_UPGRADE_HEADERS)

    async def test_gpu_routed_socket_holds_the_budget(
        self, gpu_service: ProxyHarness
    ) -> None:
        """A GPU-routed socket counts as in-flight until it closes."""
        async with gpu_service.websocket("/v1/completions", "svc.local") as ws:
            await ws.send_str("hi")
            assert (await ws.receive()).data == "target:hi"
            assert await gpu_service.wait_for_in_flight("svc", 1) == 1
            assert (await gpu_service.service("svc"))["state"] == "hot"
        assert await gpu_service.wait_for_in_flight("svc", 0) == 0
        assert (await gpu_service.routing("svc"))["target"] == 1

    async def test_websocket_client_disconnect_is_cancelled_not_failed(
        self, gpu_service: ProxyHarness, backup: UpstreamServer
    ) -> None:
        """Dropping a GPU-routed socket releases in-flight without backup or failure."""
        ws = await gpu_service.session.ws_connect(
            f"{gpu_service.base_url}/v1/completions",
            headers={"Host": "svc.local"},
        )
        await ws.send_str("hi")
        assert (await ws.receive()).data == "target:hi"
        assert await gpu_service.wait_for_in_flight("svc", 1) == 1
        await ws.close()
        assert await gpu_service.wait_for_in_flight("svc", 0) == 0
        assert len(backup.requests) == 0
        assert await gpu_service.events("target_failed", "svc") == []
        assert await gpu_service.events("backup_failed", "svc") == []

    async def test_gpu_routed_socket_uses_backup_without_gpu_data(
        self, gpu_service: ProxyHarness, backup: UpstreamServer, fake_gpu: FakeGpu
    ) -> None:
        """The GPU gate applies to upgrades exactly as it does to requests."""
        fake_gpu.poll_fails = True
        backup.ws_close_after = 1
        await gpu_service.wait_for_dashboard(
            lambda snapshot: snapshot["proxy"]["health"] == "unhealthy"
        )
        async with gpu_service.websocket("/v1/completions", "svc.local") as ws:
            await ws.send_str("hi")
            assert (await ws.receive()).data == "backup:hi"
        assert (await gpu_service.routing("svc"))["backup"] == 1


class TestStreaming:
    """Streamed responses are relayed chunk by chunk."""

    async def test_chunks_arrive_in_order(
        self, gpu_service: ProxyHarness, target: UpstreamServer
    ) -> None:
        """Every chunk written upstream reaches the client in order."""
        target.chunks = [b"data: one\n\n", b"data: two\n\n", b"data: [DONE]\n\n"]
        reply = await gpu_service.send("/v1/completions", "svc.local", "POST", b"{}")
        assert reply.status == HTTP_OK
        assert reply.body == b"data: one\n\ndata: two\n\ndata: [DONE]\n\n"

    async def test_chunks_are_forwarded_as_they_are_produced(
        self, gpu_service: ProxyHarness, target: UpstreamServer
    ) -> None:
        """The client sees the first chunk before the stream has finished."""
        target.chunks = [b"data: one\n\n", b"data: two\n\n"]
        target.chunk_gap = SLOW_CHUNK_GAP
        async with gpu_service.stream("/v1/completions", "svc.local", "POST") as response:
            first = await response.content.readany()
            assert first == b"data: one\n\n"
            assert await gpu_service.wait_for_in_flight("svc", 1) == 1

    async def test_in_flight_is_released_when_the_stream_ends(
        self, gpu_service: ProxyHarness, target: UpstreamServer
    ) -> None:
        """A finished stream frees the budget it held."""
        target.chunks = [b"data: one\n\n", b"data: [DONE]\n\n"]
        target.chunk_gap = BRIEF_CHUNK_GAP
        await gpu_service.send("/v1/completions", "svc.local", "POST", b"{}")
        assert await gpu_service.wait_for_in_flight("svc", 0) == 0
        assert await gpu_service.wait_for_state("svc", "cold") == "cold"

    async def test_client_disconnect_does_not_start_a_backup_attempt(
        self, gpu_service: ProxyHarness, target: UpstreamServer, backup: UpstreamServer
    ) -> None:
        """A client that walks away mid-stream is not an upstream failure."""
        target.chunks = [b"data: one\n\n", b"data: two\n\n", b"data: three\n\n"]
        target.chunk_gap = SLOW_CHUNK_GAP
        streaming = asyncio.create_task(
            gpu_service.send("/v1/completions", "svc.local", "POST", b"{}")
        )
        assert await gpu_service.wait_for_in_flight("svc", 1) == 1
        streaming.cancel()
        assert await gpu_service.wait_for_in_flight("svc", 0) == 0
        assert len(backup.requests) == 0
        assert (await gpu_service.routing("svc"))["backup"] == 0
        assert await gpu_service.events("target_failed", "svc") == []
        assert await gpu_service.events("backup_failed", "svc") == []

    async def test_client_disconnect_before_headers_skips_backup_and_failure_log(
        self,
        start_with_plugins: StartWithPlugins,
        target: UpstreamServer,
        backup: UpstreamServer,
    ) -> None:
        """A client disconnecting before response headers cancels without failover."""
        svc = replace(
            make_service("svc", target.url),
            backup=backup.url,
            gpu_routes=["/v1/completions"],
            plugins=["tracker"],
        )
        harness, manager = await start_with_plugins(
            [svc],
            [plugin_entry("tracker", "CancelTrackingPlugin")],
            [CancelTrackingPlugin],
        )
        target.response_delay = STALLED_RESPONSE
        pending = asyncio.create_task(
            harness.send("/v1/completions", "svc.local", "POST", b"{}")
        )
        await wait_until(lambda: bool(target.requests))
        assert await harness.wait_for_in_flight("svc", 1) == 1
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert await harness.wait_for_in_flight("svc", 0) == 0
        assert len(backup.requests) == 0
        assert (await harness.routing("svc"))["backup"] == 0
        assert await harness.events("target_failed", "svc") == []
        assert await harness.events("backup_failed", "svc") == []
        plugin = manager.get_plugin_instance("svc", "tracker")
        assert plugin.cancelled_requests == [1]

    async def test_concurrent_waiter_survives_other_client_disconnecting(
        self,
        start_proxy: StartProxy,
        target: UpstreamServer,
        backup: UpstreamServer,
        fake_gpu: FakeGpu,
    ) -> None:
        """One disconnecting waiter does not cancel another waiting for the same service."""
        blocker = replace(
            make_service("blocker", backup.url, priority=2, vram_budget=LARGE_SERVICE_VRAM),
            container="blocker",
            gpu_routes=["/api"],
        )
        svc = replace(
            make_service("svc", target.url, priority=1, vram_budget=LARGE_SERVICE_VRAM),
            gpu_routes=["/v1/completions"],
        )
        fake_gpu.occupy({"blocker": LARGE_SERVICE_VRAM})
        harness = await start_proxy(
            make_config([blocker, svc], free_port(), eviction_timeout=PATIENT_EVICTION)
        )
        await harness.wait_for_state("blocker", "warm")

        backup.response_delay = STALLED_RESPONSE
        blocker_req = asyncio.create_task(
            harness.send("/api", "blocker.local", "POST", b"{}")
        )
        await wait_until(lambda: bool(backup.requests))
        await harness.wait_for_state("blocker", "hot")

        waiter_one = asyncio.create_task(
            harness.send("/v1/completions", "svc.local", "POST", b"{}")
        )
        waiter_two = asyncio.create_task(
            harness.send("/v1/completions", "svc.local", "POST", b"{}")
        )
        await harness.wait_for_state("blocker", "evicting")
        await wait_until(
            lambda: any(
                entry.service_name == "svc" and entry.waiters == 2
                for entry in harness.proxy.wait_queue.entries
            )
        )

        waiter_one.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter_one
        await wait_until(
            lambda: any(
                entry.service_name == "svc" and entry.waiters == 1
                for entry in harness.proxy.wait_queue.entries
            )
        )
        assert not waiter_two.done()

        blocker_req.cancel()
        with pytest.raises(asyncio.CancelledError):
            await blocker_req
        fake_gpu.release_all()

        reply = await waiter_two
        assert reply.status == HTTP_OK
        assert reply.body == b"upstream-ok"
        assert len(target.requests) == 1


class TestLoopbackDetection:
    """Targets that point back at the proxy's own listening port."""

    async def test_self_loop_is_refused(self, start_proxy: StartProxy) -> None:
        """A service whose target is its own hostname reports a loop."""
        port = free_port()
        svc = make_service("mirror", f"http://mirror.local:{port}")
        harness = await start_proxy(make_config([svc], port))
        reply = await harness.send("/api", "mirror.local")
        assert reply.status == HTTP_LOOP_DETECTED
        assert reply.reason == "Loop Detected"

    async def test_loop_on_the_target_falls_back_to_backup(
        self, start_proxy: StartProxy, backup: UpstreamServer
    ) -> None:
        """A looping target is abandoned in favour of the backup."""
        port = free_port()
        svc = replace(
            make_service("mirror", f"http://mirror.local:{port}"), backup=backup.url
        )
        harness = await start_proxy(make_config([svc], port))
        backup.body = b"backup-answered"
        reply = await harness.send("/api", "mirror.local")
        assert reply.status == HTTP_OK
        assert reply.text == "backup-answered"

    async def test_target_on_another_service_is_handled_internally(
        self, start_proxy: StartProxy, target: UpstreamServer
    ) -> None:
        """A target naming a sibling service is redirected without a network hop."""
        port = free_port()
        front = make_service("front", f"http://back.local:{port}")
        back = replace(make_service("back", target.url, priority=2), gpu_routes=[])
        harness = await start_proxy(make_config([front, back], port))
        target.body = b"reached-the-back"
        reply = await harness.send("/api", "front.local")
        assert reply.status == HTTP_OK
        assert reply.text == "reached-the-back"
        assert len(target.requests) == 1

    async def test_backup_may_redirect_to_another_service(
        self, start_proxy: StartProxy, target: UpstreamServer
    ) -> None:
        """A backup naming a sibling service is redirected the same way."""
        port = free_port()
        front = replace(
            make_service("front", f"http://{LOCALHOST}:{free_port()}"),
            backup=f"http://back.local:{port}",
        )
        back = make_service("back", target.url, priority=2)
        harness = await start_proxy(make_config([front, back], port))
        target.body = b"backup-redirected"
        reply = await harness.send("/api", "front.local")
        assert reply.status == HTTP_OK
        assert reply.text == "backup-redirected"

    async def test_redirect_chain_stops_at_the_recursion_limit(
        self, start_proxy: StartProxy
    ) -> None:
        """A ring of services redirecting to each other is cut off as a loop."""
        port = free_port()
        names = ["hop0", "hop1", "hop2", "hop3", "hop4", "hop5"]
        chain = [
            make_service(name, f"http://{names[i + 1]}.local:{port}", priority=i + 1)
            for i, name in enumerate(names[:-1])
        ]
        last = make_service(names[-1], f"http://{LOCALHOST}:{free_port()}", priority=6)
        harness = await start_proxy(make_config([*chain, last], port))
        reply = await harness.send("/api", "hop0.local")
        assert reply.status == HTTP_LOOP_DETECTED
        assert reply.reason.startswith("Loop Detected")

    async def test_redirected_service_error_reaches_the_client(
        self, start_proxy: StartProxy, target: UpstreamServer, backup: UpstreamServer
    ) -> None:
        """The deepest layer's failure is what the client is told about."""
        port = free_port()
        front = make_service("front", f"http://back.local:{port}")
        back = replace(
            make_service("back", target.url, priority=2), backup=backup.url
        )
        target.status = HTTP_BAD_GATEWAY
        backup.status = HTTP_INTERNAL_SERVER_ERROR
        backup.body = b"backup failed too"
        harness = await start_proxy(make_config([front, back], port))
        reply = await harness.send("/api", "front.local")
        assert reply.status == HTTP_INTERNAL_SERVER_ERROR
        assert reply.text == "backup failed too"

    async def test_websocket_self_loop_is_refused(
        self, start_proxy: StartProxy
    ) -> None:
        """A looping WebSocket target never opens a connection."""
        port = free_port()
        svc = replace(
            make_service("mirror", f"http://mirror.local:{port}"), gpu_routes=["/socket"]
        )
        harness = await start_proxy(make_config([svc], port))
        await harness.wait_for_poll()
        with pytest.raises(ServerDisconnectedError):
            await harness.send("/socket", "mirror.local", headers=WS_UPGRADE_HEADERS)
        assert await harness.wait_for_in_flight("mirror", 0) == 0

    async def test_websocket_loop_falls_back_to_backup(
        self, start_proxy: StartProxy, backup: UpstreamServer
    ) -> None:
        """A looping WebSocket target hands over to the backup."""
        port = free_port()
        svc = replace(
            make_service("mirror", f"http://mirror.local:{port}"), backup=backup.url
        )
        harness = await start_proxy(make_config([svc], port))
        backup.ws_close_after = 1
        async with harness.websocket("/socket", "mirror.local") as ws:
            await ws.send_str("hi")
            assert (await ws.receive()).data == "backup:hi"


class TestHealthChecks:
    """The background health check loop and the gate it feeds."""

    async def test_passing_command_keeps_the_service_healthy(
        self, start_proxy: StartProxy, target: UpstreamServer
    ) -> None:
        """A command that exits zero leaves the service healthy."""
        svc = replace(
            make_service("svc", target.url),
            health_check_command="/bin/true",
            health_check_interval=1,
        )
        harness = await start_proxy(make_config([svc], free_port()))
        assert await harness.wait_for_health("svc", True) is True
        assert (await harness.service("svc"))["state"] == "cold"

    async def test_failing_command_marks_the_service_unhealthy(
        self, start_proxy: StartProxy, target: UpstreamServer
    ) -> None:
        """A command that exits non-zero moves the service to unhealthy."""
        svc = replace(
            make_service("svc", target.url),
            health_check_command="/bin/false",
            health_check_interval=1,
        )
        harness = await start_proxy(make_config([svc], free_port()))
        assert await harness.wait_for_health("svc", False) is False
        assert await harness.wait_for_state("svc", "unhealthy") == "unhealthy"
        events = await harness.events("become_unhealthy", "svc")
        assert events[0]["details"] == "exit code 1"

    async def test_recovery_is_announced_once(
        self, start_proxy: StartProxy, target: UpstreamServer, tmp_path: Path
    ) -> None:
        """A service that starts failing and then passes reports both changes once."""
        flag = tmp_path / "healthy"
        svc = replace(
            make_service("svc", target.url),
            health_check_command=f"/usr/bin/test -f {flag}",
            health_check_interval=1,
        )
        harness = await start_proxy(make_config([svc], free_port()))
        assert await harness.wait_for_health("svc", False) is False
        flag.write_text("up")
        assert await harness.wait_for_health("svc", True) is True
        assert len(await harness.events("become_healthy", "svc")) == 1
        assert len(await harness.events("become_unhealthy", "svc")) == 1

    async def test_hanging_command_counts_as_unhealthy(
        self, start_proxy: StartProxy, target: UpstreamServer
    ) -> None:
        """A check that outlives its interval is treated as a failure."""
        svc = replace(
            make_service("svc", target.url),
            health_check_command="/bin/sleep 30",
            health_check_interval=1,
        )
        harness = await start_proxy(make_config([svc], free_port()))
        assert await harness.wait_for_health("svc", False) is False
        events = await harness.events("become_unhealthy", "svc")
        assert events[0]["details"] == "timeout"

    async def test_missing_command_reports_the_exit_code(
        self, start_proxy: StartProxy, target: UpstreamServer
    ) -> None:
        """A command that does not exist fails the check with the shell's code."""
        svc = replace(
            make_service("svc", target.url),
            health_check_command="/definitely/not/a/command",
            health_check_interval=1,
        )
        harness = await start_proxy(make_config([svc], free_port()))
        assert await harness.wait_for_health("svc", False) is False
        events = await harness.events("become_unhealthy", "svc")
        assert events[0]["details"] == "exit code 127"

    async def test_unrunnable_command_marks_the_service_unhealthy(
        self, start_proxy: StartProxy, target: UpstreamServer
    ) -> None:
        """A command the operating system refuses to launch fails the check."""
        svc = replace(
            make_service("svc", target.url),
            health_check_command="/bin/true \x00",
            health_check_interval=1,
        )
        harness = await start_proxy(make_config([svc], free_port()))
        assert await harness.wait_for_health("svc", False) is False
        events = await harness.events("become_unhealthy", "svc")
        assert events[0]["details"].startswith("error ")

    async def test_unhealthy_service_is_bypassed(
        self, start_proxy: StartProxy, target: UpstreamServer, backup: UpstreamServer
    ) -> None:
        """Requests to an unhealthy service go to the backup without trying it."""
        svc = replace(
            make_service("svc", target.url),
            health_check_command="/bin/false",
            health_check_interval=1,
            backup=backup.url,
            gpu_routes=["/v1/completions"],
        )
        harness = await start_proxy(make_config([svc], free_port()))
        assert await harness.wait_for_health("svc", False) is False
        backup.body = b"backup-serving"
        reply = await harness.send("/v1/models", "svc.local")
        assert reply.text == "backup-serving"
        assert len(target.requests) == 0
        assert len(await harness.events("target_unreachable", "svc")) == 1

    @pytest.mark.parametrize("path", ["/v1/models", "/v1/completions"])
    async def test_unhealthy_service_without_backup_returns_503(
        self, start_proxy: StartProxy, target: UpstreamServer, path: str
    ) -> None:
        """With no backup, an unhealthy service answers 503 on any route."""
        svc = replace(
            make_service("svc", target.url),
            health_check_command="/bin/false",
            health_check_interval=1,
            gpu_routes=["/v1/completions"],
        )
        harness = await start_proxy(make_config([svc], free_port()))
        assert await harness.wait_for_health("svc", False) is False
        reply = await harness.send(path, "svc.local")
        assert reply.status == HTTP_SERVICE_UNAVAILABLE
        assert len(target.requests) == 0

    async def test_unhealthy_service_diverts_websockets_too(
        self, start_proxy: StartProxy, target: UpstreamServer, backup: UpstreamServer
    ) -> None:
        """An upgrade to an unhealthy service is relayed by the backup."""
        svc = replace(
            make_service("svc", target.url),
            health_check_command="/bin/false",
            health_check_interval=1,
            backup=backup.url,
        )
        harness = await start_proxy(make_config([svc], free_port()))
        assert await harness.wait_for_health("svc", False) is False
        backup.ws_close_after = 1
        async with harness.websocket("/socket", "svc.local") as ws:
            await ws.send_str("hi")
            assert (await ws.receive()).data == "backup:hi"
        assert len(target.requests) == 0


class TestLifecycle:
    """Starting and stopping the proxy."""

    async def test_uptime_is_zero_before_start(self, target: UpstreamServer) -> None:
        """A proxy that was never started reports no uptime."""
        config = make_config([make_service("svc", target.url)], free_port())
        proxy = GpuPriorityProxy(config, clock=RealClock())
        assert proxy.get_uptime_seconds() == 0

    async def test_stopping_ends_polling_and_health_checks(
        self, start_proxy: StartProxy, target: UpstreamServer, fake_gpu: FakeGpu
    ) -> None:
        """After stop the GPU is no longer polled."""
        svc = replace(
            make_service("svc", target.url),
            health_check_command="/bin/true",
            health_check_interval=1,
        )
        harness = await start_proxy(make_config([svc], free_port()))
        await harness.wait_for_poll()
        await harness.proxy.stop()
        polls = fake_gpu.polls
        await asyncio.sleep(POLL_INTERVAL * 5)
        assert fake_gpu.polls == polls

    async def test_shutdown_does_not_wait_for_in_flight_work(
        self, gpu_service: ProxyHarness, target: UpstreamServer
    ) -> None:
        """Shutdown returns promptly even while a slow request is running."""
        target.response_delay = STALLED_RESPONSE
        pending = asyncio.create_task(
            gpu_service.send("/v1/completions", "svc.local", "POST", b"{}")
        )
        assert await gpu_service.wait_for_in_flight("svc", 1) == 1
        started = asyncio.get_running_loop().time()
        await gpu_service.proxy.shutdown()
        assert asyncio.get_running_loop().time() - started < SHUTDOWN_TIMEOUT
        pending.cancel()

    async def test_shutdown_is_idempotent(self, gpu_service: ProxyHarness) -> None:
        """Stopping twice is harmless: the second call leaves the proxy answering."""
        await gpu_service.proxy.shutdown()
        await gpu_service.proxy.stop()
        data = await gpu_service.dashboard()
        assert data["services"]["svc"]["state"] == "cold"

    async def test_request_after_stop_opens_a_new_connection_pool(
        self, gpu_service: ProxyHarness, target: UpstreamServer
    ) -> None:
        """A request arriving after stop still reaches the target."""
        await gpu_service.proxy.stop()
        target.body = b"still-serving"
        reply = await gpu_service.send("/v1/models", "svc.local")
        assert reply.status == HTTP_OK
        assert reply.text == "still-serving"


def plugin_entry(
    name: str, plugin_type: str, **settings: str | bool
) -> dict[str, str | dict]:
    """Describe one configured plugin instance."""
    return {"name": name, "type": plugin_type, "config": dict(settings)}


StartWithPlugins = Callable[..., Awaitable[tuple[ProxyHarness, PluginManager]]]


@pytest.fixture
def start_with_plugins(start_proxy: StartProxy, tmp_path: Path) -> StartWithPlugins:
    """Factory that starts proxies whose plugin manager holds the given classes."""

    async def _start(
        services: list[ServiceConfig],
        entries: list[dict[str, str | dict]],
        classes: list[type[PluginBase]],
    ) -> tuple[ProxyHarness, PluginManager]:
        config = make_config(services, free_port(), plugins=entries)
        manager = PluginManager(str(tmp_path), config, test_plugins=classes)
        harness = await start_proxy(config, manager)
        return harness, manager

    return _start


class TestRequestPipeline:
    """Plugins rewrite a request before it is forwarded."""

    @pytest.mark.parametrize("suffix", ["-tagged", "-trace-456"])
    async def test_body_written_by_a_plugin_reaches_the_target(
        self,
        start_with_plugins: StartWithPlugins,
        target: UpstreamServer,
        suffix: str,
    ) -> None:
        """A body a plugin writes onto a request without one is sent upstream."""
        svc = replace(make_service("svc", target.url), plugins=["tagger"])
        harness, _ = await start_with_plugins(
            [svc],
            [plugin_entry("tagger", "BodyAppenderPlugin", suffix=suffix)],
            [BodyAppenderPlugin],
        )
        await harness.send("/api", "svc.local")
        assert target.last_request.body == suffix.encode()

    async def test_content_length_matches_the_body_the_target_reads(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer
    ) -> None:
        """The Content-Length the target sees matches its body."""
        svc = replace(make_service("svc", target.url), plugins=["tagger"])
        harness, _ = await start_with_plugins(
            [svc],
            [plugin_entry("tagger", "BodyAppenderPlugin", suffix="-tagged")],
            [BodyAppenderPlugin],
        )
        await harness.send("/api", "svc.local", "POST", b"payload")
        assert target.last_request.body == b"payload-tagged"
        assert target.last_request.headers["Content-Length"] == str(
            len(target.last_request.body)
        )

    async def test_plugins_run_in_configured_order(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer
    ) -> None:
        """Each plugin sees what the previous one produced."""
        svc = replace(make_service("svc", target.url), plugins=["first", "second"])
        harness, _ = await start_with_plugins(
            [svc],
            [
                plugin_entry("first", "BodyAppenderPlugin", suffix="-one"),
                plugin_entry("second", "BodyAppenderPlugin", suffix="-two"),
            ],
            [BodyAppenderPlugin],
        )
        await harness.send("/api", "svc.local")
        assert target.last_request.body == b"-one-two"

    async def test_header_rewrite_reaches_the_target(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer
    ) -> None:
        """A header added by a plugin is sent upstream."""
        svc = replace(make_service("svc", target.url), plugins=["headers"])
        harness, _ = await start_with_plugins(
            [svc],
            [plugin_entry("headers", "HeaderPlugin")],
            [HeaderPlugin],
        )
        await harness.send("/api", "svc.local", "POST", b"{}")
        assert target.last_request.headers["X-Modified"] == "true"

    async def test_rewritten_body_and_length_agree(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer
    ) -> None:
        """A plugin that grows the body also corrects Content-Length."""
        svc = replace(make_service("svc", target.url), plugins=["length"])
        harness, _ = await start_with_plugins(
            [svc],
            [plugin_entry("length", "ContentLengthPlugin")],
            [ContentLengthPlugin],
        )
        await harness.send("/api", "svc.local", "POST", b'{"original": "body"}')
        assert target.last_request.body == b'{"original": "body", "injected": true}'
        assert target.last_request.headers["Content-Length"] == str(
            len(target.last_request.body)
        )

    async def test_service_without_plugins_is_untouched(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer
    ) -> None:
        """A configured plugin only runs for services that list it."""
        svc = make_service("svc", target.url)
        harness, _ = await start_with_plugins(
            [svc],
            [plugin_entry("tagger", "BodyAppenderPlugin", suffix="-tagged")],
            [BodyAppenderPlugin],
        )
        await harness.send("/api", "svc.local", "POST", b"payload")
        assert target.last_request.body == b"payload"

    async def test_failing_plugin_does_not_break_the_request(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer
    ) -> None:
        """A plugin that raises is skipped and the request still goes out."""
        svc = replace(make_service("svc", target.url), plugins=["broken"])
        harness, _ = await start_with_plugins(
            [svc],
            [plugin_entry("broken", "FailingPlugin")],
            [FailingPlugin],
        )
        reply = await harness.send("/api", "svc.local", "POST", b"payload")
        assert reply.status == HTTP_OK
        assert target.last_request.body == b"payload"

    async def test_dashboard_is_outside_the_pipeline(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer
    ) -> None:
        """Dashboard requests never enter a service's plugin pipeline."""
        svc = replace(make_service("svc", target.url), plugins=["broken"])
        harness, _ = await start_with_plugins(
            [svc],
            [plugin_entry("broken", "FailingPlugin")],
            [FailingPlugin],
        )
        data = await harness.dashboard()
        assert data["services"]["svc"]["state"] == "cold"

    async def test_shutdown_tells_plugins_the_request_was_dropped(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer
    ) -> None:
        """A request cut short by shutdown is reported to its plugins."""
        svc = replace(make_service("svc", target.url), plugins=["tracker"])
        harness, manager = await start_with_plugins(
            [svc],
            [plugin_entry("tracker", "CancelTrackingPlugin")],
            [CancelTrackingPlugin],
        )
        target.response_delay = STALLED_RESPONSE
        pending = asyncio.create_task(harness.send("/api", "svc.local", "POST", b"{}"))
        await wait_until(lambda: bool(target.requests))
        await harness.stop_serving()
        pending.cancel()
        plugin = manager.get_plugin_instance("svc", "tracker")
        assert plugin.cancelled_requests == [1]


class TestPrepareTargetPipeline:
    """prepare_target runs right before traffic reaches the service's target."""

    async def test_hook_runs_when_the_target_is_served(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer
    ) -> None:
        """A request served by the target prepares the target exactly once."""
        svc = replace(make_service("svc", target.url), plugins=["prep"])
        harness, manager = await start_with_plugins(
            [svc],
            [plugin_entry("prep", "PrepareTargetRecorder")],
            [PrepareTargetRecorder],
        )
        reply = await harness.send("/api", "svc.local", "POST", b"payload")
        assert reply.status == HTTP_OK
        plugin = manager.get_plugin_instance("svc", "prep")
        assert plugin.prepared == [(1, "/api")]
        assert target.last_request.body == b"payload"

    async def test_hook_runs_after_admission_on_a_gpu_route(
        self,
        start_with_plugins: StartWithPlugins,
        target: UpstreamServer,
        backup: UpstreamServer,
    ) -> None:
        """On a GPU route the hook fires only once the request was admitted."""
        svc = replace(
            make_service("svc", target.url, vram_budget=SMALL_SERVICE_VRAM),
            gpu_routes=["/v1/completions"],
            backup=backup.url,
            plugins=["prep"],
        )
        harness, manager = await start_with_plugins(
            [svc],
            [plugin_entry("prep", "PrepareTargetRecorder")],
            [PrepareTargetRecorder],
        )
        await harness.wait_for_poll()
        reply = await harness.send("/v1/completions", "svc.local", "POST", b"{}")
        assert reply.status == HTTP_OK
        plugin = manager.get_plugin_instance("svc", "prep")
        assert plugin.prepared == [(1, "/v1/completions")]

    async def test_hook_is_skipped_when_the_service_is_unhealthy(
        self,
        start_with_plugins: StartWithPlugins,
        target: UpstreamServer,
        backup: UpstreamServer,
    ) -> None:
        """Traffic diverted to the backup never prepares the stopped target."""
        backup.body = b"served-by-backup"
        svc = replace(
            make_service("svc", target.url),
            backup=backup.url,
            plugins=["prep"],
        )
        harness, manager = await start_with_plugins(
            [svc],
            [plugin_entry("prep", "PrepareTargetRecorder")],
            [PrepareTargetRecorder],
        )
        state = harness.proxy.service_states["svc"]
        state.is_healthy = False
        state.transition(ServiceStateEnum.unhealthy)
        reply = await harness.send("/api", "svc.local", "POST", b"payload")
        assert reply.status == HTTP_OK
        assert reply.text == "served-by-backup"
        plugin = manager.get_plugin_instance("svc", "prep")
        assert plugin.prepared == []

    async def test_failing_hook_does_not_break_the_request(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer
    ) -> None:
        """A plugin that raises in prepare_target is logged, and the request goes out."""
        svc = replace(make_service("svc", target.url), plugins=["prep"])
        harness, _ = await start_with_plugins(
            [svc],
            [plugin_entry("prep", "PrepareTargetRecorder", fail=True)],
            [PrepareTargetRecorder],
        )
        reply = await harness.send("/api", "svc.local", "POST", b"payload")
        assert reply.status == HTTP_OK
        assert target.last_request.body == b"payload"


class TestCancelledForward:
    """A forward cancelled mid-flight is treated as a cancellation, not a failure."""

    async def test_cancelled_forward_is_not_recorded_as_failure(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer
    ) -> None:
        """Cancelling a target forward does not record a target failure."""
        svc = replace(make_service("svc", target.url), plugins=["canceller"])
        harness, _ = await start_with_plugins(
            [svc],
            [plugin_entry("canceller", "CancelledForwardPlugin")],
            [CancelledForwardPlugin],
        )
        target.response_delay = STALLED_RESPONSE
        reply = await harness.send("/cancel-forward", "svc.local", "POST", b"{}")
        assert reply.status == HTTP_OK
        assert await harness.events("target_failed", "svc") == []

    async def test_cancelled_backup_forward_is_not_recorded_as_failure(
        self, start_with_plugins: StartWithPlugins, backup: UpstreamServer
    ) -> None:
        """Cancelling a backup retry after a dead target does not record a backup failure."""
        svc = replace(
            make_service("svc", f"http://{LOCALHOST}:{free_port()}"),
            backup=backup.url,
            plugins=["canceller"],
        )
        harness, _ = await start_with_plugins(
            [svc],
            [plugin_entry("canceller", "CancelledForwardPlugin")],
            [CancelledForwardPlugin],
        )
        backup.response_delay = STALLED_RESPONSE
        reply = await harness.send("/cancel-forward", "svc.local", "POST", b"{}")
        assert reply.status == HTTP_OK
        assert await harness.events("backup_failed", "svc") == []

    async def test_cancelled_diverted_forward_is_not_recorded_as_failure(
        self,
        start_with_plugins: StartWithPlugins,
        target: UpstreamServer,
        backup: UpstreamServer,
        fake_gpu: FakeGpu,
    ) -> None:
        """Cancelling a request diverted straight to backup does not record a failure."""
        svc = replace(
            make_service("svc", target.url),
            backup=backup.url,
            gpu_routes=["/cancel-forward"],
            plugins=["canceller"],
        )
        harness, _ = await start_with_plugins(
            [svc],
            [plugin_entry("canceller", "CancelledForwardPlugin")],
            [CancelledForwardPlugin],
        )
        await harness.wait_for_poll()
        fake_gpu.poll_fails = True
        await harness.wait_for_dashboard(
            lambda snapshot: snapshot["proxy"]["health"] == "unhealthy"
        )
        backup.response_delay = STALLED_RESPONSE
        reply = await harness.send("/cancel-forward", "svc.local", "POST", b"{}")
        assert reply.status == HTTP_OK
        assert target.requests == []
        assert await harness.events("backup_failed", "svc") == []


class ResponseHeaderPlugin(HeaderPlugin):
    """Add a mark to the response headers."""

    @override
    def on_response_headers(
        self, _request_id: int, headers: dict[str, str], status_code: int
    ) -> dict[str, str]:
        headers["X-Response-Steps"] = headers.get("X-Response-Steps", "") + self.config["mark"]
        headers["X-Mutator-Status"] = str(status_code)
        return headers


class TestResponsePipeline:
    """Plugins see, and may rewrite, the response on its way back."""

    async def test_every_chunk_is_offered_then_a_final_marker(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer, tmp_path: Path
    ) -> None:
        """Each streamed chunk is passed to the plugin, then a closing call."""
        log = tmp_path / "chunks.json"
        log.write_text("[]")
        svc = replace(make_service("svc", target.url), plugins=["capture"])
        harness, _ = await start_with_plugins(
            [svc],
            [plugin_entry("capture", "ChunkCapturePlugin", call_file=str(log))],
            [ChunkCapturePlugin],
        )
        target.chunks = [b"one", b"two", b"three"]
        target.chunk_gap = BRIEF_CHUNK_GAP
        await harness.send("/v1/chat/completions", "svc.local", "POST", b"{}")
        calls = json.loads(log.read_text())
        assert [call["chunk"] for call in calls if not call["is_final"]] == [
            "one",
            "two",
            "three",
        ]
        assert calls[-1]["is_final"] is True
        assert calls[-1]["chunk"] == ""
        assert calls[0]["status_code"] == HTTP_OK
        assert calls[0]["path"] == "/v1/chat/completions"

    async def test_rewritten_chunks_reach_the_client(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer
    ) -> None:
        """What the plugin returns is what the client reads."""
        svc = replace(make_service("svc", target.url), plugins=["mutator"])
        harness, _ = await start_with_plugins(
            [svc],
            [plugin_entry("mutator", "MutatorPlugin")],
            [MutatorPlugin],
        )
        target.chunks = [b"data: original\n\n"]
        reply = await harness.send("/v1/chat/completions", "svc.local", "POST", b"{}")
        marker = sse_event({"choices": [{"delta": {"content": "MUTATED"}}]})
        assert reply.body.startswith(marker)
        assert b"data: original\n\n" in reply.body

    async def test_websocket_messages_are_offered(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer, tmp_path: Path
    ) -> None:
        """Messages relayed back over a socket also pass through the pipeline."""
        log = tmp_path / "chunks.json"
        log.write_text("[]")
        svc = replace(make_service("svc", target.url), plugins=["capture"])
        harness, _ = await start_with_plugins(
            [svc],
            [plugin_entry("capture", "ChunkCapturePlugin", call_file=str(log))],
            [ChunkCapturePlugin],
        )
        target.ws_close_after = 1
        async with harness.websocket("/socket", "svc.local") as ws:
            await ws.send_str("hello")
            assert (await ws.receive()).data == "target:hello"
        calls = json.loads(log.read_text())
        assert [call["chunk"] for call in calls] == ["target:hello"]
        assert calls[0]["path"] == "/socket"
        assert calls[0]["status_code"] == HTTP_SWITCHING_PROTOCOLS

    @pytest.mark.parametrize("is_guarded", [False, True])
    async def test_nested_header_changes_run_inside_to_outside(
        self, start_proxy: StartProxy, target: UpstreamServer, tmp_path: Path,
        is_guarded: bool,
    ) -> None:
        """Inner response headers reach the outer plugin before they are sent."""
        port = free_port()
        raw = replace(make_service("raw", target.url), plugins=["inner"], gpu_routes=[])
        outer_plugins = ["guard", "outer"] if is_guarded else ["outer"]
        outer = replace(
            make_service("llm", f"http://raw.local:{port}"),
            plugins=outer_plugins, external=True,
        )
        config = make_config([outer, raw], port, plugins=[
            plugin_entry("inner", "ResponseHeaderPlugin", mark="-inner"),
            plugin_entry("outer", "ResponseHeaderPlugin", mark="-outer"),
            plugin_entry("guard", "OpenAIGuardRail"),
        ])
        manager = PluginManager(
            str(tmp_path), config, test_plugins=[ResponseHeaderPlugin, OpenAIGuardRail],
        )
        harness = await start_proxy(config, manager)
        target.response_headers = {"X-Response-Steps": "upstream"}
        target.chunks = [sse_delta_content("msg1", "Hello"), sse_done()]
        reply = await harness.send(
            "/v1/chat/completions", "llm.local", "POST",
            b'{"messages": [], "stream": false}',
        )
        assert reply.status == HTTP_OK
        assert reply.headers["X-Response-Steps"] == "upstream-inner-outer"
        assert reply.headers["X-Mutator-Status"] == str(HTTP_OK)
        if is_guarded:
            assert reply.headers["Content-Type"] == "application/json"
            assert json.loads(reply.body)["choices"][0]["message"]["content"] == "Hello"
        else:
            assert reply.body == b"".join(target.chunks)

    async def test_header_changes_reach_the_websocket_handshake(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer,
    ) -> None:
        """The socket handshake includes the response header changes."""
        svc = replace(make_service("svc", target.url), plugins=["headers"], gpu_routes=[])
        harness, _ = await start_with_plugins(
            [svc], [plugin_entry("headers", "ResponseHeaderPlugin", mark="socket")],
            [ResponseHeaderPlugin],
        )
        reply = await harness.send("/socket", "svc.local", headers={
            **WS_UPGRADE_HEADERS,
            "Sec-WebSocket-Version": "13",
            "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==",
        })
        assert reply.status == HTTP_SWITCHING_PROTOCOLS
        assert reply.headers["X-Response-Steps"] == "socket"
        assert reply.headers["X-Mutator-Status"] == str(HTTP_SWITCHING_PROTOCOLS)


@pytest.fixture
def cached_forward_slot() -> Iterator[None]:
    """Empty CacheForwardPlugin's shared slot around a test, keeping it isolated."""
    CacheForwardPlugin.cached_forward = None
    yield
    CacheForwardPlugin.cached_forward = None


class TestPathOverride:
    """Plugins may claim a path and answer it themselves."""

    async def test_plugin_answers_the_claimed_path(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer
    ) -> None:
        """The plugin's response is served and the service is never contacted."""
        svc = replace(make_service("svc", target.url), plugins=["custom"])
        harness, _ = await start_with_plugins(
            [svc],
            [plugin_entry("custom", "CustomResponsePlugin")],
            [CustomResponsePlugin],
        )
        reply = await harness.send("/custom", "svc.local")
        assert reply.status == HTTP_OK
        assert reply.text == "custom response"
        assert target.requests == []

    async def test_other_paths_still_reach_the_service(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer
    ) -> None:
        """Claiming one path leaves every other path alone."""
        svc = replace(make_service("svc", target.url), plugins=["custom"])
        harness, _ = await start_with_plugins(
            [svc],
            [plugin_entry("custom", "CustomResponsePlugin")],
            [CustomResponsePlugin],
        )
        target.body = b"from-service"
        reply = await harness.send("/other", "svc.local")
        assert reply.text == "from-service"
        assert target.last_request.path == "/other"

    async def test_plugin_can_delegate_to_the_service(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer
    ) -> None:
        """A plugin that forwards gets the service's own answer."""
        svc = replace(make_service("svc", target.url), plugins=["delegate"])
        harness, _ = await start_with_plugins(
            [svc],
            [plugin_entry("delegate", "DelegatePlugin")],
            [DelegatePlugin],
        )
        target.body = b"delegated-answer"
        reply = await harness.send("/delegate", "svc.local")
        assert reply.text == "delegated-answer"
        assert target.last_request.path == "/delegate"

    async def test_delegated_request_can_fall_back_to_the_backup(
        self, start_with_plugins: StartWithPlugins, backup: UpstreamServer
    ) -> None:
        """Delegation runs the whole routing pipeline, backup included."""
        svc = replace(
            make_service("svc", f"http://{LOCALHOST}:{free_port()}"),
            backup=backup.url,
            plugins=["delegate"],
        )
        harness, _ = await start_with_plugins(
            [svc],
            [plugin_entry("delegate", "ForwardToBackupPlugin")],
            [ForwardToBackupPlugin],
        )
        backup.body = b"backup-answered"
        reply = await harness.send("/forward-to-backup", "svc.local")
        assert reply.text == "backup-answered"

    async def test_delegated_request_still_passes_the_gpu_gate(
        self,
        start_with_plugins: StartWithPlugins,
        target: UpstreamServer,
        backup: UpstreamServer,
        fake_gpu: FakeGpu,
    ) -> None:
        """A delegated GPU route is gated like any other GPU route."""
        svc = replace(
            make_service("svc", target.url),
            gpu_routes=["/v1/completions"],
            backup=backup.url,
            plugins=["gpu-delegate"],
        )
        harness, _ = await start_with_plugins(
            [svc],
            [plugin_entry("gpu-delegate", "GpuDelegatePlugin")],
            [GpuDelegatePlugin],
        )
        fake_gpu.poll_fails = True
        await harness.wait_for_dashboard(
            lambda snapshot: snapshot["proxy"]["health"] == "unhealthy"
        )
        backup.body = b"gated-to-backup"
        reply = await harness.send("/v1/completions", "svc.local", "POST", b"{}")
        assert reply.text == "gated-to-backup"
        assert target.requests == []

    @pytest.mark.usefixtures("cached_forward_slot")
    async def test_forwarding_is_refused_after_the_handler_returned(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer
    ) -> None:
        """A cached forward callable cannot be used once the request is over."""
        svc = replace(make_service("svc", target.url), plugins=["cache"])
        harness, _ = await start_with_plugins(
            [svc],
            [plugin_entry("cache", "CacheForwardPlugin")],
            [CacheForwardPlugin],
        )
        reply = await harness.send("/cache-forward", "svc.local")
        assert reply.status == HTTP_OK
        with pytest.raises(RuntimeError, match=r"forward\(\) can only be called"):
            await CacheForwardPlugin.cached_forward({}, b"", "/x", None, None)


class TestPluginDashboardPanels:
    """Panels plugins publish for the dashboard."""

    async def test_published_panel_appears(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer
    ) -> None:
        """A publishing plugin adds its panel to the snapshot."""
        svc = replace(make_service("svc", target.url), plugins=["panel"])
        harness, _ = await start_with_plugins(
            [svc],
            [
                plugin_entry(
                    "panel", "UIPublisherPlugin", title="Token Usage", content="<b>7</b>"
                )
            ],
            [UIPublisherPlugin],
        )
        data = await harness.dashboard()
        assert data["plugin_uis"]["svc"] == [
            {"title": "Token Usage", "html": "<b>7</b>"}
        ]

    async def test_snapshot_omits_panels_when_none_are_published(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer
    ) -> None:
        """A plugin that publishes nothing adds no key to the snapshot."""
        svc = replace(make_service("svc", target.url), plugins=["quiet"])
        harness, _ = await start_with_plugins(
            [svc],
            [plugin_entry("quiet", "BodyAppenderPlugin", suffix="")],
            [BodyAppenderPlugin],
        )
        assert "plugin_uis" not in await harness.dashboard()

    async def test_snapshot_omits_panels_without_plugins(
        self, gpu_service: ProxyHarness
    ) -> None:
        """A proxy running without a plugin manager reports no panels."""
        assert "plugin_uis" not in await gpu_service.dashboard()

    async def test_each_service_gets_its_own_panels(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer
    ) -> None:
        """Panels are grouped by the service that published them."""
        first = replace(make_service("one", target.url), plugins=["panel-one"])
        second = replace(
            make_service("two", target.url, priority=2), plugins=["panel-two"]
        )
        harness, _ = await start_with_plugins(
            [first, second],
            [
                plugin_entry("panel-one", "UIPublisherPlugin", title="Tokens", content="1"),
                plugin_entry("panel-two", "UIPublisherPlugin", title="Stats", content="2"),
            ],
            [UIPublisherPlugin],
        )
        panels = (await harness.dashboard())["plugin_uis"]
        assert panels["one"] == [{"title": "Tokens", "html": "1"}]
        assert panels["two"] == [{"title": "Stats", "html": "2"}]


_COMPLETION_PLUGIN_TYPES: list[tuple[str, type[PluginBase]]] = [
    ("OpenAITokenStats", OpenAITokenStats),
    ("OpenAIGuardRail", OpenAIGuardRail),
]


class TestCompletionResponseHeaders:
    """Completion headers describe the format the client requested."""

    @pytest.mark.parametrize("plugin_name,plugin_type,is_internal", [
        ("OpenAITokenStats", OpenAITokenStats, False),
        ("OpenAIGuardRail", OpenAIGuardRail, False),
        ("OpenAITokenStats", OpenAITokenStats, True),
    ])
    @pytest.mark.parametrize("stream", [None, False, True])
    @pytest.mark.parametrize("content_type_header", ["Content-Type", "content-type"])
    async def test_content_type_matches_client_stream_setting(
        self, start_proxy: StartProxy, target: UpstreamServer, tmp_path: Path,
        plugin_name: str, plugin_type: type[PluginBase], stream: Optional[bool],
        content_type_header: str, is_internal: bool,
    ) -> None:
        """Full replies use JSON and streams keep their original headers."""
        port = free_port()
        raw = replace(make_service("raw", target.url), gpu_routes=[])
        svc = replace(
            make_service("llm", f"http://raw.local:{port}" if is_internal else target.url),
            plugins=["completion"], gpu_routes=[],
        )
        config = make_config([svc, raw] if is_internal else [svc], port, plugins=[
            plugin_entry("completion", plugin_name),
        ])
        manager = PluginManager(str(tmp_path), config, test_plugins=[plugin_type])
        harness = await start_proxy(config, manager)
        target.response_headers = {
            content_type_header: "text/event-stream; charset=utf-8",
            "X-Upstream-Trace": "keep-me",
        }
        target.chunks = [sse_delta_content("msg1", "Hello"), sse_done()]
        body = {"model": "m", "messages": []}
        if stream is not None:
            body["stream"] = stream
        async with harness.session.post(
            f"{harness.base_url}/v1/chat/completions",
            headers={"Host": "llm.local"}, json=body,
        ) as response:
            assert response.status == HTTP_OK
            assert response.headers["X-Upstream-Trace"] == "keep-me"
            assert json.loads(target.last_request.body)["stream"] is True
            if stream:
                assert response.headers.getall("Content-Type") == [
                    "text/event-stream; charset=utf-8"
                ]
                reply_body = await response.read()
                assert extract_sse_field(reply_body.removesuffix(sse_done())) == "Hello"
            else:
                assert response.headers.getall("Content-Type") == ["application/json"]
                reply_json = await response.json()
                assert reply_json["choices"][0]["message"]["content"] == "Hello"

    @pytest.mark.parametrize("plugin_name,plugin_type", _COMPLETION_PLUGIN_TYPES)
    async def test_error_content_type_is_preserved(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer,
        plugin_name: str, plugin_type: type[PluginBase],
    ) -> None:
        """An upstream error keeps its content type and body."""
        svc = replace(make_service("llm", target.url), plugins=["completion"], gpu_routes=[])
        harness, _ = await start_with_plugins(
            [svc], [plugin_entry("completion", plugin_name)], [plugin_type],
        )
        target.status = HTTP_INTERNAL_SERVER_ERROR
        target.content_type = "text/plain"
        target.body = b"upstream exploded"
        reply = await harness.send(
            "/v1/chat/completions", "llm.local", "POST", b'{"messages": [], "stream": false}',
        )
        assert reply.status == HTTP_INTERNAL_SERVER_ERROR
        assert reply.headers["Content-Type"] == "text/plain"
        assert reply.body == target.body

    @pytest.mark.parametrize("stream", [False, True])
    async def test_nested_guard_and_token_routes_keep_client_format(
        self, start_proxy: StartProxy, target: UpstreamServer, tmp_path: Path,
        stream: bool,
    ) -> None:
        """Nested completion routes keep the format the client requested."""
        port = free_port()
        raw = replace(make_service("raw", target.url), plugins=["tokens"], gpu_routes=[])
        guarded = replace(
            make_service("llm", f"http://raw.local:{port}"),
            plugins=["guard"], external=True,
        )
        config = make_config([guarded, raw], port, plugins=[
            plugin_entry("guard", "OpenAIGuardRail"),
            plugin_entry("tokens", "OpenAITokenStats"),
        ])
        manager = PluginManager(
            str(tmp_path), config, test_plugins=[OpenAIGuardRail, OpenAITokenStats],
        )
        harness = await start_proxy(config, manager)
        target.chunks = [sse_delta_content("msg1", "Hello"), sse_done()]
        reply = await harness.send(
            "/v1/chat/completions", "llm.local", "POST",
            json.dumps({"messages": [], "stream": stream}).encode(),
        )
        assert reply.status == HTTP_OK
        if stream:
            assert reply.headers["Content-Type"] == "text/event-stream"
            assert extract_sse_field(reply.body.removesuffix(sse_done())) == "Hello"
        else:
            assert reply.headers["Content-Type"] == "application/json"
            assert json.loads(reply.body)["choices"][0]["message"]["content"] == "Hello"


class TestTokenAccounting:
    """The shipped token-stats plugin against a real streamed response."""

    async def test_usage_reported_by_the_model_is_summarised(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer
    ) -> None:
        """Counts from the stream reach the panel, and the client gets its answer.

        The plugin asks the model to stream so that it can read the usage block,
        then reassembles the reply for a client that did not ask for a stream.
        """
        svc = replace(
            make_service("llm", target.url), plugins=["tokens"], gpu_routes=[]
        )
        harness, _ = await start_with_plugins(
            [svc],
            [plugin_entry("tokens", "OpenAITokenStats")],
            [OpenAITokenStats],
        )
        target.chunks = [
            sse_delta_content("msg1", "Hello"),
            sse_usage(12, 8),
            sse_done(),
        ]
        reply = await harness.send(
            "/v1/chat/completions",
            "llm.local",
            "POST",
            json.dumps({"model": "m", "messages": []}).encode(),
        )
        completion = json.loads(reply.text)
        assert completion["object"] == "chat.completion"
        assert completion["choices"][0]["message"]["content"] == "Hello"
        data = await harness.wait_for_dashboard(lambda snapshot: "12 in" in token_panel(snapshot))
        assert "8 out" in token_panel(data)
        assert "Stats" in token_panel(data)

    async def test_paths_without_completions_are_not_counted(
        self, start_with_plugins: StartWithPlugins, target: UpstreamServer
    ) -> None:
        """A response on an unrelated path leaves the counters at zero."""
        svc = replace(
            make_service("llm", target.url), plugins=["tokens"], gpu_routes=[]
        )
        harness, _ = await start_with_plugins(
            [svc],
            [plugin_entry("tokens", "OpenAITokenStats")],
            [OpenAITokenStats],
        )
        target.chunks = [sse_usage(999, 888), sse_done()]
        await harness.send("/v1/models", "llm.local", "POST", b"{}")
        data = await harness.wait_for_dashboard(lambda snapshot: bool(token_panel(snapshot)))
        assert "Stats" not in token_panel(data)
        assert "999" not in token_panel(data)


def token_panel(snapshot: dict) -> str:
    """All panel markup the llm service's plugins have published."""
    return "".join(
        panel["html"] for panel in snapshot.get("plugin_uis", {}).get("llm", [])
    )


class TestConcurrency:
    """Independent services keep serving while one is under pressure."""

    async def test_requests_to_other_services_are_not_blocked(
        self, start_busy_eviction: StartBusyEviction
    ) -> None:
        """A slow eviction on one service does not hold up another."""
        pending = await start_busy_eviction()
        passthrough = await pending.setup.harness.send("/v1/models", "high.local")
        assert passthrough.status == HTTP_OK
        assert (await pending.busy).status == HTTP_OK
        assert (await pending.evicting).status == HTTP_OK

    async def test_parallel_requests_share_one_budget_decision(
        self, gpu_service: ProxyHarness, target: UpstreamServer
    ) -> None:
        """Four simultaneous GPU requests all complete and settle the counter."""
        target.response_delay = QUICK_RESPONSE
        replies = await asyncio.gather(
            *[
                gpu_service.send("/v1/completions", "svc.local", "POST", b"{}")
                for _ in range(4)
            ]
        )
        assert [reply.status for reply in replies] == [HTTP_OK] * 4
        assert await gpu_service.wait_for_in_flight("svc", 0) == 0
        assert (await gpu_service.routing("svc"))["target"] == 4

    async def test_several_services_are_served_at_once(
        self, start_proxy: StartProxy, target: UpstreamServer
    ) -> None:
        """Requests to different hosts are handled independently."""
        first = replace(make_service("one", target.url, priority=1), gpu_routes=["/v1"])
        second = replace(make_service("two", target.url, priority=2), gpu_routes=["/v1"])
        harness = await start_proxy(make_config([first, second], free_port()))
        await harness.wait_for_poll()
        target.response_delay = QUICK_RESPONSE
        replies = await asyncio.gather(
            harness.send("/v1/go", "one.local", "POST", b"{}"),
            harness.send("/v1/go", "two.local", "POST", b"{}"),
            harness.send("/other", "one.local"),
        )
        assert [reply.status for reply in replies] == [HTTP_OK, HTTP_OK, HTTP_OK]
        assert (await harness.routing("one"))["target"] == 2
        assert (await harness.routing("two"))["target"] == 1
        assert (await harness.routing("one"))["backup"] == 0
