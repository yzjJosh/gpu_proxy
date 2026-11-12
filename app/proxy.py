"""Main proxy application with request routing and dashboard."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Optional, TYPE_CHECKING
from aiohttp import (
    web,
    ClientSession,
    ClientTimeout,
    ClientWebSocketResponse,
    WSMsgType,
    TCPConnector,
    WSServerHandshakeError,
)
from yarl import URL

from app.config import ProxyConfig, ServiceConfig
from app.responses import (
    AbstractMutatingResponse,
    MutatingStreamResponse,
    MutatingWebSocketResponse,
    identity_mutator,
)
from app.gpu_monitor import ClockProtocol, GpuMonitor
from app.gpu_monitor import GPUStatus, ContainerVRAM
from app.service_state import ServiceState, ServiceStateEnum
from app.eviction import EvictionLog, EvictionEventType
from app.wait_queue import VRAMWaitQueue
from app.budget import compute_effective_remaining, get_eviction_candidates
from app.plugins.base import forward_handler, make_forward_guard
from urllib.parse import urlparse

if TYPE_CHECKING:
    from app.plugins.manager import PluginManager

__all__ = ["GpuPriorityProxy"]

_logger = logging.getLogger("gpu_proxy")


def _is_gpu_route(service: ServiceConfig, path: str) -> bool:
    """Check if a request path matches any GPU route prefix for a service."""
    if not service.gpu_routes:
        return False
    for route in service.gpu_routes:
        if path.startswith(route):
            return True
    return False


_DASHBOARD_DIST = Path(__file__).parent.parent / "dashboard" / "dist"
_DASHBOARD_LOG_LIMIT = 20
_DASHBOARD_UPDATE_INTERVAL = 0.2
_MAX_INTERNAL_RECURSION_DEPTH = 3
_HTTP_OK = 200
_HTTP_NOT_FOUND = 404
_HTTP_METHOD_NOT_ALLOWED = 405
_HTTP_SERVICE_UNAVAILABLE = 503
_HTTP_LOOP_DETECTED = 508
_HTTP_INTERNAL_SERVER_ERROR = 500
# Hop-by-hop headers not forwarded on responses
_HOP_BY_HOP_HEADERS = frozenset(("transfer-encoding", "connection", "keep-alive"))
# Framing headers not forwarded on requests.
_REQUEST_FRAMING_HEADERS = frozenset(
    ("transfer-encoding", "content-length", "keep-alive", "proxy-connection")
)


class GpuPriorityProxy:
    """GPU priority reverse proxy with eviction and budget-aware routing."""

    def __init__(
        self,
        config: ProxyConfig,
        clock: ClockProtocol,
        plugin_manager: Optional[PluginManager] = None,
    ) -> None:
        self.config = config
        self._plugin_manager = plugin_manager
        self.gpu_status: GPUStatus = GPUStatus()
        self.eviction_log: EvictionLog = EvictionLog()
        self.service_states: dict[str, ServiceState] = {
            name: ServiceState(name, clock=clock, eviction_log=self.eviction_log)
            for name in config.services
        }
        self.routing_stats: dict[str, dict[str, int]] = {
            name: {"target": 0, "backup": 0, "evictions": 0} for name in config.services
        }
        self.wait_queue: VRAMWaitQueue = VRAMWaitQueue(
            clock, timeout=config.wait_queue_timeout
        )
        self._clock = clock
        self._app: Optional[web.Application] = None
        self._start_time: Optional[float] = None
        self._dashboard_clients: set[web.WebSocketResponse] = set()
        self._last_dashboard_push: float = 0.0
        self._dashboard_update_handle: Optional[asyncio.TimerHandle] = None
        self._gpu_monitor: Optional[GpuMonitor] = None
        self._sessions: dict[str, ClientSession] = {}
        self._health_check_tasks: dict[str, asyncio.Task] = {}
        self._next_request_id = 1

    def set_plugin_manager(self, plugin_manager: PluginManager) -> None:
        """Set the plugin manager after proxy construction."""
        self._plugin_manager = plugin_manager

    @staticmethod
    def is_websocket_upgrade(headers: Optional[dict]) -> bool:
        """Check if request headers indicate a WebSocket upgrade.

        Per RFC 6455, a valid WebSocket upgrade requires both
        Upgrade: websocket and Connection: Upgrade headers.

        Args:
            headers: Request headers dict (case-insensitive lookup).

        Returns:
            True if the headers indicate a WebSocket upgrade request.
        """
        if not headers:
            return False

        def _get_header_value(key: str) -> str:
            """Get header value, handling both string and list types."""
            val = headers.get(key, "") or headers.get(key.lower(), "")
            if isinstance(val, list):
                return "; ".join(val)
            return str(val)

        upgrade = _get_header_value("Upgrade")
        connection = _get_header_value("Connection")
        return upgrade.lower() == "websocket" and "upgrade" in connection.lower()

    def _make_session(self) -> ClientSession:
        return ClientSession(
            timeout=ClientTimeout(
                sock_connect=self.config.connection_timeout,
                sock_read=self.config.sock_read_timeout,
            ),
            connector=TCPConnector(limit=100, ttl_dns_cache=300),
        )

    def _get_session(self, service_name: str) -> ClientSession:
        if service_name not in self._sessions:
            self._sessions[service_name] = self._make_session()
        session = self._sessions[service_name]
        if session.closed:
            self._sessions[service_name] = self._make_session()
            session = self._sessions[service_name]
        return session

    def _detect_loopback(
        self, target_url: Optional[str], inbound_host: str
    ) -> tuple[Optional[str], bool]:
        """Check if a target URL points back to our own listen port.

        Args:
            target_url: The outbound URL to check for loopback.
            inbound_host: The hostname of the incoming request (stripped of port).

        Returns:
            Tuple of (redirect_host, is_self_loop). If redirect_host is not None,
            the request should be internally redirected. If is_self_loop is True,
            the request loops back to the same host and should be aborted.
        """
        parsed_url = URL(target_url)
        target_port = parsed_url.port or {
            "http": 80,
            "ws": 80,
            "https": 443,
            "wss": 443,
        }.get(parsed_url.scheme, 80)
        if target_port != self.config.listen_port:
            return None, False
        new_host = parsed_url.host or ""
        if new_host.lower() == inbound_host.lower():
            return None, True
        return new_host, False

    async def _handle_loopback(
        self,
        target_url: Optional[str],
        host: str,
        path: str,
        method: str,
        web_request: web.Request,
        response: AbstractMutatingResponse,
        headers: Optional[dict],
        body: Optional[bytes],
        recursion_depth: int,
        commit_on_error: bool = True,
    ) -> Optional[bool]:
        """Check for loopback and handle internally or abort.

        Returns True if handled successfully.
        Returns False if error committed.
        Returns None if not handled (no loopback, or error not committed).
        """
        redirect_host, is_self_loop = self._detect_loopback(target_url, host)
        if is_self_loop:
            _logger.warning("Self-loop detected for host %s", host)
            if commit_on_error:
                await self._fail_immediately(
                    web_request,
                    response,
                    _HTTP_LOOP_DETECTED,
                    "Loop Detected",
                )
            return False  # loopback detected, committed or not
        if redirect_host is not None:
            _logger.info(
                "Internal redirection: %s -> %s (depth %d)",
                host,
                redirect_host,
                recursion_depth,
            )
            return await self._handle_request(
                host=redirect_host,
                path=path,
                method=method,
                web_request=web_request,
                response=response,
                headers=headers,
                body=body,
                recursion_depth=recursion_depth + 1,
                commit_on_error=commit_on_error,
            )
        return None

    async def start(self) -> None:
        """Start the proxy server and background GPU monitor."""
        self._start_time = self._clock.time()
        for name in self.config.services:
            self._sessions[name] = self._make_session()
        self._gpu_monitor = GpuMonitor(
            poll_interval=self.config.poll_interval,
            on_update=self._on_gpu_update,
            clock=self._clock,
        )
        await self._gpu_monitor.start()

        # Start health check loops
        for name, svc in self.config.services.items():
            if svc.health_check_command:
                # Initial check to ensure health is known before accepting traffic
                await self._run_health_check(name)
                self._health_check_tasks[name] = asyncio.create_task(
                    self._health_check_loop(name)
                )

        self._app = web.Application()
        self._app.router.add_route("*", "/{path:.*}", self.proxy_handler)
        # Static file route for Vue assets
        static_path = _DASHBOARD_DIST / "assets"
        if static_path.exists():
            self._app.router.add_static(
                "/dashboard/assets", path=static_path, follow_symlinks=True
            )
        _logger.info("Proxy started on port %d", self.config.listen_port)

    async def stop(self) -> None:
        """Stop the proxy immediately (don't wait for in-flight)."""
        if self._dashboard_update_handle is not None:
            self._dashboard_update_handle.cancel()
            self._dashboard_update_handle = None
        if self._gpu_monitor:
            await self._gpu_monitor.stop()

        # Stop health check loops
        for task in self._health_check_tasks.values():
            task.cancel()
        if self._health_check_tasks:
            await asyncio.gather(
                *self._health_check_tasks.values(), return_exceptions=True
            )
        self._health_check_tasks.clear()

        for session in self._sessions.values():
            if not session.closed:
                await session.close()
        self._sessions.clear()
        self._app = None

    async def shutdown(self) -> None:
        """Shut down the proxy immediately (don't wait for in-flight).

        Reserved for future use (e.g., graceful drain via signal handler).
        """
        await self.stop()

    def _on_gpu_update(self, status: GPUStatus) -> None:
        """Callback from gpu_monitor after each poll.

        Updates GPU status and triggers state transitions for all services
        based on the new GPU data (e.g., hot → cold when VRAM drops to 0).
        """
        self.gpu_status = status
        self._evaluate_state_transitions()
        self._schedule_dashboard_update()
        if status.poll_failed:
            self.eviction_log.append(
                EvictionEventType.gpu_poll_failed, "GPU poll failed"
            )

    @staticmethod
    def _log_task_error(task: asyncio.Task) -> None:
        """Log errors from orphaned async tasks."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            _logger.error("Background task failed: %s", exc, exc_info=True)

    @staticmethod
    def _get_request_details(request: web.Request) -> tuple[Optional[str], str, str]:
        """Extract host, path, and method from an incoming request.

        Args:
            request: The incoming aiohttp request.

        Returns:
            A tuple of (clean_host, path, method).
        """
        host = request.host.split(":")[0] if request.host else None
        return host, request.path_qs, request.method

    async def proxy_handler(self, request: web.Request) -> web.StreamResponse:
        """Main entry point for incoming proxy requests.

        Demultiplexes by Host header and routes through the GPU gate or dashboard.
        Creates the response object once, then delegates to _handle_request.

        Args:
            request: The incoming aiohttp request.

        Returns:
            A web.StreamResponse (either the mutated response or an error response).
        """
        host, path, method = self._get_request_details(request)
        if not host:
            return web.Response(status=_HTTP_NOT_FOUND, reason="Unknown host")

        # Dashboard/static paths handled before response creation
        if host.split(":")[0] == self.config.dashboard_hostname:
            if path.startswith("/dashboard/assets/"):
                return await self._serve_static_asset(path)
            return await self._handle_dashboard_request(
                method, path, dict(request.query), request
            )

        body = await request.read() if request.can_read_body else b""
        is_ws = self.is_websocket_upgrade(dict(request.headers))

        if is_ws:
            response = MutatingWebSocketResponse(identity_mutator)
        else:
            response = MutatingStreamResponse(identity_mutator, status=_HTTP_OK)

        result = await self._handle_request(
            host,
            path,
            method,
            request,
            headers={
                k: v
                for k, v in request.headers.items()
                if k.lower() not in _REQUEST_FRAMING_HEADERS
            },
            body=body,
            response=response,
        )
        if isinstance(result, web.StreamResponse):
            return result
        return response

    async def _handle_request(
        self,
        host: str,
        path: str,
        method: str,
        web_request: web.Request,
        response: AbstractMutatingResponse,
        headers: Optional[dict] = None,
        body: Optional[bytes] = None,
        recursion_depth: int = 0,
        commit_on_error: bool = True,
    ) -> bool:
        """Route a request based on host, path, and method.

        Returns True on success, False on failure.
        If commit_on_error is True, the error response is committed.
        """
        if recursion_depth > _MAX_INTERNAL_RECURSION_DEPTH:
            _logger.error("Max recursion depth reached for host %s", host)
            if commit_on_error:
                await self._fail_immediately(
                    web_request,
                    response,
                    _HTTP_LOOP_DETECTED,
                    "Loop Detected: Max recursion depth reached",
                )
            return False

        # Strip port from host header (robustness for direct calls)
        host = host.split(":")[0]

        # Check service hostname
        service = self.config.hostname_map.get(host)
        if not service:
            if commit_on_error:
                await self._fail_immediately(
                    web_request, response, _HTTP_NOT_FOUND, "Unknown host"
                )
            return False

        # Run plugin pipeline on headers+body+path
        request_id = -1
        if self._plugin_manager:
            request_id = self._next_request_id
            self._next_request_id += 1
            try:
                headers, body, path = await self._plugin_manager.run_request_pipeline(
                    request_id, headers, body, path, service.name
                )
            except Exception as e:
                _logger.error("Plugin request pipeline failed: %s", e, exc_info=True)

            # Add plugin response mutator to response
            response.add_mutator(
                lambda resp_headers, chunk, is_final, status_code: self._run_response_plugins(
                    resp_headers,
                    chunk,
                    path,
                    status_code,
                    service.name,
                    is_final,
                    request_id,
                )
            )
            response.add_header_mutator(
                lambda resp_headers, status_code: self._plugin_manager.run_response_headers_pipeline(
                    request_id, resp_headers, status_code, service.name
                )
            )

        async def do_forward(
            fh: Optional[dict],
            fb: Optional[bytes],
            fp: str,
            fr: web.Request,
            resp: AbstractMutatingResponse,
        ) -> bool:
            try:
                return await self._handle_service_request(
                    service=service,
                    host=host,
                    path=fp,
                    method=method,
                    web_request=fr,
                    response=resp,
                    headers=fh,
                    body=fb,
                    recursion_depth=recursion_depth,
                    commit_on_error=commit_on_error,
                    request_id=request_id,
                )
            except:
                if self._plugin_manager:
                    self._plugin_manager.run_cancelled_pipeline(request_id, service.name)
                raise

        # Check path overrides
        if self._plugin_manager:
            path_overrides = self._plugin_manager.get_path_overrides(service.name)
            raw_path = urlparse(path).path
            if raw_path in path_overrides:
                plugin = path_overrides[raw_path]
                forward = make_forward_guard(do_forward)
                with forward_handler():
                    return await plugin.handle_overridden_request(
                        request_id,
                        headers,
                        body,
                        path,
                        web_request,
                        forward,
                        response,
                    )

        return await do_forward(headers, body, path, web_request, response)

    async def _handle_dashboard_request(
        self,
        method: str,
        path: str,
        query: Optional[dict[str, str]],
        web_request: web.Request,
    ) -> web.StreamResponse:
        """Handle requests directed to the dashboard hostname.

        Args:
            method: The HTTP method (e.g., "GET").
            path: The request path.
            query: The parsed query parameters.
            web_request: The incoming aiohttp request.

        Returns:
            A web.Response containing the dashboard HTML or JSON, or a 405 error.
        """
        if method != "GET":
            return web.Response(
                status=_HTTP_METHOD_NOT_ALLOWED, reason="Method Not Allowed"
            )

        # Health endpoint on dashboard hostname
        if path == "/health":
            return await self.health_handler_web()

        # WebSocket endpoint
        if path == "/ws":
            return await self._handle_dashboard_ws(web_request)

        fmt = (query or {}).get("format", "html")
        if fmt == "json":
            return await self._dashboard_json()
        return await self._serve_dashboard_html()

    async def _handle_dashboard_ws(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket endpoint for real-time dashboard updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # Send initial state
        await ws.send_json({"type": "initial", "data": self._get_dashboard_data()})

        # Track this client
        self._dashboard_clients.add(ws)

        try:
            async for msg in ws:
                if msg.type == WSMsgType.CLOSED:
                    break
        finally:
            self._dashboard_clients.discard(ws)

        return ws

    def schedule_dashboard_update(self) -> None:
        """Schedule a throttled dashboard broadcast.

        Public hook for the entry point to refresh dashboard clients after
        wiring up the plugin manager.
        """
        self._schedule_dashboard_update()

    def _schedule_dashboard_update(self) -> None:
        """Broadcast dashboard data to all WebSocket clients (throttled).

        Max 1 broadcast per second. Fires immediately if throttle allows,
        and always schedules a delayed broadcast at the 1s mark so the
        user gets updated within 1s even if events arrive mid-window.
        """
        now = self._clock.time()
        # Cancel any pending delayed broadcast
        if self._dashboard_update_handle is not None:
            self._dashboard_update_handle.cancel()
            self._dashboard_update_handle = None

        if now - self._last_dashboard_push >= _DASHBOARD_UPDATE_INTERVAL:
            # Throttle window open — broadcast immediately
            self._last_dashboard_push = now
            self._send_dashboard_push()
        # Always schedule a delayed broadcast at the 1s mark
        loop = asyncio.get_running_loop()
        delay = _DASHBOARD_UPDATE_INTERVAL - (now - self._last_dashboard_push)
        self._dashboard_update_handle = loop.call_later(
            delay, self._broadcast_dashboard
        )

    def _broadcast_dashboard(self) -> None:
        """Send delayed dashboard data to all connected WebSocket clients."""
        self._dashboard_update_handle = None
        now = self._clock.time()
        if now - self._last_dashboard_push < _DASHBOARD_UPDATE_INTERVAL:
            return
        self._last_dashboard_push = now
        self._send_dashboard_push()

    def _send_dashboard_push(self) -> None:
        """Build and send dashboard data to all connected WebSocket clients."""
        try:
            data = self._get_dashboard_data()
        except Exception:
            return
        for ws in list(self._dashboard_clients):
            if not ws.closed:
                asyncio.create_task(self._safe_send_dashboard(ws, data))

    async def _safe_send_dashboard(self, ws: web.WebSocketResponse, data: dict) -> None:
        """Send dashboard JSON to a WebSocket client, suppressing disconnect errors."""
        try:
            await ws.send_json({"type": "update", "data": data})
        except (ConnectionError, ConnectionResetError, RuntimeError):
            pass

    async def _serve_static_asset(self, path: str) -> web.Response:
        """Serve static assets from dashboard/dist/assets."""
        assets_path = _DASHBOARD_DIST / "assets"
        filename = path[len("/dashboard/assets/") :]
        file_path = assets_path / filename

        if not file_path.exists():
            return web.Response(status=_HTTP_NOT_FOUND, reason="Not found")

        content_type = "application/octet-stream"
        if filename.endswith(".js"):
            content_type = "application/javascript"
        elif filename.endswith(".css"):
            content_type = "text/css"
        elif filename.endswith(".png"):
            content_type = "image/png"
        elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
            content_type = "image/jpeg"
        elif filename.endswith(".gif"):
            content_type = "image/gif"
        elif filename.endswith(".svg"):
            content_type = "image/svg+xml"

        content = file_path.read_bytes()
        return web.Response(body=content, content_type=content_type, status=_HTTP_OK)

    async def _serve_dashboard_html(self) -> web.Response:
        """Serve the built Vue app index.html."""
        dist_path = _DASHBOARD_DIST / "index.html"

        if dist_path.exists():
            content = dist_path.read_text()
            return web.Response(text=content, content_type="text/html", status=_HTTP_OK)

        # Fallback to original HTML generation during development
        return await self._dashboard_html()

    async def _handle_service_request(
        self,
        service: ServiceConfig,
        host: str,
        path: str,
        method: str,
        web_request: web.Request,
        response: AbstractMutatingResponse,
        headers: Optional[dict],
        body: Optional[bytes],
        recursion_depth: int,
        commit_on_error: bool = True,
        request_id: int = -1,
    ) -> bool:
        self._schedule_dashboard_update()
        state = self.service_states.get(service.name)
        is_ws = self.is_websocket_upgrade(headers)
        is_gpu = _is_gpu_route(service, path) and not service.external

        if state and state.state == ServiceStateEnum.unhealthy:
            self.eviction_log.append(EvictionEventType.target_unreachable, service.name)
            if service.backup:
                return await self._handle_backup(
                    service,
                    host,
                    path,
                    method,
                    headers,
                    body,
                    web_request,
                    response,
                    is_ws,
                    recursion_depth,
                    commit_on_error=commit_on_error,
                )
            if commit_on_error:
                await self._fail_immediately(
                    web_request,
                    response,
                    _HTTP_SERVICE_UNAVAILABLE,
                    "Service Unavailable",
                )
            return False
        if is_gpu:
            return await self._handle_gpu_request(
                service,
                host,
                path,
                method,
                web_request,
                response,
                headers,
                body,
                is_ws,
                recursion_depth,
                commit_on_error=commit_on_error,
                request_id=request_id,
            )
        else:
            await self._prepare_target(service, request_id, path)
            return await self._handle_passthrough(
                service,
                host,
                path,
                method,
                headers,
                body,
                web_request,
                response,
                is_ws,
                recursion_depth,
                commit_on_error=commit_on_error,
            )

    async def _handle_gpu_request(
        self,
        service: ServiceConfig,
        host: str,
        path: str,
        method: str,
        web_request: web.Request,
        response: AbstractMutatingResponse,
        headers: Optional[dict],
        body: Optional[bytes],
        is_ws: bool,
        recursion_depth: int,
        commit_on_error: bool = True,
        request_id: int = -1,
    ) -> bool:
        state = self.service_states[service.name]

        entry = None
        try:
            # Try enqueue and check the decision made by the queue.
            entry = await self.wait_queue.enqueue(
                service.name, self.gpu_status, self.config, self.service_states
            )
            if not entry.future.done():
                # We should wait for a decision. Try eviction if necessary.
                asyncio.ensure_future(
                    self._trigger_eviction(service)
                ).add_done_callback(
                    lambda f: (
                        _logger.error("Eviction failed: %s", f.exception(), exc_info=True)
                        if f.exception()
                        else None
                    )
                )

            # Get the decision.
            decision, reason = await entry.future
        finally:
            if entry:
                await self.wait_queue.cancel_entry(entry)

        if decision == "target":

            def on_route_to_target() -> None:
                state.increment_in_flight()
                self.eviction_log.append(
                    EvictionEventType.request_approved, service.name, reason or ""
                )
                _logger.info("Routing %s to target: %s", service.name, reason)
                self._evaluate_state_transitions()

            def on_complete() -> None:
                state.decrement_in_flight()
                self._evaluate_state_transitions()

            await self._prepare_target(service, request_id, path)
            return await self._handle_passthrough(
                service,
                host,
                path,
                method,
                headers,
                body,
                web_request,
                response,
                is_ws,
                recursion_depth,
                on_target_attempt=on_route_to_target,
                on_target_result=lambda _success: on_complete(),
                commit_on_error=commit_on_error,
            )
        elif decision == "backup":
            if "unhealthy" in reason:
                self.eviction_log.append(
                    EvictionEventType.target_unreachable, service.name, reason
                )
            else:
                self.eviction_log.append(
                    EvictionEventType.request_rejected, service.name, reason
                )
            _logger.warning("Routing %s to backup: %s", service.name, reason)
            return await self._handle_backup(
                service,
                host,
                path,
                method,
                headers,
                body,
                web_request,
                response,
                is_ws,
                recursion_depth,
                commit_on_error=commit_on_error,
            )
        else:
            _logger.error("Unknown decision: %s", decision)
            if commit_on_error:
                await self._fail_immediately(
                    web_request,
                    response,
                    _HTTP_INTERNAL_SERVER_ERROR,
                    f"Unknown decision: {decision}",
                )
            return False

    def _trigger_queue_wake_up(self) -> None:
        """Trigger a wait queue re-evaluation in the background.

        Constructs the current service budget map and dispatches a background
        task to process the wait queue based on current GPU status.
        """
        asyncio.create_task(
            self.wait_queue.process_wake_up(
                self.gpu_status, self.config, self.service_states
            )
        ).add_done_callback(self._log_task_error)

    async def _prepare_target(
        self, service: ServiceConfig, request_id: int, path: str
    ) -> None:
        """Run the plugins' prepare_target pipeline before dispatching to the target.

        The request is dispatched to the target regardless of the outcome: a hook
        that fails is logged, and the passthrough's target-to-backup failover
        then covers a target that is still unusable.
        """
        if not self._plugin_manager:
            return
        try:
            await self._plugin_manager.run_target_pipeline(
                request_id, path, service.name
            )
        except Exception as e:
            _logger.error("Plugin prepare_target pipeline failed: %s", e, exc_info=True)

    async def _run_response_plugins(
        self,
        headers: dict[str, str],
        chunk: bytes,
        path: str,
        status_code: int,
        service_name: str,
        is_final: bool,
        request_id: int,
    ) -> bytes:
        """Run response chunk pipeline for plugins attached to a service.

        Returns the (possibly modified) chunk from the plugin pipeline.
        """
        if not self._plugin_manager or not service_name:
            return chunk
        svc_conf = self.config.services.get(service_name)
        if not svc_conf or not svc_conf.plugins:
            return chunk
        try:
            return await self._plugin_manager.run_response_chunk_pipeline(
                request_id, headers, chunk, path, status_code, service_name, is_final
            )
        except Exception as e:
            _logger.error("Plugin response pipeline failed: %s", e, exc_info=True)
            return chunk

    async def _fail_immediately(
        self,
        web_request: web.Request,
        response: AbstractMutatingResponse,
        status: int,
        reason: str,
    ) -> None:
        """Prepare and commit an immediate error response if not already prepared.

        If the response has not been prepared yet, sets the given status and reason,
        prepares the response to the client, and writes EOF. Silently skips if the
        response was already prepared.
        """
        if response.prepared:
            return  # already prepared, no-op
        response.set_status(status, reason=reason)
        try:
            await response.prepare(web_request)
            await response.write_eof()
        except Exception:
            pass  # e.g., WebSocket handshake fails
        _logger.error(
            f'Request failed immediately with {status} error. Reason: "{reason}".'
        )
    
    async def _monitor_disconnection(
        self, request: web.Request, service_name: str, on_disconnected: Callable
    ) -> None:
        """Watch for a client disconnect and notify the caller while it persists."""
        while True:
            if request.transport.is_closing():
                _logger.error(f"Client disconnected from service {service_name} before the request finishes!")
                on_disconnected()
            await asyncio.sleep(0.5)

    async def _proxy_http_request(
        self,
        url: str,
        host: str,
        path: str,
        method: str,
        headers: Optional[dict],
        body: Optional[bytes],
        recursion_depth: int,
        web_request: web.Request,
        response: AbstractMutatingResponse,
        service_name: str = "",
        commit_on_error: bool = True,
    ) -> bool:
        """Proxy a request to the given URL using the passed response object."""
        loopback_result = await self._handle_loopback(
            url,
            host,
            path,
            method,
            web_request,
            response,
            headers,
            body,
            recursion_depth,
            commit_on_error=commit_on_error,
        )
        if loopback_result is not None:
            return loopback_result

        # Construct full URL with path and query string
        url = url.rstrip("/") + path

        actual_body = body if body else None
        session = self._get_session(service_name)
        client_prepared = False
        try:
            async with session.request(
                method,
                url,
                headers=headers,
                data=actual_body,
            ) as resp:
                response.set_status(resp.status, resp.reason)
                # Copy response headers
                for key, val in resp.headers.items():
                    # Skip hop-by-hop headers
                    lower = key.lower()
                    if lower in _HOP_BY_HOP_HEADERS:
                        continue
                    response.headers[key] = val

                if resp.status != _HTTP_OK:
                    payload = await resp.read()
                    resp.close()
                    if commit_on_error:
                        await response.prepare(web_request)
                        await response.write(payload)
                        await response.write_eof()
                    _logger.error(f"Error: upstream server {url} responded status code {resp.status} and reason \"{resp.reason}\". Response body: {payload}.")
                    return False

                monitor_task = None
                try:
                    await response.prepare(web_request)
                    client_prepared = True
                    monitor_task = asyncio.create_task(
                        self._monitor_disconnection(web_request, service_name, resp.close)
                    )
                    # iter_any() yields data as soon as it's available from upstream,
                    # which is essential for SSE streaming.
                    async for chunk in resp.content.iter_any():
                        await response.write(chunk)
                finally:
                    if monitor_task:
                        monitor_task.cancel()
                    await response.write_eof()
        except Exception as e:
            _logger.error("Error during http forwarding. URL: %s. Error: %s.", url, e)
            if commit_on_error:
                await self._fail_immediately(
                    web_request,
                    response,
                    _HTTP_SERVICE_UNAVAILABLE,
                    str(e),
                )
        return client_prepared

    async def _relay_ws(
        self,
        host: str,
        path: str,
        method: str,
        url: str,
        headers: Optional[dict],
        request: web.Request,
        response: MutatingWebSocketResponse,
        recursion_depth: int,
        service_name: str = "",
        commit_on_error: bool = True,
    ) -> bool:
        """Proxy a WebSocket request to the given URL."""
        loopback_result = await self._handle_loopback(
            url,
            host,
            path,
            method,
            request,
            response,
            headers,
            None,
            recursion_depth,
            commit_on_error=commit_on_error,
        )
        if loopback_result is not None:
            return loopback_result

        session = self._get_session(service_name)
        client_prepared = False
        monitor_task = None
        url = url.rstrip("/") + path
        try:
            async with session.ws_connect(
                url, headers=headers
            ) as ws_server:
                try:
                    await response.prepare(request)
                    client_prepared = True
                    monitor_task = asyncio.create_task(
                        self._monitor_disconnection(request, service_name, ws_server.close)
                    )
                    await self._bidirectional_websocket_relay(response, ws_server)
                finally:
                    if monitor_task:
                        monitor_task.cancel()
                    await response.close()
        except WSServerHandshakeError as e:
            if commit_on_error:
                status_code = e.status
                reason = e.message
                await self._fail_immediately(request, response, status_code, reason)
            return False
        except Exception as e:
            _logger.error("Error during ws relay. URL: %s. Error: %s.", url, e)
            if commit_on_error:
                await self._fail_immediately(
                    request, response, _HTTP_SERVICE_UNAVAILABLE, str(e)
                )
        return client_prepared

    async def _attempt_forward(
        self,
        url: Optional[str],
        on_attempt: Optional[Callable[[], None]],
        on_result: Optional[Callable[[bool, Optional[str]], None]],
        forward: Callable[[str, bool], Awaitable[bool]],
        commit_on_error: bool,
        failure_action: str,
        web_request: web.Request,
    ) -> bool:
        """Run one forwarding attempt at url and report the outcome to the callbacks.

        Args:
            url: The URL to forward to, or None to report failure without attempting.
            on_attempt: Called before the attempt starts.
            on_result: Called with (result, err_msg) once the attempt settles.
            forward: Performs the attempt, taking the URL and commit_on_error.
            commit_on_error: Whether forward may commit an error response.
            failure_action: Names the attempt in the failure message.
            web_request: The incoming request, used for the failure message path.

        Returns:
            True if the attempt succeeded. False if url is None, if forward
            reported failure, or if forward raised an Exception.

        Raises:
            asyncio.CancelledError: If the attempt is cancelled. Cancellation
                derives from BaseException rather than Exception, so it is not
                caught here and propagates out once on_result has run.
        """
        if url is None:
            return False
        if on_attempt:
            on_attempt()
        result = False
        err_msg = None
        try:
            result = await forward(url, commit_on_error)
            if result:
                return True
            else:
                err_msg = (
                    f"Failed to {failure_action} to endpoint {url + web_request.path}."
                )
                _logger.error(err_msg)
                return False
        except Exception as e:
            err_msg = str(e)
            _logger.error(err_msg)
            return False
        finally:
            if on_result:
                on_result(result, err_msg)

    async def _forward_ws(
        self,
        host: str,
        path: str,
        method: str,
        headers: Optional[dict],
        web_request: web.Request,
        response: MutatingWebSocketResponse,
        recursion_depth: int,
        target_url: Optional[str],
        backup_url: Optional[str] = None,
        on_target_attempt: Optional[Callable[[], None]] = None,
        on_target_result: Optional[Callable[[bool, Optional[str]], None]] = None,
        on_backup_attempt: Optional[Callable[[], None]] = None,
        on_backup_result: Optional[Callable[[bool, Optional[str]], None]] = None,
        service_name: str = "",
        commit_on_error: bool = True,
    ) -> bool:
        """Forward a WebSocket request to target_url, with optional fallback."""
        if target_url is None:
            if commit_on_error:
                await self._fail_immediately(
                    web_request,
                    response,
                    _HTTP_SERVICE_UNAVAILABLE,
                    "Service Unavailable",
                )
            return False

        async def relay(url: str, commit: bool) -> bool:
            """Relay the WebSocket to one URL."""
            return await self._relay_ws(
                host,
                path,
                method,
                url,
                headers,
                web_request,
                response,
                recursion_depth,
                service_name=service_name,
                commit_on_error=commit,
            )

        result = await self._attempt_forward(
            target_url,
            on_target_attempt,
            on_target_result,
            relay,
            commit_on_error and backup_url is None,
            "relay websocket",
            web_request,
        )
        if result:
            return result
        return await self._attempt_forward(
            backup_url,
            on_backup_attempt,
            on_backup_result,
            relay,
            commit_on_error,
            "relay websocket",
            web_request,
        )

    async def _forward_http(
        self,
        host: str,
        path: str,
        method: str,
        headers: Optional[dict],
        body: Optional[bytes],
        web_request: web.Request,
        response: AbstractMutatingResponse,
        recursion_depth: int,
        target_url: Optional[str],
        backup_url: Optional[str] = None,
        on_target_attempt: Optional[Callable[[], None]] = None,
        on_target_result: Optional[Callable[[bool, Optional[str]], None]] = None,
        on_backup_attempt: Optional[Callable[[], None]] = None,
        on_backup_result: Optional[Callable[[bool, Optional[str]], None]] = None,
        service_name: str = "",
        commit_on_error: bool = True,
    ) -> bool:
        """Forward an HTTP request to target_url, with optional fallback."""
        if target_url is None:
            if commit_on_error:
                await self._fail_immediately(
                    web_request,
                    response,
                    _HTTP_SERVICE_UNAVAILABLE,
                    "Service Unavailable",
                )
            return False

        async def send(url: str, commit: bool) -> bool:
            """Send the HTTP request to one URL."""
            return await self._proxy_http_request(
                url,
                host,
                path,
                method,
                headers,
                body,
                recursion_depth,
                web_request,
                response,
                service_name=service_name,
                commit_on_error=commit,
            )

        result = await self._attempt_forward(
            target_url,
            on_target_attempt,
            on_target_result,
            send,
            commit_on_error and backup_url is None,
            "forward http request",
            web_request,
        )
        if result:
            return result
        return await self._attempt_forward(
            backup_url,
            on_backup_attempt,
            on_backup_result,
            send,
            commit_on_error,
            "forward http request",
            web_request,
        )

    async def _handle_passthrough(
        self,
        service: ServiceConfig,
        host: str,
        path: str,
        method: str,
        headers: Optional[dict],
        body: Optional[bytes],
        web_request: web.Request,
        response: AbstractMutatingResponse,
        is_ws: bool,
        recursion_depth: int,
        commit_on_error: bool = True,
        on_target_attempt: Optional[Callable[[], None]] = None,
        on_target_result: Optional[Callable[[bool], None]] = None,
        on_backup_attempt: Optional[Callable[[], None]] = None,
        on_backup_result: Optional[Callable[[bool], None]] = None,
    ) -> bool:
        """Forward request to the target, and fallback to backup if target fails."""

        def target_attempt() -> None:
            if on_target_attempt:
                on_target_attempt()
            self.routing_stats[service.name]["target"] += 1

        def target_result(success: bool, err_msg: Optional[str]) -> None:
            if on_target_result:
                on_target_result(success)
            if not success:
                self.eviction_log.append(
                    (
                        EvictionEventType.target_unreachable
                        if err_msg and "unreachable" in err_msg.lower()
                        else EvictionEventType.target_failed
                    ),
                    service.name,
                    err_msg or "",
                )
                _logger.warning(
                    "Target request failed for %s. Error: %s", service.name, err_msg
                )

        def backup_attempt() -> None:
            if on_backup_attempt:
                on_backup_attempt()
            self.routing_stats[service.name]["backup"] += 1

        def backup_result(success: bool, err_msg: Optional[str]) -> None:
            if on_backup_result:
                on_backup_result(success)
            if not success:
                self.eviction_log.append(
                    (
                        EvictionEventType.backup_unreachable
                        if err_msg and "unreachable" in err_msg.lower()
                        else EvictionEventType.backup_failed
                    ),
                    service.name,
                    err_msg or "",
                )
                _logger.warning(
                    "Backup request failed for %s. Error: %s", service.name, err_msg
                )

        return await self._forward_request(
            host,
            path,
            method,
            headers,
            body,
            web_request,
            response,
            is_ws,
            recursion_depth,
            target_url=service.target,
            backup_url=service.backup if service.fallback_on_error else None,
            on_target_attempt=target_attempt,
            on_target_result=target_result,
            on_backup_attempt=backup_attempt,
            on_backup_result=backup_result,
            service_name=service.name,
            commit_on_error=commit_on_error,
        )

    async def _handle_backup(
        self,
        service: ServiceConfig,
        host: str,
        path: str,
        method: str,
        headers: Optional[dict],
        body: Optional[bytes],
        web_request: web.Request,
        response: AbstractMutatingResponse,
        is_ws: bool,
        recursion_depth: int,
        commit_on_error: bool = True,
        on_attempt: Optional[Callable[[], None]] = None,
        on_result: Optional[Callable[[bool], None]] = None,
    ) -> bool:
        """Forward request to backup service directly."""

        def backup_attempt() -> None:
            if on_attempt:
                on_attempt()
            self.routing_stats[service.name]["backup"] += 1

        def backup_result(success: bool, err_msg: Optional[str]) -> None:
            if on_result:
                on_result(success)
            if not success:
                self.eviction_log.append(
                    (
                        EvictionEventType.backup_unreachable
                        if err_msg and "unreachable" in err_msg.lower()
                        else EvictionEventType.backup_failed
                    ),
                    service.name,
                    err_msg or "",
                )
                _logger.warning(
                    "Backup request failed for %s. Error: %s", service.name, err_msg
                )

        return await self._forward_request(
            host,
            path,
            method,
            headers,
            body,
            web_request,
            response,
            is_ws,
            recursion_depth,
            target_url=service.backup,
            on_target_attempt=backup_attempt,
            on_target_result=backup_result,
            service_name=service.name,
            commit_on_error=commit_on_error,
        )

    async def _forward_request(
        self,
        host: str,
        path: str,
        method: str,
        headers: Optional[dict],
        body: Optional[bytes],
        web_request: web.Request,
        response: AbstractMutatingResponse,
        is_ws: bool,
        recursion_depth: int,
        target_url: Optional[str],
        backup_url: Optional[str] = None,
        on_target_attempt: Optional[Callable[[], None]] = None,
        on_target_result: Optional[Callable[[bool, Optional[str]], None]] = None,
        on_backup_attempt: Optional[Callable[[], None]] = None,
        on_backup_result: Optional[Callable[[bool, Optional[str]], None]] = None,
        service_name: str = "",
        commit_on_error: bool = True,
    ) -> bool:
        """Forward request to target_url with optional fallback to backup_url."""
        if is_ws:
            return await self._forward_ws(
                host,
                path,
                method,
                headers,
                web_request,
                response,
                recursion_depth,
                target_url,
                backup_url,
                on_target_attempt,
                on_target_result,
                on_backup_attempt,
                on_backup_result,
                service_name=service_name,
                commit_on_error=commit_on_error,
            )
        else:
            return await self._forward_http(
                host,
                path,
                method,
                headers,
                body,
                web_request,
                response,
                recursion_depth,
                target_url,
                backup_url,
                on_target_attempt,
                on_target_result,
                on_backup_attempt,
                on_backup_result,
                service_name=service_name,
                commit_on_error=commit_on_error,
            )

    async def _bidirectional_websocket_relay(
        self,
        ws_client: web.WebSocketResponse,
        ws_server: ClientWebSocketResponse,
    ) -> None:
        """Relay messages bidirectionally between client and backend WebSockets."""

        async def client_to_backend() -> None:
            async for msg in ws_client:
                if msg.type == WSMsgType.TEXT:
                    await ws_server.send_str(msg.data)
                elif msg.type == WSMsgType.BINARY:
                    await ws_server.send_bytes(msg.data)
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    await ws_server.close()
                    break

        async def backend_to_client() -> None:
            async for msg in ws_server:
                if msg.type == WSMsgType.TEXT:
                    await ws_client.send_str(msg.data)
                elif msg.type == WSMsgType.BINARY:
                    await ws_client.send_bytes(msg.data)
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    await ws_client.close()
                    break

        await asyncio.gather(
            client_to_backend(),
            backend_to_client(),
            return_exceptions=True,
        )

    async def _evict_single_service(
        self,
        svc: ServiceConfig,
        st: ServiceState,
        requesting_service: ServiceConfig,
    ) -> None:
        """Evict a single service: wait for in-flight drain, run eviction command."""
        if st.state in (ServiceStateEnum.evicting, ServiceStateEnum.restarting):
            # Already being evicted. Join the wait.
            await st.eviction_complete.wait()
            return

        # Transition to evicting
        st.transition(ServiceStateEnum.evicting)
        self.eviction_log.append(
            EvictionEventType.eviction_started,
            svc.name,
            f"evicted by {requesting_service.name} (priority {requesting_service.priority})",
        )
        _logger.info("Evicting %s for %s", svc.name, requesting_service.name)

        # Wait for in_flight_zero or timeout
        try:
            await asyncio.wait_for(
                st.in_flight_zero.wait(),
                timeout=self.config.eviction_timeout,
            )
            self.eviction_log.append(
                EvictionEventType.eviction_completed, svc.name, "in_flight drained"
            )
        except asyncio.TimeoutError:
            self.eviction_log.append(
                EvictionEventType.eviction_timeout,
                svc.name,
                f"eviction_timeout ({self.config.eviction_timeout}s) expired",
            )

        # Run eviction command
        cmd = svc.eviction_command.replace("{container}", svc.container)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd.split(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(
                proc.communicate(), timeout=self.config.eviction_command_timeout
            )
            if proc.returncode == 0:
                self.eviction_log.append(
                    EvictionEventType.eviction_command_completed, svc.name, cmd
                )
            else:
                self.eviction_log.append(
                    EvictionEventType.eviction_command_failed,
                    svc.name,
                    f"{cmd} (exit_code={proc.returncode})",
                )
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            self.eviction_log.append(
                EvictionEventType.eviction_command_failed,
                svc.name,
                f"{cmd} (timeout after {self.config.eviction_command_timeout}s)",
            )
        except Exception as e:
            self.eviction_log.append(
                EvictionEventType.eviction_command_failed, svc.name, f"{cmd} ({e})"
            )

        # Run health check after eviction.
        await self._run_health_check(svc.name)
        # Poll new GPU status after eviction.
        if self._gpu_monitor:
            await self._gpu_monitor.poll_once()

        # Transition to restarting
        st.transition(ServiceStateEnum.restarting)
        self.routing_stats[svc.name]["evictions"] += 1
        # Trigger an immediate state evaluation.
        self._evaluate_state_transitions()
        # Schedule a state transition after timeout.
        self._schedule_state_evaluation(delay_seconds=svc.health_check_interval)
        # Push dashboard change
        self._schedule_dashboard_update()

    async def _trigger_eviction(self, requesting_service: ServiceConfig) -> None:
        """Trigger eviction of lower-priority services to free VRAM.

        Evictions run concurrently via asyncio.gather.
        After completion, triggers immediate re-poll and processes wait queue.
        """
        evictable, _, _ = get_eviction_candidates(
            requesting_service,
            self.gpu_status,
            self.config,
            self.service_states,
            self.wait_queue,
        )

        # Run evictions concurrently
        if evictable:
            await asyncio.gather(
                *[
                    self._evict_single_service(svc, st, requesting_service)
                    for svc, st in evictable
                ],
                return_exceptions=True,
            )

            # Wake up the queue for processing
            self._trigger_queue_wake_up()

    async def health_handler_web(self) -> web.Response:
        """Generate health response."""
        uptime = self.get_uptime_seconds()
        status = self.gpu_status

        if status.poll_failed and status.last_updated is None:
            return web.Response(
                status=_HTTP_SERVICE_UNAVAILABLE,
                content_type="application/json",
                text=json.dumps(
                    {
                        "status": "unhealthy",
                        "uptime_s": uptime,
                        "details": "no_gpu_data",
                    }
                ),
            )
        elif status.poll_failed:
            return web.Response(
                status=_HTTP_OK,
                content_type="application/json",
                text=json.dumps(
                    {
                        "status": "warning",
                        "uptime_s": uptime,
                        "details": "gpu_poll_failed",
                    }
                ),
            )
        else:
            return web.Response(
                status=_HTTP_OK,
                content_type="application/json",
                text=json.dumps({"status": "ok", "uptime_s": uptime}),
            )

    def _get_dashboard_data(self) -> dict:
        """Generate dashboard data dictionary (shared by JSON and WebSocket)."""
        remaining_budget = compute_effective_remaining(
            self.gpu_status, self.config, self.service_states, self.wait_queue
        )
        effective_budget = min(
            self.gpu_status.available_vram_mib, self.config.total_vram_budget
        )

        services_data = {}
        for name, svc in self.config.services.items():
            st = self.service_states[name]
            rs = self.routing_stats[name]
            services_data[name] = {
                "state": st.state.value,
                "priority": svc.priority,
                "vram_budget_mib": svc.vram_budget,
                "vram_used_mib": self.gpu_status.containers.get(
                    svc.container, ContainerVRAM()
                ).vram_mib,
                "in_flight": st.in_flight,
                "is_healthy": st.is_healthy,
                "external": svc.external,
                "routing": {
                    "target": rs["target"],
                    "backup": rs["backup"],
                },
            }

        containers_data = {}
        for name, cv in self.gpu_status.containers.items():
            containers_data[name] = {
                "vram_mib": cv.vram_mib,
                "gpu_util_pct": cv.gpu_util_pct,
            }

        health = "ok"
        if self.gpu_status.poll_failed and self.gpu_status.last_updated is None:
            health = "unhealthy"
        elif self.gpu_status.poll_failed:
            health = "warning"

        result = {
            "proxy": {
                "uptime_s": self.get_uptime_seconds(),
                "last_poll": (
                    datetime.fromtimestamp(
                        self.gpu_status.last_updated, tz=timezone.utc
                    ).isoformat()
                    if self.gpu_status.last_updated
                    else None
                ),
                "health": health,
            },
            "gpu": {
                "effective_budget_mib": effective_budget,
                "remaining_budget_mib": remaining_budget,
                "available_vram_mib": self.gpu_status.available_vram_mib,
                "total_vram_mib": self.gpu_status.total_vram_mib,
                "system_vram_mib": self.gpu_status.system_vram_mib,
                "docker_vram_mib": self.gpu_status.docker_vram_mib,
                "gpu_util_pct": self.gpu_status.gpu_util_pct,
                "power_usage_w": self.gpu_status.power_usage_w,
                "power_limit_w": self.gpu_status.power_limit_w,
                "temperature_c": self.gpu_status.temperature_c,
                "gr_clock_mhz": self.gpu_status.gr_clock_mhz,
                "mem_clock_mhz": self.gpu_status.mem_clock_mhz,
                "pstate": self.gpu_status.pstate,
                "containers": containers_data,
            },
            "services": services_data,
            "eviction_log": self.eviction_log.rows(),
            "quiet_eviction_log": self.eviction_log.quiet_rows(),
            "routing_stats": {name: dict(rs) for name, rs in self.routing_stats.items()},
        }

        if self._plugin_manager:
            plugin_uis = {}
            for service_name in self.config.services:
                uis = self._plugin_manager.get_plugin_uis(service_name)
                if uis:
                    plugin_uis[service_name] = [
                        {"title": title, "html": html} for title, html in uis
                    ]
            if plugin_uis:
                result["plugin_uis"] = plugin_uis

        return result

    async def _dashboard_json(self) -> web.Response:
        """Generate JSON dashboard response."""
        data = self._get_dashboard_data()
        return web.Response(
            status=_HTTP_OK,
            content_type="application/json",
            text=json.dumps(data),
        )

    def _get_gpu_metric_threshold(self, value: float, metric_type: str) -> str:
        """Determine threshold level for GPU metrics.

        Args:
            value: The metric value to evaluate
            metric_type: One of 'gpu_util', 'power', 'temperature', 'vram_usage'

        Returns:
            'low', 'medium', or 'high' based on threshold boundaries
        """
        if metric_type == "gpu_util":
            if value < 60:
                return "low"
            elif value <= 80:
                return "medium"
            else:
                return "high"
        elif metric_type == "power":
            if value < 180:
                return "low"
            elif value <= 280:
                return "medium"
            else:
                return "high"
        elif metric_type == "temperature":
            if value < 60:
                return "low"
            elif value <= 75:
                return "medium"
            else:
                return "high"
        elif metric_type == "vram_usage":
            if value < 50:
                return "low"
            elif value <= 80:
                return "medium"
            else:
                return "high"
        return "low"

    async def _dashboard_html(self) -> web.Response:
        """Generate HTML dashboard response."""
        effective_budget = compute_effective_remaining(
            self.gpu_status, self.config, self.service_states, self.wait_queue
        )

        vram_usage_pct = 0
        if self.gpu_status.total_vram_mib > 0:
            vram_usage_pct = int(
                100
                * (self.gpu_status.total_vram_mib - self.gpu_status.available_vram_mib)
                / self.gpu_status.total_vram_mib
            )

        power_pct = 0
        if self.gpu_status.power_limit_w > 0:
            power_pct = int(
                100 * self.gpu_status.power_usage_w / self.gpu_status.power_limit_w
            )

        gpu_util_threshold = self._get_gpu_metric_threshold(
            self.gpu_status.gpu_util_pct, "gpu_util"
        )
        # Use gpu_util thresholds (percentage-based: <60%, 60-80%, >80%) for power percentage
        power_threshold = self._get_gpu_metric_threshold(power_pct, "gpu_util")
        temp_threshold = self._get_gpu_metric_threshold(
            self.gpu_status.temperature_c, "temperature"
        )
        vram_threshold = self._get_gpu_metric_threshold(vram_usage_pct, "vram_usage")

        rows = ""
        mobile_service_rows = ""
        for name, svc in self.config.services.items():
            st = self.service_states[name]
            rs = self.routing_stats[name]
            used_vram = self.gpu_status.containers.get(
                svc.container, ContainerVRAM()
            ).vram_mib

            health_class = "health-healthy" if st.is_healthy else "health-unhealthy"
            health_label = "Healthy" if st.is_healthy else "Unhealthy"
            external_label = " (External)" if svc.external else ""

            rows += f"""<tr>
                <td>{name}{external_label}</td>
                <td>{svc.priority}</td>
                <td class="{health_class}">{health_label}</td>
                <td class="state-{st.state.value}">{st.state.value}</td>
                <td>{used_vram} / {svc.vram_budget} MiB</td>
                <td>{st.in_flight}</td>
                <td>{rs["target"]}</td>
                <td>{rs["backup"]}</td>
            </tr>"""
            mobile_service_rows += f"""<div class="mobile-card">
                <div class="mobile-card-header">{name}{external_label}</div>
                <div class="mobile-card-row"><span class="mobile-card-label">Priority</span><span>{svc.priority}</span></div>
                <div class="mobile-card-row"><span class="mobile-card-label">Health</span><span class="{health_class}">{health_label}</span></div>
                <div class="mobile-card-row"><span class="mobile-card-label">State</span><span class="state-{st.state.value}">{st.state.value}</span></div>
                <div class="mobile-card-row"><span class="mobile-card-label">VRAM</span><span>{used_vram} / {svc.vram_budget} MiB</span></div>
                <div class="mobile-card-row"><span class="mobile-card-label">In-Flight</span><span>{st.in_flight}</span></div>
                <div class="mobile-card-row"><span class="mobile-card-label">Target/Backup</span><span>{rs["target"]}/{rs["backup"]}</span></div>
            </div>"""

        container_rows = ""
        for name, cv in self.gpu_status.containers.items():
            container_rows += f"""<tr>
                <td>{name}</td>
                <td>{cv.vram_mib}</td>
                <td>{cv.gpu_util_pct}%</td>
            </tr>"""

        log_rows = ""
        for row in self.eviction_log.rows()[-_DASHBOARD_LOG_LIMIT:]:
            log_rows += f"""<tr>
                <td>{row["timestamp"] or ''}</td>
                <td>{row["event"]}</td>
                <td>{row["service"] or ''}</td>
                <td>{row["details"]}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="5">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU Priority Proxy Dashboard</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ background: #f0f2f5; color: #1a1a1a; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 16px; }}
        h1 {{ color: #000000; font-size: 1.5rem; margin: 0 0 16px 0; font-weight: 700; }}
        h2 {{ color: #2c2c2c; font-size: 1.125rem; margin: 0 0 12px 0; border-bottom: 2px solid #c0c0c0; padding-bottom: 8px; font-weight: 600; }}
        .card {{ background: #ffffff; border-radius: 8px; padding: 16px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.12); border: 1px solid #d0d0d0; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 16px; }}
        .stat-item {{ background: #f7f7f7; padding: 12px; border-radius: 6px; text-align: center; border: 1px solid #e0e0e0; }}
        .stat-value {{ font-size: 1.5rem; font-weight: 700; color: #004494; }}
        .stat-label {{ font-size: 0.75rem; color: #404040; margin-top: 4px; font-weight: 500; }}
        .table-wrapper {{ overflow-x: auto; -webkit-overflow-scrolling: touch; border-radius: 6px; }}
        table {{ border-collapse: collapse; width: 100%; font-size: 0.875rem; }}
        th {{ background: #f0f0f0; color: #202020; border: 1px solid #c0c0c0; padding: 10px 8px; text-align: left; white-space: nowrap; position: sticky; top: 0; font-weight: 700; }}
        td {{ color: #1a1a1a; border: 1px solid #c0c0c0; padding: 10px 8px; text-align: left; white-space: nowrap; }}
        .state-cold {{ color: #404040; }}
        .state-hot {{ color: #8b0000; }}
        .state-warm {{ color: #b8860b; }}
        .state-evicting {{ color: #b8860b; }}
        .state-restarting {{ color: #00008b; }}
        .health-healthy {{ color: #107c10; font-weight: bold; }}
        .health-unhealthy {{ color: #8b0000; font-weight: bold; }}
        .external-label {{ font-size: 0.75rem; color: #666; margin-left: 4px; font-style: italic; }}
        .gpu-metric-low {{ color: #107c10; }}
        .gpu-metric-medium {{ color: #b8860b; }}
        .gpu-metric-high {{ color: #8b0000; }}
        .mobile-card {{ display: none; border-radius: 8px; padding: 12px; margin-bottom: 12px; background: #ffffff; border: 1px solid #d0d0d0; }}
        .mobile-card-header {{ font-weight: bold; color: #004494; margin-bottom: 8px; }}
        .mobile-card-row {{ display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #c0c0c0; }}
        .mobile-card-row:last-child {{ border-bottom: none; }}
        .mobile-card-label {{ color: #404040; font-size: 0.875rem; }}
        @media (max-width: 768px) {{
            body {{ padding: 12px; }}
            h1 {{ font-size: 1.25rem; }}
            h2 {{ font-size: 1rem; }}
            .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .card {{ padding: 12px; }}
            .desktop-table {{ display: none; }}
            .mobile-cards {{ display: block; }}
            .mobile-card {{ display: block; }}
        }}
    </style>
</head>
<body>
    <h1>GPU Priority Proxy</h1>

    <div class="card">
        <h2>Proxy Status</h2>
        <div class="stats-grid">
            <div class="stat-item"><div class="stat-value">{self.get_uptime_seconds()}</div><div class="stat-label">Uptime (s)</div></div>
            <div class="stat-item"><div class="stat-value">{effective_budget}</div><div class="stat-label">Effective Budget (MiB)</div></div>
        </div>
        <p style="font-size: 0.875rem; color: #999;">Last GPU Poll: {datetime.fromtimestamp(self.gpu_status.last_updated, tz=timezone.utc).isoformat() if self.gpu_status.last_updated else 'N/A'}</p>
    </div>

    <div class="card">
        <h2>GPU Overview</h2>
        <div class="stats-grid">
            <div class="stat-item"><div class="stat-value gpu-metric-{gpu_util_threshold}">{self.gpu_status.gpu_util_pct}%</div><div class="stat-label">GPU Util</div></div>
            <div class="stat-item"><div class="stat-value gpu-metric-{power_threshold}">{self.gpu_status.power_usage_w}W</div><div class="stat-label">Power</div></div>
            <div class="stat-item"><div class="stat-value">{self.gpu_status.power_limit_w}W</div><div class="stat-label">Power Limit</div></div>
            <div class="stat-item"><div class="stat-value gpu-metric-{temp_threshold}">{self.gpu_status.temperature_c}°C</div><div class="stat-label">Temp</div></div>
            <div class="stat-item"><div class="stat-value">{self.gpu_status.total_vram_mib}</div><div class="stat-label">Total VRAM (MiB)</div></div>
            <div class="stat-item"><div class="stat-value gpu-metric-{vram_threshold}">{self.gpu_status.available_vram_mib}</div><div class="stat-label">Available VRAM (MiB)</div></div>
            <div class="stat-item"><div class="stat-value">{self.gpu_status.docker_vram_mib}</div><div class="stat-label">Docker VRAM (MiB)</div></div>
            <div class="stat-item"><div class="stat-value">{self.gpu_status.system_vram_mib}</div><div class="stat-label">System VRAM (MiB)</div></div>
        </div>
        <h2>Containers</h2>
        <div class="table-wrapper">
            <table>
                <tr><th>Container</th><th>VRAM (MiB)</th><th>GPU Util</th></tr>
                {container_rows}
            </table>
        </div>
    </div>

    <div class="card">
        <h2>Services</h2>
        <div class="table-wrapper desktop-table">
            <table>
                <tr><th>Name</th><th>Priority</th><th>Health</th><th>State</th><th>VRAM (Used / Budget)</th><th>In-Flight</th><th>Target</th><th>Backup</th></tr>
                {rows}
            </table>
        </div>
        <div class="mobile-cards">
            {mobile_service_rows}
        </div>
    </div>

    <div class="card">
        <h2>Eviction Event Log</h2>
        <div class="table-wrapper">
            <table>
                <tr><th>Timestamp</th><th>Event</th><th>Service</th><th>Details</th></tr>
                {log_rows}
            </table>
        </div>
    </div>

    <div class="card">
        <h2>Routing Stats</h2>
        <div class="table-wrapper">
            <table>
                <tr><th>Service</th><th>Target</th><th>Backup</th><th>Evictions</th></tr>
                {"".join(f'<tr><td>{n}</td><td>{r["target"]}</td><td>{r["backup"]}</td><td>{r["evictions"]}</td></tr>' for n, r in self.routing_stats.items())}
            </table>
        </div>
    </div>
</body>
</html>"""

        return web.Response(text=html, content_type="text/html", status=_HTTP_OK)

    def get_uptime_seconds(self) -> int:
        """Return proxy uptime in seconds."""
        if self._start_time is None:
            return 0
        return int(self._clock.time() - self._start_time)

    def _schedule_state_evaluation(self, delay_seconds: float) -> None:
        """Schedule a state evaluation for all services."""
        if delay_seconds > 0:
            asyncio.get_running_loop().call_later(
                delay=delay_seconds,
                func=lambda: self._evaluate_state_transitions(),
            )
        else:
            raise ValueError("Invalid delay value. Must be greater than 0.")

    def _evaluate_state_transitions(self) -> None:
        """Evaluate all services for state transitions."""
        changed = False

        for name, state in self.service_states.items():
            svc = self.config.services.get(name)
            container_vram = 0
            if svc and svc.container in self.gpu_status.containers:
                container_vram = self.gpu_status.containers[svc.container].vram_mib

            current = state.state
            if current == ServiceStateEnum.evicting:
                # Stay in evicting until explicitly state set
                continue
            if (
                current == ServiceStateEnum.restarting
                and not state.is_healthy
                and (
                    self._clock.time() - state.last_state_change_time
                    < svc.health_check_interval
                )
            ):
                # Stay in restarting state and wait for health check to succeed.
                continue
            if not state.is_healthy:
                changed |= state.transition(
                    ServiceStateEnum.unhealthy, details="health check failed"
                )
                continue
            if state.in_flight > 0:
                changed |= state.transition(
                    ServiceStateEnum.hot, details="received request"
                )
                continue
            if current == ServiceStateEnum.hot:
                changed |= state.transition(
                    ServiceStateEnum.cooling_down,
                    details="request finished",
                )
                if changed:
                    current = state.state
                else:
                    continue
            if current == ServiceStateEnum.cooling_down and (
                self.gpu_status.last_updated is None
                or state.last_state_change_time > self.gpu_status.last_updated
            ):
                continue
            if container_vram > 0:
                changed |= state.transition(
                    ServiceStateEnum.warm, details="VRAM detected"
                )
                continue
            if container_vram == 0:
                changed |= state.transition(
                    ServiceStateEnum.cold, details="VRAM dropped to 0"
                )
                continue

        if changed:
            self._trigger_queue_wake_up()
            self._schedule_dashboard_update()

    def _mark_service_unhealthy(
        self, service_name: str, state: ServiceState, reason: str
    ) -> None:
        """Mark a service as unhealthy and fail it in the wait queue if previously healthy.

        Args:
            service_name: The name of the service to mark unhealthy.
            state: The current ServiceState object for the service.
            reason: A description of why the service is unhealthy (used for logging).
        """
        if state.is_healthy:
            _logger.warning("Service %s became unhealthy (%s)", service_name, reason)
            self.eviction_log.append(
                EvictionEventType.become_unhealthy, service_name, reason
            )
            state.is_healthy = False

    async def _run_health_check(self, service_name: str) -> None:
        """Run a single health check iteration for a given service.

        Spawns the service's configured health check command as a subprocess
        and waits for it to complete within the configured interval. Updates
        the service's health state based on the exit code or any errors.

        Args:
            service_name: The name of the service to check.

        Raises:
            No exceptions are raised; errors are caught and logged, marking
            the service as unhealthy.
        """
        svc = self.config.services.get(service_name)
        if not svc or not svc.health_check_command:
            return

        state = self.service_states[service_name]
        try:
            proc = await asyncio.create_subprocess_shell(
                svc.health_check_command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await asyncio.wait_for(
                    proc.communicate(), timeout=svc.health_check_interval
                )

                if proc.returncode == 0:
                    if not state.is_healthy:
                        _logger.info("Service %s became healthy", service_name)
                        self.eviction_log.append(
                            EvictionEventType.become_healthy, service_name
                        )
                    state.is_healthy = True
                else:
                    self._mark_service_unhealthy(
                        service_name, state, f"exit code {proc.returncode}"
                    )
            finally:
                if proc.returncode is None:
                    try:
                        proc.kill()
                        await proc.wait()
                    except ProcessLookupError:
                        pass
        except (TimeoutError, asyncio.TimeoutError) as e:
            _logger.error("Health check timed out for %s: %s", service_name, e)
            self._mark_service_unhealthy(service_name, state, "timeout")
        except Exception as e:
            _logger.error("Error running health check for %s: %s", service_name, e)
            self._mark_service_unhealthy(service_name, state, f"error {e}")
        self._evaluate_state_transitions()
        self._schedule_dashboard_update()

    async def _health_check_loop(self, service_name: str) -> None:
        """Continuously monitor a service's health in the background.

        Loops indefinitely, running the health check at the configured
        interval until the proxy is stopped or the task is cancelled.

        Args:
            service_name: The name of the service to monitor.

        Raises:
            asyncio.CancelledError: Handled gracefully when the loop is
                cancelled during shutdown.
        """
        svc = self.config.services.get(service_name)
        if not svc:
            return

        try:
            while True:
                start_time = asyncio.get_running_loop().time()
                await self._run_health_check(service_name)
                elapsed = asyncio.get_running_loop().time() - start_time
                await asyncio.sleep(max(0.1, svc.health_check_interval - elapsed))
        except asyncio.CancelledError:
            pass
