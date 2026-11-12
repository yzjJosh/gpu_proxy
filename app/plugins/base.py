"""Plugin base class for request transformation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from aiohttp import web
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Awaitable, Callable, Generator

from app.config import ServiceConfig
from app.responses import AbstractMutatingResponse

__all__ = ["PluginBase", "forward_handler", "make_forward_guard"]

_forward_active: ContextVar[bool] = ContextVar("_forward_active", default=False)


@contextmanager
def forward_handler() -> Generator[None, None, None]:
    """Activate the forward guard so that forward() calls are allowed.

    Must be entered before invoking a plugin's handle_overridden_request.
    The guarded forward() callable will raise RuntimeError if called
    outside this context.
    """
    token = _forward_active.set(True)
    try:
        yield
    finally:
        _forward_active.reset(token)


def make_forward_guard(fn: Callable) -> Callable:
    """Wrap a callable that can only be invoked within an active override handler."""
    async def guarded(
        headers: dict[str, str],
        body: bytes,
        path: str,
        request: web.Request,
        response: AbstractMutatingResponse,
    ) -> bool:
        if not _forward_active.get(False):
            raise RuntimeError(
                "forward() can only be called from within a path-override handler"
            )
        return await fn(headers, body, path, request, response)
    return guarded


class PluginBase(ABC):
    """Abstract base class for request transformation plugins.

    All plugins must extend this class and implement the
    name and on_request methods. Optional hooks on_response_chunk
    and init_ui are provided with no-op defaults.
    """

    def __init__(
        self, config: dict, service_config: ServiceConfig | None = None
    ) -> None:
        """Initialize the plugin with configuration.

        Args:
            config: Plugin configuration dictionary
            service_config: The ServiceConfig of the service this instance
                belongs to, or None when the plugin was built standalone
        """
        self.config = dict(config)
        self.service_config: ServiceConfig | None = service_config
        self._ongoing_requests: set[int] = set()

    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this plugin.

        Returns:
            Plugin identifier string
        """

    async def on_request(
        self, request_id: int, headers: dict[str, str], body: bytes, path: str
    ) -> tuple[dict[str, str], bytes, str]:
        """Transform an incoming request before proxying.

        Default implementation returns the request unchanged. Override to
        inspect or modify request headers, body, or path. May perform
        blocking or async work.

        Args:
            request_id: Unique identifier for this client request.
            headers: The request headers dict
            body: The raw request body bytes
            path: The request URL path

        Returns:
            Tuple of (modified headers, modified body bytes, modified path)
        """
        self._ongoing_requests.add(request_id)
        return headers, body, path

    def on_response_headers(
        self, _request_id: int, headers: dict[str, str], _status_code: int
    ) -> dict[str, str]:
        """Return the headers to send to the client.

        Args:
            _request_id: The client request id.
            headers: The response headers.
            _status_code: The response status code.

        Returns:
            The response headers.
        """
        return headers

    async def on_response_chunk(
        self,
        request_id: int,
        _headers: dict[str, str],
        chunk: bytes,
        _path: str,
        _status_code: int,
        is_final: bool,
    ) -> bytes:
        """Process a chunk of the upstream response.

        Default implementation returns the chunk unchanged. Override to
        inspect or transform streaming response data. May perform blocking
        or async work.

        Args:
            request_id: Unique identifier for this client request.
            chunk: The raw response chunk bytes
            is_final: Whether this is the final chunk of the response

        Note:
            The unused ``headers``, ``path`` and ``status_code`` parameters are
            prefixed with an underscore so the hook stays stable for overriders
            who pass arguments positionally, while satisfying the linter.

        Returns:
            The response chunk, possibly modified.
        """
        if is_final:
            self._ongoing_requests.discard(request_id)
        return chunk

    async def prepare_target(self, _request_id: int, _path: str) -> None:
        """Called before traffic is dispatched to this service's target.

        Default implementation is a no-op. Override to bring the target to a
        usable state before the request reaches it (e.g. start a stopped
        container and wait until it is ready). May block.

        Args:
            _request_id: Unique identifier for this client request.
            _path: The request URL path

        Note:
            The unused parameters are prefixed with an underscore so the hook
            stays stable for overriders who pass arguments positionally, while
            satisfying the linter.
        """

    def overridden_paths(self) -> list[str]:
        """Return list of paths this plugin wants to handle directly."""
        return []

    async def handle_overridden_request(
        self,
        _request_id: int,
        _headers: dict[str, str],
        _body: bytes,
        _path: str,
        _web_request: web.Request,
        _forward: Callable[[dict[str, str], bytes, str, web.Request, AbstractMutatingResponse], Awaitable[bool]],
        _response: AbstractMutatingResponse,
    ) -> bool:
        """Handle an overridden path by the plugin.

        The plugin can serve paths that don't exist on the backend, or intercept
        and transform both request and response.

        Args:
            _request_id: Unique identifier for this client request.
            _headers: Request headers
            _body: Request body bytes
            _path: Request path
            _web_request: The aiohttp web.Request
            _forward: Callable to delegate to the real backend. Takes (_headers, _body, _path, _web_request, _response).
                The plugin MUST pass a response object — forward writes to it.
            _response: The client's StreamResponse — plugin writes final output here.

        Note:
            The unused parameters are prefixed with an underscore so the hook
            stays stable for overriders who pass arguments positionally, while
            satisfying the linter.

        Returns:
            True on success, False on failure.
        """
        return False

    def init_ui(self, _publish: Callable[[str, str], None]) -> None:
        """Initialize plugin-specific UI components.

        Default implementation is a no-op. Override to register
        UI elements using the provided publish function.

        Args:
            _publish: Callback to publish UI elements. Takes (title: str, html: str).

        Note:
            The unused ``_publish`` parameter is prefixed with an underscore so
            the hook stays stable for overriders who pass the callback
            positionally, while satisfying the linter.
        """

    def is_request_ongoing(self, request_id: int) -> bool:
        """Return whether request_id is currently tracked as ongoing.

        A request is ongoing if on_request() was called but
        on_response_chunk(is_final=True) has not yet completed.
        """
        return request_id in self._ongoing_requests

    def on_cancelled(self, request_id: int) -> None:
        """Called when on_request() completed but the response stream was interrupted.

        Only invoked when is_request_ongoing(request_id) returns True.
        Override to clean up per-request state.
        Default implementation removes request_id from _ongoing_requests.
        """
        self._ongoing_requests.discard(request_id)
