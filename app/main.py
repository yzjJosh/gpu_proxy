"""Entry point for GPU Priority Proxy.

Loaded by the systemd unit: ExecStart=python3 -m app.main
"""

import asyncio
import logging
import sys
from pathlib import Path

from aiohttp import web
from app.config import load_config
from app.gpu_monitor import RealClock
from app.plugins.manager import PluginManager
from app.proxy import GpuPriorityProxy
from app.watchdog import EventLoopWatchdog


async def main() -> None:
    """Load config, create proxy, and run the aiohttp web server."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)

    log_level = logging.getLevelName(config.log_level.upper())
    if log_level is None:
        logging.getLogger().warning("Invalid log_level '%s', defaulting to INFO", config.log_level)
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logging.getLogger().setLevel(log_level)
    logging.getLogger("aiohttp").setLevel(log_level)
    proxy = GpuPriorityProxy(config, clock=RealClock())
    plugins_dir = str(Path(__file__).parent / "plugins")
    plugin_manager = PluginManager(
        plugins_dir, config, dashboard_refresh_fn=proxy.schedule_dashboard_update
    )
    proxy.set_plugin_manager(plugin_manager)

    watchdog = EventLoopWatchdog()
    watchdog.start()

    async def _ping_watchdog() -> None:
        """Periodically ping the watchdog to confirm the event loop is alive."""
        while True:
            watchdog.ping()
            await asyncio.sleep(5)

    ping_task = asyncio.create_task(_ping_watchdog())

    await proxy.start()

    app = web.Application(client_max_size=500 * 1024**2)
    app.router.add_route("*", "/{path:.*}", proxy.proxy_handler)

    await _run_server(app, config.listen_port, proxy, watchdog, ping_task)


async def _run_server(
    app: web.Application,
    port: int,
    proxy: GpuPriorityProxy,
    watchdog: EventLoopWatchdog,
    ping_task: asyncio.Task,
) -> None:
    """Run the aiohttp web server until interrupted, then clean up.

    Args:
        app: The configured aiohttp application.
        port: The port to bind.
        proxy: The running proxy to stop on shutdown.
        watchdog: The watchdog to stop on shutdown.
        ping_task: The watchdog ping task to cancel on shutdown.
    """
    runner = web.AppRunner(app)
    try:
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", port)
        await site.start()
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        await runner.cleanup()
        ping_task.cancel()
        try:
            await ping_task
        except asyncio.CancelledError:
            pass
        watchdog.stop()
        await proxy.stop()


if __name__ == "__main__":
    asyncio.run(main())