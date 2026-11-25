# gpu-proxy - GPU Priority Reverse Proxy

## Project Context

A Python `aiohttp` reverse proxy that multiplexes multiple GPU-backed services sharing a single GPU's VRAM. It demultiplexes incoming requests by `Host` header, forwards them to containerized services, and actively arbitrates GPU VRAM contention between them — admitting, queueing, and evicting services based on priority and live VRAM usage.

**Tech stack:** Python 3.12, aiohttp, aiodocker, nvidia-ml-py, PyYAML, tiktoken (Python); Vue 3 + Vite (dashboard); systemd (deployment). Runs directly on an NVIDIA GPU host (not containerized).

**Key modules** (`app/`):
- `proxy.py` — main proxy: Host-based routing, HTTP/WebSocket/SSE streaming relay, target→backup failover, internal loopback detection, and dashboard serving
- `responses.py` — mutating response interface, HTTP, WebSocket, and capture responses, and header and body mutator composition
- `config.py` — YAML config loading and strict validation (`ServiceConfig`, `ProxyConfig`)
- `budget.py` — VRAM budget math: `compute_effective_remaining`, `get_eviction_candidates`
- `wait_queue.py` — priority-ordered VRAM admission queue (admit / wait / backup decision)
- `service_state.py` — per-service state machine (`cold → warm → hot → cooling_down → evicting → restarting → unhealthy`) with validated transitions and in-flight tracking
- `gpu_monitor.py` — background polling of GPU + per-container VRAM into a `GPUStatus` snapshot
- `gpu_usage.py` — reads the GPU through NVML and gives each Docker container exactly the VRAM NVML reports for its processes (containers found via `/proc/<pid>/cgroup`); VRAM no process owns is never attributed to a container
- `eviction.py` — bounded eviction/routing event log (quiet + full buffers)
- `watchdog.py` — background thread that detects asyncio event-loop stalls
- `main.py` — entry point (`python3 -m app.main`)
- `plugins/` — dynamic plugin system (see below)

**Plugin system** (`app/plugins/`): `PluginManager` discovers `PluginBase` subclasses from `app/plugins/*.py` at startup and instantiates a fresh instance per service that references it. Plugins run in a request pipeline (`on_request`) and a streaming response-chunk pipeline (`on_response_chunk`), and can claim paths to serve/intercept directly (`overridden_paths` + `handle_overridden_request`). Plugins may publish HTML telemetry panels to the dashboard via `init_ui(publish)`.

Bundled plugins:
- `OpenAIRequestModifier` (base) + `OpenAI{Chat,Embedding,Reranking}RequestModifier` — deep-merge configured overrides into OpenAI request bodies
- `OpenAIContextModifier` — prunes/dedupes reasoning & tool-call context to fit the window
- `OpenAIGuardRail` — intercepts `/v1/chat/completions` and `/v1/responses`, forces streaming, detects repetition loops / duplicate + invalid tool calls / missing final output, and retries the model
- `OpenAITokenStats` — real-time token counts and tokens-per-second (prefill + generate) telemetry

The four OpenAI plugins read and write the wire format through per-API format adapters in `app/plugins/openai_api_formats/`, so chat completions, legacy completions, and the Responses API share the same logic:
- `openai_api_formats/base.py` — `OpenAIAPIFormat`, `ConversationAPIFormat`, `ChoicesBasedAPIFormat`, shared data types, and path lookup helpers
- `openai_api_formats/chat_completions.py` — `ChatCompletionsAPIFormat`
- `openai_api_formats/legacy_completions.py` — `LegacyCompletionsAPIFormat`
- `openai_api_formats/responses.py` — `ResponsesAPIFormat`: Responses message history editing, stream event parsing, multi-attempt retry stream combining, and non-streaming response assembly

**Configuration:** `config.yaml` at the repo root (template: `config.example.yaml`). Each service declares a `priority`, `vram_budget`, `hostname` (the demux key), `target` URL, optional `backup` URL, `gpu_routes`, and optional `health_check_command` / `is_persistent`. `total_vram_budget` and `listen_port` are set at the top level.

**Host requirements:** `app/gpu_usage.py` needs the NVIDIA driver (NVML, memory info v2), the Docker socket, and read access to `/proc/<pid>/cgroup` for GPU processes, so the proxy runs as root on the GPU host.

**Environment variables:** none — all runtime configuration lives in `config.yaml`.

## Running Tests

Tests use pytest with asyncio auto mode (`asyncio_mode = "auto"` in `pyproject.toml`). Test files mirror the `app/` layout under `tests/app/`. A `conftest.py` provides shared fixtures, SSE builders, a virtual `FakeClock`, test-only fake plugins, an **autouse fixture that mocks `asyncio.create_subprocess_exec`** so no test runs a real `docker restart`, and `FakeNvidiaHost`, which stands in for NVML, the Docker API and `/proc` (via `with host.installed():`) so no test touches GPU hardware.

**From `/var/gpu_proxy`** (use the project venv):
```bash
# Run all tests
venv/bin/pytest

# Run a specific test file
venv/bin/pytest tests/app/test_wait_queue.py

# Run tests matching a name
venv/bin/pytest -k eviction

# Deselect live/integration tests (registered in conftest.py)
venv/bin/pytest -m "not live"
```

There is no `run_tests.sh`; run `venv/bin/pytest` directly from the repo root.

## Build and Deploy

Deployment is a **systemd service on the GPU host** (`gpu-proxy.service`, installed at `/etc/systemd/system/gpu-proxy.service`). It runs `venv/bin/python3 -m app.main` from `WorkingDirectory=/var/gpu_proxy` and appends output to `/var/log/gpu-proxy.log`.

```bash
# After editing config.yaml or app code, restart the service
sudo systemctl daemon-reload   # only if the .service unit file changed
sudo systemctl restart gpu-proxy

# Check status / recent logs
systemctl status gpu-proxy
tail -f /var/log/gpu-proxy.log

# Enable on boot
sudo systemctl enable gpu-proxy
```

The proxy listens on the configured `listen_port` (default `8080`) and serves the dashboard at the configured `dashboard_hostname` (default `gpu-proxy.local`).

**Dashboard frontend** (`dashboard/`, Vue 3 + Vite): the built SPA is served from `dashboard/dist/`. After changing dashboard source, rebuild and the proxy will pick it up:
```bash
cd dashboard
npm install   # first time only
npm run build # outputs to dashboard/dist/
```

**Rust loop detector extension** (`rust/loop_detector/`, PyO3 + maturin): `OpenAIGuardRail` requires the compiled `gpu_proxy_loop_detector` module installed in `/var/gpu_proxy/venv`. After changing Rust source or upgrading the venv Python minor version, rebuild and install it:
```bash
venv/bin/pip install maturin                                                                       # first time only
VIRTUAL_ENV=/var/gpu_proxy/venv venv/bin/maturin develop --release --manifest-path rust/loop_detector/Cargo.toml
cargo test --manifest-path rust/loop_detector/Cargo.toml                                           # Rust unit & oracle tests
```

## Code Standards

Refer to the **code-standard** skill for the full set of coding conventions and rules. The skill must be loaded before writing any new code.

After making any code changes:

1. **Self-check lint compliance** — run the lint script to verify the code passes all linting checks (lint the whole project, not a single file — cross-file rules need full context):

```bash
python3 /root/.config/opencode/skills/code-standard/scripts/lint.py .
```

The baseline already contains a few pre-approved `coding.skip-lint` notes (e.g. the `sys._current_frames()` access in `watchdog.py` and plugin `__all__` exports loaded by reflection). Do not add new `lint-skip` directives without explicit user approval.

2. **Code review** — once lint passes, invoke the **code-review** skill to perform a code review of all changes before considering the work complete.

## Error Handling

Classify errors by recoverability and pick the matching failure mode:

- **Non-recoverable errors — fail fast, fail deterministically.** Conditions that are a property of the deployment rather than of transient runtime state (invalid config, violated invariants) must raise immediately at the point they're detected, with a message that names the exact problem. Do **not** log-and-continue, drop the bad input, or return an empty/sentinel value — that hides the defect and degrades behavior silently.

- **Temporary or input-related runtime errors — fail gracefully with an error log.** Conditions that are transient or tied to a single request/GPU sample should be logged with enough context to debug, and recovered from so the service keeps serving. Don't crash the process or tear down unrelated in-flight state.

When in doubt, ask: *will this still be true on the next request / next poll cycle?* Yes → graceful + log. No → fail fast.
