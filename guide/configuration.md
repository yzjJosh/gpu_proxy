# Configuration reference

All settings live in `config.yaml`. The [README](../README.md#3-describe-your-services) has a working example.

## Top-level settings

| Option | Default | Meaning |
| --- | --- | --- |
| `listen_port` | `8080` | The port where the proxy accepts requests. The proxy listens on all network interfaces. |
| `total_vram_budget` | `20480` | The most VRAM that all services can use together, in MiB. Set this a little below your GPU's total VRAM to leave room for spikes. The proxy also checks how much VRAM is really free, so you don't need to subtract what your desktop or other programs use. |
| `poll_interval` | `5` | How often the proxy checks VRAM use, in seconds. It uses the latest reading to decide what fits and what to evict. |
| `wait_queue_timeout` | `60` | How many seconds a request waits for enough VRAM. After that, the proxy sends the request to the service's `backup` server. If the service has no `backup`, the proxy returns a 503 error. |
| `eviction_timeout` | `60` | How many seconds a busy service gets to finish its current requests when a more important service needs VRAM. After that, the proxy evicts the busy service anyway. |
| `eviction_command_timeout` | `60` | How many seconds a service's `eviction_command` can run. |
| `connection_timeout` | `30` | How many seconds the proxy waits to connect to a model server. |
| `sock_read_timeout` | `120` | How many seconds the proxy waits for data from a model server. Raise this if your models take a long time to answer. |
| `dashboard_hostname` | `gpu-proxy.local` | The proxy serves the dashboard at this hostname, on `listen_port`. |
| `log_level` | `INFO` | How much the proxy logs: `DEBUG`, `INFO`, `WARNING` or `ERROR`. |
| `services` | none | The services the proxy manages, each under a name you choose. You must include this section. See [Per-service settings](#per-service-settings). |
| `plugins` | none | The plugins that services can use. Each entry has a `name`, a `type` and an optional `config` block. Leave this out if you don't use plugins. See [Plugins](plugins.md). |

## Per-service settings

| Option | Required | Default | Meaning |
| --- | --- | --- | --- |
| `hostname` | yes | none | Requests for this hostname go to this service. Each service needs its own hostname. |
| `target` | yes | none | The URL of the model server, like `http://127.0.0.1:8001`. |
| `priority` | yes, unless `external` is `true` | none | A positive whole number, where `1` is the most important. Each service needs a different number. When VRAM runs short, the proxy evicts idle services before busy ones, starting with the least important. The proxy can evict an idle service even for a less important one, unless you set `is_persistent` to `true`. The proxy only evicts a busy service to make room for a more important one. |
| `vram_budget` | yes, unless `external` is `true` | none | How much VRAM you expect this service to use at its peak, in MiB. It can't be more than `total_vram_budget`. The proxy uses `vram_budget` to decide whether the service fits and what to evict. |
| `container` | yes, unless `external` is `true` | none | The name of the service's Docker container. The proxy uses this name to measure the container's VRAM use and to evict the container. |
| `gpu_routes` | no | none | The URL path prefixes that use the GPU, like `/v1/chat/completions`. Set this on every GPU service. Requests to other paths skip the VRAM check. Without `gpu_routes`, the proxy never checks VRAM for the service. |
| `backup` | no | none | The URL of a fallback server, such as one running the same model on the CPU. The proxy sends requests there when it can't free enough VRAM in time or when the service is unhealthy. Without a `backup`, the proxy returns a 503 error in both cases. By default, when a request fails on `target`, the proxy also retries the request on `backup`. Set `fallback_on_error` to `false` to turn those retries off. |
| `fallback_on_error` | no | `true` | When `true`, the proxy retries a request on `backup` if the request fails on `target`. This setting only applies when `backup` is set. Set it to `false` to use `backup` only when a request times out waiting for VRAM or when the service is unhealthy. |
| `eviction_command` | no | `docker restart {container}` | The command the proxy runs to free the service's VRAM. `{container}` becomes the container name. For servers that load their model at startup, use `docker stop {container}` with the [`LazyStartContainer`](plugins.md#lazystartcontainer) plugin. The command runs without a shell, so put any pipes or `&&` in a script. |
| `is_persistent` | no | `false` | Normally, the proxy can evict an idle service to make room for any other service. When `true`, the proxy never evicts this service for a less important one, even while idle. The proxy can still evict this service for a more important one. |
| `health_check_command` | no | none | A shell command that checks whether the service is healthy. When the command fails, the proxy marks the service unhealthy and sends its requests to `backup`. Without a `backup`, the proxy returns a 503 error. Don't set this on a service that uses `docker stop` with [`LazyStartContainer`](plugins.md#lazystartcontainer). A stopped container fails this check, so requests never reach `LazyStartContainer` to start the container. Give that container a Docker health check instead. |
| `health_check_interval` | no | `5` | How often the proxy runs `health_check_command`, in seconds. |
| `plugins` | no | none | The names of plugins this service uses, from the top-level `plugins` list. |
| `external` | no | `false` | Set this to `true` for a service on another machine or on the CPU. The proxy doesn't manage VRAM for this service, so leave out `priority`, `vram_budget`, `container` and `gpu_routes`. Routing, health checks, backup and the OpenAI plugins still work. `LazyStartContainer` does nothing here, because there's no container to start. |
