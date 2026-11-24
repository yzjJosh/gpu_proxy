# Plugins

To use a plugin, add an entry to the top-level `plugins` list in `config.yaml`. Give each entry a `name`, a `type` and an optional `config` block. Then add that name to the `plugins` list of each service that should use the plugin:

```yaml
plugins:
  - name: lazy_start
    type: LazyStartContainer

services:
  llm:
    ...
    plugins: [lazy_start]
```

| Plugin type | What it does |
| --- | --- |
| [`LazyStartContainer`](#lazystartcontainer) | Starts a service's stopped container when a request arrives. |
| [`OpenAIGuardRail`](#openaiguardrail) | Catches bad replies from an LLM and asks it to try again. |
| [`OpenAIContextModifier`](#openaicontextmodifier) | Trims old reasoning and tool results in long agent chats. It can also repeat the system prompt. |
| [`OpenAITokenStats`](#openaitokenstats) | Shows live token counts and speed on the dashboard. |
| [`OpenAIChatRequestModifier`](#request-modifiers) | Sets fields like `temperature` on every LLM request. |
| [`OpenAIEmbeddingRequestModifier`](#request-modifiers) | Sets fields on every embedding request. |
| [`OpenAIRerankingRequestModifier`](#request-modifiers) | Sets fields on every reranking request. |

`LazyStartContainer` works with any container. The other plugins need a server with an OpenAI-compatible API, like vLLM, SGLang or llama.cpp.

To write your own plugin, see [Writing your own plugin](#writing-your-own-plugin).

## LazyStartContainer

When a request arrives for a service whose container is stopped, this plugin starts the container. Pair the plugin with `eviction_command: "docker stop {container}"` for servers that load their model as soon as they start.

Give the container a Docker health check that passes once the model has loaded. The proxy waits for that check to pass before sending the request. Without a health check, the proxy sends the request as soon as the container starts. The request then fails because the model is still loading.

Don't set `health_check_command` on a service that uses this plugin. A stopped container fails that check, so the proxy marks the service unhealthy. The proxy then sends requests to `backup` before this plugin can start the container. Without a `backup`, the proxy returns a 503 error.

| Option | Default | Meaning |
| --- | --- | --- |
| `start_timeout` | `120` | How many seconds the proxy waits for the Docker health check to pass. After that, the proxy sends the request anyway. |

## OpenAIGuardRail

This plugin watches LLM replies on `/v1/chat/completions` and `/v1/responses`. When a check fails, the plugin keeps the good part of the reply, drops the rest and asks the model to continue. Your app still gets one unbroken reply.

Streaming still works, so text reaches your app as the model writes it. The plugin holds back tool calls until it has checked them.

All checks are off by default, so turn on the ones you want:

```yaml
- name: guard_rail
  type: OpenAIGuardRail
  config:
    break_loops: true
    dedup_tool_calls: true
    validate_tool_calls: true
    force_final_output: true
```

| Option | Default | Meaning |
| --- | --- | --- |
| `break_loops` | `false` | Turn this on to retry when the model starts repeating itself. |
| `min_loop_distance` | `50` | How many characters have to repeat before `break_loops` treats the text as a loop. |
| `dedup_tool_calls` | `false` | Turn this on to retry if the model repeats a tool call with the same arguments after the latest user message. |
| `tool_call_dedup_exclusions` | none | Tool calls that `dedup_tool_calls` lets the model repeat. See [Tool call patterns](#tool-call-patterns). |
| `validate_tool_calls` | `false` | Turn this on to retry when a tool call's arguments aren't valid JSON. |
| `force_final_output` | `false` | Turn this on to retry when a reply ends with no text and no tool call. |
| `max_retry_attempt` | `2` | The most times the plugin retries one reply. |

## OpenAIContextModifier

This plugin saves context space in long agent chats on `/v1/chat/completions` and `/v1/responses`. It removes old reasoning and old or repeated tool results before the request reaches the model. It can also repeat the system prompt as the chat grows.

Several options count turns. A turn is a user message plus everything after it, up to the next user message.

```yaml
- name: context
  type: OpenAIContextModifier
  config:
    preserve_thinking_tokens_from_previous_turns: 2
    preserve_tool_calls_min_window_size: 5
    dedup_tool_calls_per_turn: true
```

| Option | Default | Meaning |
| --- | --- | --- |
| `preserve_thinking_tokens_from_previous_turns` | none | How many recent turns keep the model's reasoning. The plugin removes reasoning from older turns. Set this to `0` to remove all reasoning, or leave it out to keep all reasoning. |
| `preserve_tool_calls_min_window_size` | none | The fewest recent turns that keep their full tool results. Leave this out to keep all tool results. In older turns, the plugin replaces tool results with a short note and shortens long tool call arguments. It trims in batches instead of on every turn, so the model server can reuse its prompt cache. To find when to trim, the plugin adds `total turns × preserve_tool_calls_max_window_size_ratio` to this setting. Once more than that many turns have full tool results, it trims the oldest ones so only the latest `preserve_tool_calls_min_window_size` turns keep them. |
| `preserve_tool_calls_max_window_size_ratio` | `0.5` | How many extra turns can keep their full tool results before a trim, as a fraction of all turns in the chat. This setting only applies when `preserve_tool_calls_min_window_size` is set. The plugin adds `total turns × this ratio` to `preserve_tool_calls_min_window_size` to get the point where it trims. A larger ratio means fewer, bigger trims, so the start of the prompt stays the same on more requests. |
| `tool_call_prune_exclusions` | none | Tool calls whose results the plugin never trims. See [Tool call patterns](#tool-call-patterns). |
| `dedup_tool_calls_per_turn` | `false` | Turn this on to drop repeated tool results. When the model makes the same tool call with the same arguments more than once in a turn, the plugin keeps only the latest result. |
| `repeat_system_prompt_min_window` | none | The fewest turns to wait before repeating the system prompt. Leave this out to never repeat the system prompt. Repeating the prompt helps the model keep following it in long chats. To find how many turns to wait, the plugin adds `total turns × repeat_system_prompt_max_window_ratio` to this setting. Once more than that many turns have passed since the chat began or since the last repeat, it puts a copy of the system prompt before the latest user message. |
| `repeat_system_prompt_max_window_ratio` | `0.5` | The extra turns to wait before repeating the system prompt, as a fraction of all turns in the chat. This setting only applies when `repeat_system_prompt_min_window` is set. The plugin adds `total turns × this ratio` to `repeat_system_prompt_min_window` to get how many turns to wait. |

## OpenAITokenStats

This plugin shows live token counts on the dashboard, along with how fast the model reads the prompt and writes its reply. It works on `/v1/chat/completions`, `/v1/completions` and `/v1/responses`. It has no options:

```yaml
- name: stats
  type: OpenAITokenStats
```

## Request modifiers

These plugins set fields on every request to the paths below. Use them when an app doesn't let you change a setting.

| Plugin type | Works on |
| --- | --- |
| `OpenAIChatRequestModifier` | `/v1/chat/completions`, `/v1/completions`, `/v1/responses` |
| `OpenAIEmbeddingRequestModifier` | `/v1/embeddings`, `/v1/embedding`, `/embedding` |
| `OpenAIRerankingRequestModifier` | `/v1/rerank`, `/reranking` |

Each plugin takes an `overrides` map with the fields to set:

```yaml
- name: sampling
  type: OpenAIChatRequestModifier
  config:
    overrides:
      temperature: 0.6
      max_tokens: 32768
```

Plain values like numbers and strings replace whatever the app sent. Objects merge key by key. Lists merge by position, so your first item merges with the app's first item, and any extra items you set are added at the end.

## Tool call patterns

`tool_call_dedup_exclusions` and `tool_call_prune_exclusions` each take a list of patterns. A pattern can set `tool_name_regex`, `tool_args_regex` or both. When a pattern sets both, a tool call must match both. `tool_args_regex` searches the arguments as raw JSON text.

```yaml
tool_call_prune_exclusions:
  - tool_name_regex: "^read_file$"
  - tool_name_regex: "^shell$"
    tool_args_regex: "git (status|diff)"
```

## Writing your own plugin

Add a Python file to `app/plugins/` with a subclass of `PluginBase` from `app/plugins/base.py`. The proxy loads every plugin file at startup. In `config.yaml`, use the class name as the plugin's `type`.

The proxy creates one copy of the plugin for each service that uses it. The plugin's `config` block arrives as a dictionary in `self.config`.

Your class must define `name()`. Then override the hooks you need:

| Hook | When the proxy calls it |
| --- | --- |
| `on_request` | Before it sends a request to the model server. Return the headers, body and path to send instead. |
| `on_response_headers` | Before it sends the response headers to your app. Return the headers to send. |
| `on_response_chunk` | For each piece of the response body. Return the bytes to send to your app. |
| `prepare_target` | Before it sends a request to the model server. Use it to get the server ready, like starting a container. |
| `overridden_paths` and `handle_overridden_request` | For requests to the paths you list. Your plugin handles these requests itself. It can answer them directly or pass them on to the model server and change the reply. |
| `init_ui` | Once at startup. It gets a function your plugin can call at any time to show its own panel on the dashboard. |

When you override `on_request` or `on_response_chunk`, call the `super()` version too, like the bundled plugins do. The base class uses those calls to track which requests are still running.
