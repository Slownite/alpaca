# Architecture

Alpaca makes Ray-distributed vLLM feel like Ollama: pull a model, serve it, and scale it across nodes from a simple CLI.

This document describes both the **design** (the seams and why they exist) and the **contracts** (the concrete behaviour each module must implement). It is the reference a contributor uses to build a single module without having read the rest of the codebase. Each section maps to a GitHub issue.

## Status

The repository is a skeleton. The `Engine` interface ([alpaca/engine/base.py](../alpaca/engine/base.py)) is the only fully-implemented piece. Every other module is a docstring stub or raises `NotImplementedError`. The sections below specify what each stub must become.

## Design principles

1. **Thin orchestration, not a re-implementation.** Alpaca orchestrates vLLM and Ray. It does not reimplement the inference engine, the KV cache, or attention. When a behaviour already exists in vLLM or Ray, shell out to it or call its API rather than rebuilding it.
2. **Engine-agnostic core.** A small `Engine` interface decouples the CLI from the backend. vLLM today; TensorRT-LLM can be added without touching the CLI or cluster code.
3. **Testable without a GPU.** A `MockEngine` (an OpenAI-compatible HTTP stub) lets the entire CLI and Ray orchestration run locally and in CI with no GPU, so only real benchmark runs cost money. Anything the CLI does above the `Engine` seam must be exercisable against `MockEngine`.
4. **Minimal networking glue.** Keep only the advertise-IP wiring that multi-node Ray genuinely requires. No convenience flags, no fallback layers, no persisted config.

## Module layout

```
alpaca/
  cli.py            # argparse, command dispatch                  -> "CLI dispatch"
  engine/base.py    # Engine interface (start/stop/health/endpoint)  [DONE]
  engine/vllm.py    # VLLMEngine (real backend)                   -> "VLLMEngine"
  engine/mock.py    # MockEngine (OpenAI-compatible stub, no GPU) -> "MockEngine"
  models.py         # HF pull/cache + registry                    -> "models"
  process.py        # PID tracking, ps/stop                       -> "process"
  cluster.py        # Ray head/worker/cluster-up, placement groups -> "cluster"
  serve_ray.py      # distributed serve (tensor parallel, multi-node) -> "serve_ray"
  bench/client.py   # async streaming load gen (TTFT/TPOT)        -> "bench/client"
  bench/report.py   # tables / CSV / plot                         -> "bench/report"
tests/              # green against MockEngine, no GPU required    -> "tests"
```

## Runtime data layout

Alpaca keeps per-user state under a single root, resolved in this order:

1. `$ALPACA_HOME` if set,
2. otherwise `$XDG_DATA_HOME/alpaca`,
3. otherwise `~/.local/share/alpaca`.

```
$ALPACA_HOME/
  models/          # Hugging Face cache (HF_HOME points here)
  registry.json    # alias -> {repo_id, revision, path, pulled_at}
  run/
    <name>.json    # one record per running server (see process.py)
```

Model weights live under `models/`. The CLI sets `HF_HOME` to this directory so `huggingface_hub` and vLLM share one cache. Running-server bookkeeping lives under `run/`. No other persisted config exists by design.

## The Engine interface

The seam that makes the core engine-agnostic and GPU-free to test. Defined and stable in [alpaca/engine/base.py](../alpaca/engine/base.py):

```python
class Engine(ABC):
    def start(self, model: str, port: int, **opts) -> None: ...   # launch backend
    def stop(self) -> None: ...                                   # tear down
    def health(self) -> bool: ...                                 # readiness probe
    def endpoint(self) -> str: ...                                # OpenAI-compatible base URL
```

Contract every implementation must honour:

- **`start(model, port, **opts)`** launches an OpenAI-compatible HTTP server bound to `port` and returns once the process is *spawned* (not necessarily ready — readiness is `health`'s job). Calling `start` twice without `stop` is an error. Recognised `opts` keys (all optional): `host`, `dtype`, `max_model_len`, `tensor_parallel_size`, `gpu_memory_utilization`. Implementations ignore keys they don't use.
- **`stop()`** terminates the backend and frees the port. Idempotent: calling `stop` on an already-stopped engine is a no-op, not an error.
- **`health()`** returns `True` once the server answers `GET /v1/models` with HTTP 200, `False` otherwise (including connection-refused). Never raises.
- **`endpoint()`** returns the base URL ending in `/v1` (e.g. `http://127.0.0.1:8000/v1`), valid after `start`. Raises `RuntimeError` if called before `start`.

`VLLMEngine` launches a real vLLM server; `MockEngine` serves a trivial OpenAI-compatible response with no model and no GPU; a future `TensorRTLLMEngine` implements the same four methods and nothing else changes.

### MockEngine

`MockEngine.start` spins up an in-process HTTP server (stdlib `http.server` in a daemon thread is sufficient — no extra dependency) that implements exactly enough of the OpenAI API for the CLI and benchmarks to exercise it:

- `GET /v1/models` → 200 with a one-entry model list echoing the requested `model`.
- `POST /v1/chat/completions` → an OpenAI-shaped chat completion. When the request body has `"stream": true`, respond with `text/event-stream` SSE chunks (`data: {...}\n\n`, terminated by `data: [DONE]`) so the benchmark client's token-timing path works. Emit a deterministic short response token-by-token with a small, configurable inter-token delay so TTFT/TPOT measurements produce non-zero, repeatable numbers.
- `POST /v1/completions` → the legacy-shaped equivalent.

`health` returns `True` once the thread is accepting connections. `endpoint` returns `http://127.0.0.1:<port>/v1`. `stop` shuts the server down and joins the thread. No GPU, no model download, no network egress.

## Process tracking ([process.py](../alpaca/process.py))

Tracks servers started by `serve`/`serve-ray` so `ps` and `stop` work across CLI invocations. A run record is a JSON file at `$ALPACA_HOME/run/<name>.json`:

```json
{
  "name": "llama3-8b",
  "pid": 48213,
  "model": "meta-llama/Meta-Llama-3-8B-Instruct",
  "host": "0.0.0.0",
  "port": 8000,
  "engine": "vllm",
  "started_at": "2026-05-28T10:31:00Z",
  "ray": false
}
```

Public functions:

- `record(meta: dict) -> Path` — write a run record; `name` defaults to a slug of the model if unset.
- `list_running() -> list[dict]` — read every record, drop (and delete) records whose PID is no longer alive (`psutil.pid_exists`), return the live ones. This self-heals stale records left by crashes.
- `find(name: str) -> dict | None` — look up a single record by name.
- `stop(name: str, timeout: float = 10.0) -> bool` — SIGTERM the PID, wait up to `timeout`, SIGKILL if still alive, delete the record. Returns `False` if no such record.

`psutil` (already a dependency) provides cross-process liveness and signalling. No daemon, no database — the filesystem is the source of truth.

## Model management ([models.py](../alpaca/models.py))

Wraps `huggingface_hub` for download/cache and maintains the alias registry at `$ALPACA_HOME/registry.json`.

- `pull(repo_id: str, revision: str | None = None) -> str` — download via `huggingface_hub.snapshot_download` into `models/`, upsert a registry entry, return the local path. Idempotent: a second pull with the same revision is a fast cache hit.
- `resolve(ref: str) -> str` — turn a user-supplied reference (a registry alias, a bare HF repo id, or a local path) into a local path vLLM can load, pulling on demand if it's a known/HF reference and not yet cached.
- `registry() -> list[dict]` — list cached models with repo id, revision, path, and pull time.
- `remove(ref: str) -> bool` — delete cached weights and the registry entry.

Errors surface clearly: gated/nonexistent repos and auth failures (`HF_TOKEN`) must produce an actionable message, not a stack trace.

## CLI ([cli.py](../alpaca/cli.py))

`argparse` with one subparser per command. The parser skeleton and command list already exist; the work is wiring options and dispatching to handlers. Resolve `--engine {vllm,mock}` globally (default `vllm`) so every command can run against `MockEngine` in tests and demos.

| Command | Purpose | Key options |
|---------|---------|-------------|
| `pull <model>` | Download and cache a model from Hugging Face | `--revision` |
| `serve <model>` | Start a single-node server | `--name`, `--port` (8000), `--host` (127.0.0.1), `--dtype`, `--max-model-len`, `--engine` |
| `run <name> <prompt>` | Send a prompt to a running server | `--max-tokens`, `--stream` |
| `ps` | List running servers | — |
| `stop <name>` | Stop a server | — |
| `cluster-up` | Start a local Ray head + register this node | `--port`, `--dashboard-port` |
| `ray-head` | Start a Ray head node | `--port`, `--advertise-ip` |
| `ray-worker <head-address>` | Join an existing Ray head | `--advertise-ip` |
| `serve-ray <model>` | Distributed serve (tensor parallel, multi-node) | `--name`, `--port`, `--tensor-parallel-size`, `--engine` |
| `bench <endpoint>` | Load-test a server and report throughput and latency | `--concurrency`, `--num-prompts`, `--max-tokens`, `--output {table,csv}`, `--plot` |

Each handler is a thin function: parse args, call into `models`/`process`/`engine`/`cluster`/`serve_ray`/`bench`, print a result, return an exit code. `main()` dispatches and turns exceptions into a non-zero exit with a one-line message (full traceback only under `--debug`).

## Single-node serving

`serve` ties the pieces together: `models.resolve` the model, instantiate the chosen `Engine`, `start` it, poll `health` until ready (with a timeout), then `process.record` the running server. `run` reads the record with `process.find`, POSTs to the server's `/v1/chat/completions` (streaming when `--stream`), and prints the completion. `ps` prints `process.list_running`; `stop` calls `process.stop`. All of this works against `MockEngine` with no GPU.

## Distributed serving

Ray manages a head node plus workers. **Cluster lifecycle** lives in [cluster.py](../alpaca/cluster.py): `ray-head` starts a head (`ray start --head`), `ray-worker` joins one (`ray start --address=...`), and `cluster-up` is the local convenience that starts a head and treats the current machine as the only node. Each command must set the node's advertised address (`--advertise-ip`, falling back to the primary outbound interface) so workers on other machines can actually reach the head — this is the one piece of networking glue the design keeps.

**Distributed serve** lives in [serve_ray.py](../alpaca/serve_ray.py): connect to the running Ray cluster (`ray.init(address="auto")`), create a **placement group** that reserves one GPU bundle per tensor-parallel rank (pinning vLLM workers to GPUs), and launch a vLLM server with `tensor_parallel_size = N` so the model is sharded across those GPUs — within a node, or across nodes when the cluster spans machines. vLLM needs each worker to advertise a reachable IP; export `VLLM_HOST_IP` to the node's advertised address before launch. On success, `process.record` the server with `ray: true` so `ps`/`stop` manage it like any other.

## Benchmarking

The `bench` command measures serving performance against any OpenAI-compatible endpoint (real vLLM or `MockEngine`).

- **Client ([bench/client.py](../alpaca/bench/client.py)):** an `async` load generator (`httpx.AsyncClient`) that fires `--num-prompts` streaming chat-completion requests at `--concurrency` in flight. For each request it timestamps send, first token, and every subsequent token, yielding a per-request record of TTFT and inter-token gaps. A semaphore bounds concurrency; results are collected into a list of records.
- **Metrics:** throughput (output tokens/s, requests/s) and latency split into **TTFT** (time to first token) and **TPOT** (time per output token), reported at **p50 and p99**.
- **Report ([bench/report.py](../alpaca/bench/report.py)):** aggregate the client's per-request records into those metrics and render them — a plain-text table by default, `--output csv` for machine-readable output, and `--plot` for a latency/throughput chart (matplotlib, imported lazily so the core install stays light).
- **Method:** stream responses token-by-token and timestamp the first token and each subsequent token. Aggregate numbers are cross-checked against vLLM's `benchmark_serving.py`. Because `MockEngine` streams with a deterministic delay, the whole pipeline is testable without a GPU.

## Testing

Everything above the `Engine` seam runs against `MockEngine` in CI with no GPU. The suite covers: CLI parsing and dispatch for every command; the `serve → ps → run → stop` lifecycle end-to-end against `MockEngine`; `process` record/list/stale-reaping; `models` registry behaviour (download mocked); and the `bench` client/report pipeline against `MockEngine`'s streaming endpoint. `pytest-asyncio` (already a dev dependency) covers the async client. GPU-only paths (`VLLMEngine.start`, real Ray serve) are marked and skipped in CI.

## CI

A GitHub Actions workflow runs the no-GPU suite on every push and PR: install the core extras (not `gpu`), run `pytest` across the supported Python versions (3.10–3.12). GPU-marked tests are excluded. This is what keeps "green against MockEngine, no GPU required" true.

## Build order

The modules form a dependency chain. Recommended order (also encoded in the issues as "blocked by" links):

1. **MockEngine** — unblocks every test and demo; depends on nothing.
2. **CLI dispatch** — wire options + dispatch; lets every later command be driven end-to-end.
3. **process** and **models** — independent of each other; both feed single-node serving.
4. **VLLMEngine** — the real backend (GPU paths skipped in CI).
5. **Single-node commands** (`serve`/`run`/`ps`/`stop`/`pull`) — compose the above into the Ollama-style flow.
6. **cluster** then **serve_ray** — distributed path; serve_ray depends on cluster.
7. **bench/client** then **bench/report** — measurement; client testable against MockEngine first.
8. **tests** and **CI** — grow alongside the above; CI green is the definition of done for the no-GPU surface.

## Roadmap

- Kubernetes deploy: containerize, request GPUs via the device plugin, put a Service in front.
- TensorRT-LLM backend behind the `Engine` interface, then a vLLM vs TensorRT-LLM comparison on identical hardware.
