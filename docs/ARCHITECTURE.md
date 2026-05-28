# Architecture

> This file is destined for the Alpaca repo root. Drop it into the rebuild branch.

Alpaca makes Ray-distributed vLLM feel like Ollama: pull a model, serve it, and scale it across nodes from a simple CLI.

## Design principles

1. **Thin orchestration, not a re-implementation.** Alpaca orchestrates vLLM and Ray. It does not reimplement the inference engine, the KV cache, or attention.
2. **Engine-agnostic core.** A small `Engine` interface decouples the CLI from the backend. vLLM today; TensorRT-LLM can be added without touching the CLI or cluster code.
3. **Testable without a GPU.** A `MockEngine` (an OpenAI-compatible HTTP stub) lets the entire CLI and Ray orchestration run locally and in CI with no GPU, so only real benchmark runs cost money.
4. **Minimal networking glue.** Keep only the advertise-IP wiring that multi-node Ray genuinely requires. No convenience flags, no fallback layers, no persisted config.

## Module layout

```
alpaca/
  cli.py            # argparse, command dispatch
  engine/base.py    # Engine interface (start/stop/health/endpoint)
  engine/vllm.py    # VLLMEngine (real backend)
  engine/mock.py    # MockEngine (OpenAI-compatible stub, no GPU)
  models.py         # HF pull/cache + registry
  process.py        # PID tracking, ps/stop
  cluster.py        # Ray head/worker/cluster-up, placement groups
  serve_ray.py      # distributed serve (tensor parallel, multi-node)
  bench/client.py   # async streaming load gen (TTFT/TPOT)
  bench/report.py   # tables / CSV / plot
tests/              # green against MockEngine, no GPU required
```

## The Engine interface

The seam that makes the core engine-agnostic and GPU-free to test.

```python
class Engine:
    def start(self, model: str, port: int, **opts) -> None: ...   # launch backend
    def stop(self) -> None: ...                                   # tear down
    def health(self) -> bool: ...                                 # readiness probe
    def endpoint(self) -> str: ...                                # OpenAI-compatible base URL
```

- `VLLMEngine` launches and supervises a vLLM OpenAI-compatible server.
- `MockEngine` serves a trivial OpenAI-compatible response with no model and no GPU, for local and CI testing of everything above the engine.
- A future `TensorRTLLMEngine` implements the same four methods; nothing else changes.

## Command surface

| Command | Purpose |
|---------|---------|
| `pull` | Download and cache a model from Hugging Face |
| `serve` | Start a single-node vLLM server |
| `run` | Send a prompt to a running server |
| `ps` | List running servers |
| `stop` | Stop a server |
| `cluster-up` / `ray-head` / `ray-worker` | Ray cluster lifecycle |
| `serve-ray` | Distributed serve (tensor parallel, multi-node) |
| `bench` | Load-test a server and report throughput and latency |

## Distributed serving

Ray manages a head node plus workers. Placement groups pin vLLM workers to GPUs. Tensor parallelism splits a model across GPUs within a node, and across nodes when the cluster spans machines. Multi-node serving advertises the head address so workers can reach it.

## Benchmarking

The `bench` command measures serving performance.

- **Metrics:** throughput (output tokens/s, requests/s) and latency split into TTFT (time to first token) and TPOT (time per output token), reported at p50 and p99.
- **Axes:** tensor-parallel degree (GPU count), request concurrency, and sequence length.
- **Method:** stream responses token-by-token and timestamp the first token and each subsequent token. Aggregate numbers are cross-checked against vLLM's `benchmark_serving.py`.

## Roadmap

- Kubernetes deploy: containerize, request GPUs via the device plugin, put a Service in front.
- TensorRT-LLM backend behind the `Engine` interface, then a vLLM vs TensorRT-LLM comparison on identical hardware.
