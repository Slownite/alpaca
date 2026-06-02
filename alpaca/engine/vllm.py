from __future__ import annotations

import subprocess

import httpx

from alpaca.engine.base import Engine


_OPT_TO_FLAG = {
    "dtype": "--dtype",
    "max_model_len": "--max-model-len",
    "tensor_parallel_size": "--tensor-parallel-size",
    "gpu_memory_utilization": "--gpu-memory-utilization",
}


class VLLMEngine(Engine):
    """Real backend: launches and supervises a vLLM OpenAI-compatible server.

    Imports vllm lazily so the module loads without the GPU extra installed.
    """

    process: subprocess.Popen | None = None
    host: str = "127.0.0.1"
    port: int | None = None

    def start(self, model: str, port: int, **opts) -> None:
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("VLLMEngine already started; call stop() first")

        self.host = opts.get("host", "127.0.0.1")
        self.port = port

        cmd = ["vllm", "serve", model, "--host", self.host, "--port", str(port)]
        for key, flag in _OPT_TO_FLAG.items():
            if key in opts:
                cmd += [flag, str(opts[key])]

        self.process = subprocess.Popen(cmd)

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None
        self.port = None

    def health(self) -> bool:
        if self.process is None or self.port is None:
            return False
        try:
            response = httpx.get(
                f"http://{self.host}:{self.port}/v1/models", timeout=1.0
            )
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    def endpoint(self) -> str:
        if self.process is None or self.port is None:
            raise RuntimeError("VLLMEngine not started; call start() first")
        return f"http://{self.host}:{self.port}/v1"
