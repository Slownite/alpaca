from __future__ import annotations

from alpaca.engine.base import Engine


class VLLMEngine(Engine):
    """Real backend: launches and supervises a vLLM OpenAI-compatible server.

    Imports vllm lazily so the module loads without the GPU extra installed.
    """

    def start(self, model: str, port: int, **opts) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def health(self) -> bool:
        raise NotImplementedError

    def endpoint(self) -> str:
        raise NotImplementedError
