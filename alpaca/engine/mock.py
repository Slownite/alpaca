from __future__ import annotations

from alpaca.engine.base import Engine


class MockEngine(Engine):
    """OpenAI-compatible stub with no model and no GPU.

    Lets the CLI and Ray orchestration run locally and in CI; only real
    benchmark runs need hardware.
    """

    def start(self, model: str, port: int, **opts) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def health(self) -> bool:
        raise NotImplementedError

    def endpoint(self) -> str:
        raise NotImplementedError
