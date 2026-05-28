from __future__ import annotations

from abc import ABC, abstractmethod


class Engine(ABC):
    """Backend seam that decouples the CLI from the inference engine.

    Implementations launch and supervise an OpenAI-compatible server. The CLI
    and Ray orchestration only ever talk to this interface, so a GPU-free
    MockEngine can stand in for vLLM during local runs and CI.
    """

    @abstractmethod
    def start(self, model: str, port: int, **opts) -> None:
        """Launch the backend serving ``model`` on ``port``."""

    @abstractmethod
    def stop(self) -> None:
        """Tear down the backend."""

    @abstractmethod
    def health(self) -> bool:
        """Return True once the backend is ready to serve."""

    @abstractmethod
    def endpoint(self) -> str:
        """Return the OpenAI-compatible base URL."""
