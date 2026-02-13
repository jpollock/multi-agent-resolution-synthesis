"""Abstract debate strategy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mar.display.renderer import Renderer
    from mar.models import DebateConfig, DebateResult
    from mar.output.writer import OutputWriter
    from mar.providers.base import LLMProvider


class DebateStrategy(ABC):
    def __init__(
        self,
        providers: dict[str, LLMProvider],
        config: DebateConfig,
        renderer: Renderer,
        writer: OutputWriter,
    ) -> None:
        self.providers = providers
        self.config = config
        self.renderer = renderer
        self.writer = writer

    @abstractmethod
    async def run(self) -> DebateResult: ...
