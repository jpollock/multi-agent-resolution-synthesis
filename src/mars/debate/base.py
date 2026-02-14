"""Abstract debate strategy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mars.display.renderer import Renderer
    from mars.models import DebateConfig, DebateResult
    from mars.output.writer import OutputWriter
    from mars.providers.base import LLMProvider


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
