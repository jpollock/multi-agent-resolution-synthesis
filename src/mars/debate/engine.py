"""Top-level debate orchestrator."""

from __future__ import annotations

from mars.analysis.attribution import AttributionAnalyzer
from mars.analysis.costs import compute_costs
from mars.config import AppConfig
from mars.debate.judge import JudgeStrategy
from mars.debate.round_robin import RoundRobinStrategy
from mars.display.renderer import Renderer
from mars.models import DebateConfig, DebateMode, DebateResult
from mars.output.writer import OutputWriter
from mars.providers.registry import get_provider


class DebateEngine:
    def __init__(self, config: DebateConfig, app_config: AppConfig) -> None:
        self.config = config
        self.app_config = app_config

    async def run(self) -> DebateResult:
        # Build providers
        providers = {}
        for name in self.config.providers:
            providers[name] = get_provider(name, self.app_config)

        renderer = Renderer(self.config.verbosity)
        writer = OutputWriter(self.config.output_dir, self.config.prompt)

        renderer.start_debate(
            self.config.prompt,
            list(providers.keys()),
            self.config.mode.value,
        )

        if self.config.mode == DebateMode.JUDGE:
            strategy = JudgeStrategy(providers, self.config, renderer, writer)
        else:
            strategy = RoundRobinStrategy(providers, self.config, renderer, writer)

        result = await strategy.run()

        # Post-debate analysis
        analyzer = AttributionAnalyzer()
        attribution = analyzer.analyze(result)
        costs = compute_costs(result)

        renderer.show_attribution(attribution)
        renderer.show_round_diffs(attribution.round_diffs)
        renderer.show_costs(costs)
        writer.write_attribution(attribution)
        writer.write_round_diffs(attribution.round_diffs)
        writer.write_costs(costs)

        renderer.show_final_answer(result)
        renderer.show_output_path(writer.base_path)

        return result
