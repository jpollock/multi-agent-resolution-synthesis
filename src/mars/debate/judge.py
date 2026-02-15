"""Judge debate strategy - one model evaluates all others."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mars.debate.base import DebateStrategy
from mars.debate.prompts import EVALUATION_RULES, JUDGE_PREAMBLE
from mars.models import (
    DebateResult,
    DebateRound,
    LLMResponse,
    Message,
)

if TYPE_CHECKING:
    from mars.providers.base import LLMProvider


class JudgeStrategy(DebateStrategy):
    async def run(self) -> DebateResult:
        result = DebateResult(
            prompt=self.config.prompt,
            context=self.config.context,
            mode=self.config.mode,
        )

        self.writer.write_prompt(self.config.prompt, self.config.context)

        judge_name = self.config.judge_provider
        if not judge_name:
            raise ValueError("Judge mode requires --judge-provider / -j")

        if judge_name not in self.providers:
            raise ValueError(
                f"Judge provider '{judge_name}' not in selected providers. "
                f"Available: {', '.join(self.providers)}"
            )

        # Step 1: All providers answer
        self.renderer.start_round(1)
        responses = await self._initial_round()
        debate_round = DebateRound(round_number=1, responses=responses)
        result.rounds.append(debate_round)
        self.writer.write_round(1, responses)

        # Step 2: Judge evaluates
        self.renderer.start_round(2)
        judge_provider = self.providers[judge_name]
        judge_model = self.config.model_overrides.get(judge_name)

        self.renderer.start_work([judge_name], "Judging")
        judgment = await self._judge(judge_provider, judge_model, responses)
        self.renderer.stop_work()

        final, resolution = self._parse_final_answer(judgment.content)

        result.final_answer = final
        result.resolution_reasoning = resolution
        result.convergence_reason = f"Judge ({judge_name}) evaluated all responses."

        judge_round = DebateRound(round_number=2, responses=[judgment])
        result.rounds.append(judge_round)
        self.writer.write_round(2, [judgment])
        self.writer.write_convergence(result.convergence_reason)
        self.writer.write_resolution(resolution)
        self.writer.write_final(final)

        return result

    async def _initial_round(self) -> list[LLMResponse]:
        system_msg = self._build_system()
        user_msg = Message(role="user", content=self._full_prompt_with_context())
        messages = [system_msg, user_msg] if system_msg else [user_msg]
        return await self._gather_responses(list(self.providers.items()), messages, phase="Round 1")

    async def _judge(
        self,
        judge: LLMProvider,
        model: str | None,
        responses: list[LLMResponse],
    ) -> LLMResponse:
        parts = [self._full_prompt_with_context()]
        parts.append("\n---\n\nResponses from each model:\n")
        for r in responses:
            parts.append(f"--- {r.provider} ({r.model}) ---\n{r.content}\n")
        parts.append(JUDGE_PREAMBLE + EVALUATION_RULES)

        system_msg = self._build_system()
        messages: list[Message] = []
        if system_msg:
            messages.append(system_msg)
        messages.append(Message(role="user", content="\n".join(parts)))

        return await self._get_response(judge, messages, model)
