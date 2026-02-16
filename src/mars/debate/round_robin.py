"""Round-robin debate strategy with iterative critique."""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING

import click

from mars.debate.base import DebateStrategy, _format_provider_error
from mars.debate.prompts import CRITIQUE_INSTRUCTIONS, EVALUATION_RULES, SYNTHESIS_PREAMBLE
from mars.models import (
    Critique,
    DebateResult,
    DebateRound,
    LLMResponse,
    Message,
    provider_base_name,
)

if TYPE_CHECKING:
    from mars.providers.base import LLMProvider


class RoundRobinStrategy(DebateStrategy):
    async def run(self) -> DebateResult:
        result = DebateResult(
            prompt=self.config.prompt,
            context=self.config.context,
            mode=self.config.mode,
        )

        self.writer.write_prompt(self.config.prompt, self.config.context)

        latest_answers: dict[str, LLMResponse] = {}

        for round_num in range(1, self.config.max_rounds + 1):
            self.renderer.start_round(round_num)
            debate_round = DebateRound(round_number=round_num)

            if round_num == 1:
                responses = await self._initial_round()
                if not responses:
                    raise click.ClickException(
                        "All providers failed in round 1. Check model names and provider configuration."
                    )
                debate_round.responses = responses
                self.writer.write_round(round_num, responses)
                for r in responses:
                    latest_answers[r.provider] = r
            else:
                critiques, responses = await self._critique_round(round_num, latest_answers)
                debate_round.critiques = critiques
                debate_round.responses = responses
                self.writer.write_round(round_num, responses, critiques)

                prev = latest_answers
                new_answers = {r.provider: r for r in responses}
                if self._has_converged(prev, new_answers):
                    reason = (
                        f"Answers converged after round {round_num} "
                        f"(similarity threshold {self.config.convergence_threshold} reached)."
                    )
                    result.convergence_reason = reason
                    self.renderer.show_convergence(reason)
                    self.writer.write_convergence(reason)
                    for r in responses:
                        latest_answers[r.provider] = r
                    result.rounds.append(debate_round)
                    break

                for r in responses:
                    latest_answers[r.provider] = r

            result.rounds.append(debate_round)
        else:
            reason = f"Maximum rounds ({self.config.max_rounds}) reached."
            result.convergence_reason = reason
            self.renderer.show_convergence(reason)
            self.writer.write_convergence(reason)

        final, resolution = await self._synthesize(latest_answers)
        result.final_answer = final
        result.resolution_reasoning = resolution
        self.writer.write_resolution(resolution)
        self.writer.write_final(final)

        return result

    async def _initial_round(self) -> list[LLMResponse]:
        system_msg = self._build_system()
        user_msg = Message(role="user", content=self._full_prompt_with_context())
        messages = [system_msg, user_msg] if system_msg else [user_msg]

        return await self._gather_responses(list(self.providers.items()), messages, phase="Round 1")

    async def _critique_round(
        self, round_num: int, latest: dict[str, LLMResponse]
    ) -> tuple[list[Critique], list[LLMResponse]]:
        critiques: list[Critique] = []

        critique_items: list[tuple[str, LLMProvider, list[Message]]] = []
        for name, provider in self.providers.items():
            if name not in latest:
                continue
            others = {k: v for k, v in latest.items() if k != name}
            if not others:
                continue
            msgs = self._build_critique_prompt(name, latest[name], others, round_num)
            critique_items.append((name, provider, msgs))

        responses: list[LLMResponse] = []

        if self.config.verbosity.value == "verbose":
            for name, provider, msgs in critique_items:
                model = self.config.model_overrides.get(name)
                try:
                    r = await self._get_response(provider, msgs, model, participant_id=name)
                    for other_name in latest:
                        if other_name != name:
                            critiques.append(
                                Critique(
                                    author=name,
                                    target=other_name,
                                    content=r.content,
                                )
                            )
                    responses.append(r)
                except Exception as e:
                    self.renderer.show_error(name, _format_provider_error(e))
        else:
            critique_names = [name for name, _, _ in critique_items]
            self.renderer.start_work(critique_names, f"Round {round_num} critiques")
            import asyncio

            tasks = []
            provider_names = []
            for name, provider, msgs in critique_items:
                provider_names.append(name)
                model = self.config.model_overrides.get(name)
                tasks.append(self._get_response(provider, msgs, model, participant_id=name))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            self.renderer.stop_work()
            for name, r in zip(provider_names, results, strict=True):  # type: ignore[assignment]
                if isinstance(r, Exception):
                    self.renderer.show_error(name, _format_provider_error(r))
                    continue
                for other_name in latest:
                    if other_name != name:
                        critiques.append(
                            Critique(
                                author=name,
                                target=other_name,
                                content=r.content,
                            )
                        )
                responses.append(r)

        return critiques, responses

    def _build_critique_prompt(
        self,
        name: str,
        own_response: LLMResponse,
        others: dict[str, LLMResponse],
        round_num: int,
    ) -> list[Message]:
        system_msg = self._build_system()
        parts = [self._full_prompt_with_context()]
        parts.append(f"\n---\n\nYour previous answer:\n{own_response.content}\n")
        parts.append("\nOther models' answers:\n")
        for other_name, resp in others.items():
            parts.append(f"--- {other_name} ---\n{resp.content}\n")
        parts.append(CRITIQUE_INSTRUCTIONS)
        messages = []
        if system_msg:
            messages.append(system_msg)
        messages.append(Message(role="user", content="\n".join(parts)))
        return messages

    def _synthesis_provider_order(self) -> list[str]:
        """Return providers in preferred order for synthesis."""
        synth_name = self.config.synthesis_provider
        if synth_name and synth_name in self.providers:
            return [synth_name] + [n for n in self.providers if n != synth_name]

        ordered: list[str] = []
        for preferred in ("anthropic", "vertex", "openai"):
            for name in self.providers:
                base = provider_base_name(name)
                model = self.config.model_overrides.get(name, "")
                if base == preferred and name not in ordered:
                    # For vertex, prefer Claude models for synthesis
                    if base == "vertex" and model.startswith("gemini"):
                        continue
                    ordered.append(name)
                    break
        for name in self.providers:
            if name not in ordered:
                ordered.append(name)
        return ordered

    async def _synthesize(self, latest: dict[str, LLMResponse]) -> tuple[str, str]:
        parts = [self._full_prompt_with_context()]
        parts.append("\n---\n\nFinal answers from each model after debate:\n")
        for name, resp in latest.items():
            parts.append(f"--- {name} ({resp.model}) ---\n{resp.content}\n")
        parts.append(SYNTHESIS_PREAMBLE + EVALUATION_RULES)

        system_msg = self._build_system()
        messages: list[Message] = []
        if system_msg:
            messages.append(system_msg)
        messages.append(Message(role="user", content="\n".join(parts)))

        ordered = self._synthesis_provider_order()
        last_error: Exception | None = None

        for name in ordered:
            provider = self.providers[name]
            model = self.config.model_overrides.get(name)
            try:
                self.renderer.start_round(0)
                self.renderer.start_work([name], "Synthesizing")
                resp = await self._get_response(provider, messages, model, participant_id=name)
                self.renderer.stop_work()

                final, resolution = self._parse_final_answer(resp.content)
                return final, resolution
            except Exception as e:
                self.renderer.stop_work()
                last_error = e
                self.renderer.show_error(name, f"Synthesis failed: {_format_provider_error(e)}")

        raise click.ClickException(f"All providers failed during synthesis. Last error: {last_error}")

    def _has_converged(self, prev: dict[str, LLMResponse], curr: dict[str, LLMResponse]) -> bool:
        common = set(prev) & set(curr)
        if not common:
            return False
        threshold = self.config.convergence_threshold
        for name in common:
            ratio = difflib.SequenceMatcher(None, prev[name].content, curr[name].content).ratio()
            if ratio < threshold:
                return False
        return True
