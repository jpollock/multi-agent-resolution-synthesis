"""Round-robin debate strategy with iterative critique."""

from __future__ import annotations

import asyncio
import difflib
from typing import TYPE_CHECKING

from mar.debate.base import DebateStrategy
from mar.models import (
    Critique,
    DebateResult,
    DebateRound,
    LLMResponse,
    Message,
    TokenUsage,
    Verbosity,
)
from mar.providers.base import retry_with_backoff

if TYPE_CHECKING:
    from mar.providers.base import LLMProvider


class RoundRobinStrategy(DebateStrategy):
    async def run(self) -> DebateResult:
        result = DebateResult(
            prompt=self.config.prompt,
            context=self.config.context,
            mode=self.config.mode,
        )

        self.writer.write_prompt(self.config.prompt, self.config.context)

        # Track latest answers per provider
        latest_answers: dict[str, LLMResponse] = {}

        for round_num in range(1, self.config.max_rounds + 1):
            self.renderer.start_round(round_num)
            debate_round = DebateRound(round_number=round_num)

            if round_num == 1:
                responses = await self._initial_round()
                debate_round.responses = responses
                self.writer.write_round(round_num, responses)
                for r in responses:
                    latest_answers[r.provider] = r
            else:
                critiques, responses = await self._critique_round(
                    round_num, latest_answers
                )
                debate_round.critiques = critiques
                debate_round.responses = responses
                self.writer.write_round(round_num, responses, critiques)

                # Check convergence
                prev = latest_answers
                new_answers = {r.provider: r for r in responses}
                if self._has_converged(prev, new_answers):
                    reason = (
                        f"Answers converged after round {round_num} "
                        f"(similarity threshold 0.9 reached)."
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

        # Final synthesis
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

        return await self._gather_responses(
            list(self.providers.items()), messages
        )

    async def _gather_responses(
        self,
        providers: list[tuple[str, LLMProvider]],
        messages: list[Message],
    ) -> list[LLMResponse]:
        """Run providers concurrently in quiet mode, sequentially in verbose."""
        responses: list[LLMResponse] = []

        if self.config.verbosity == Verbosity.VERBOSE:
            # Sequential to avoid interleaved streaming output
            for name, provider in providers:
                model = self.config.model_overrides.get(name)
                try:
                    resp = await self._get_response(provider, messages, model)
                    responses.append(resp)
                except Exception as e:
                    self.renderer.show_error(name, str(e))
        else:
            # Concurrent when not streaming
            tasks = []
            provider_names = []
            for name, provider in providers:
                provider_names.append(name)
                model = self.config.model_overrides.get(name)
                tasks.append(self._get_response(provider, messages, model))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for name, r in zip(provider_names, results):
                if isinstance(r, Exception):
                    self.renderer.show_error(name, str(r))
                    continue
                responses.append(r)

        return responses

    async def _critique_round(
        self, round_num: int, latest: dict[str, LLMResponse]
    ) -> tuple[list[Critique], list[LLMResponse]]:
        critiques: list[Critique] = []

        # Build per-provider critique prompts
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

        if self.config.verbosity == Verbosity.VERBOSE:
            for name, provider, msgs in critique_items:
                model = self.config.model_overrides.get(name)
                try:
                    r = await self._get_response(provider, msgs, model)
                    for other_name in latest:
                        if other_name != name:
                            critiques.append(
                                Critique(author=name, target=other_name, content=r.content)
                            )
                    responses.append(r)
                except Exception as e:
                    self.renderer.show_error(name, str(e))
        else:
            tasks = []
            provider_names = []
            for name, provider, msgs in critique_items:
                provider_names.append(name)
                model = self.config.model_overrides.get(name)
                tasks.append(self._get_response(provider, msgs, model))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for name, r in zip(provider_names, results):
                if isinstance(r, Exception):
                    self.renderer.show_error(name, str(r))
                    continue
                for other_name in latest:
                    if other_name != name:
                        critiques.append(
                            Critique(author=name, target=other_name, content=r.content)
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
        parts.append(
            "\nIMPORTANT: Re-read the original prompt and context above carefully. "
            "For each specific question or requirement in the original prompt, "
            "evaluate whether the other models addressed it adequately.\n\n"
            "1. Identify specific points where other answers are wrong, incomplete, "
            "or miss requirements from the original prompt.\n"
            "2. Identify what they got right that your answer missed.\n"
            "3. Call out where any answer (including yours) replaced concrete data "
            "from the original prompt with vague generalities.\n"
            "4. Provide your COMPLETE improved answer that addresses ALL "
            "requirements from the original prompt, incorporating valid points "
            "from others while correcting errors.\n\n"
            "When the prompt asks for examples, give CONCRETE examples using "
            "real data from the context - not generic placeholders. When it asks "
            "for code, prompts, or schemas, provide complete, usable output. "
            "Do not summarize or shorten - give a full, detailed answer."
        )
        messages = []
        if system_msg:
            messages.append(system_msg)
        messages.append(Message(role="user", content="\n".join(parts)))
        return messages

    def _full_prompt_with_context(self) -> str:
        """Build the complete original prompt including any context."""
        parts = []
        if self.config.context:
            parts.append("=== CONTEXT ===")
            for i, ctx in enumerate(self.config.context, 1):
                if len(self.config.context) > 1:
                    parts.append(f"\n--- Context {i} ---")
                parts.append(ctx)
            parts.append("\n=== END CONTEXT ===\n")
        parts.append(f"ORIGINAL PROMPT: {self.config.prompt}")
        return "\n".join(parts)

    def _build_system(self) -> Message | None:
        if not self.config.context:
            return None
        ctx = "\n\n---\n\n".join(self.config.context)
        return Message(
            role="system",
            content=(
                "You are participating in a structured debate. The user's prompt "
                "includes context that is essential to the task. Treat the context "
                "as primary source material - reference it directly, address its "
                "specifics, and ensure your answer covers every requirement stated "
                "in both the context and prompt.\n\n"
                f"CONTEXT:\n{ctx}"
            ),
        )

    async def _get_response(
        self, provider: LLMProvider, messages: list[Message], model: str | None
    ) -> LLMResponse:
        actual_model = model or provider.default_model
        usage = TokenUsage()

        max_tokens = self.config.max_tokens
        temperature = self.config.temperature

        if self.config.verbosity == Verbosity.VERBOSE:
            self.renderer.start_provider_stream(provider.name)
            content = ""
            async for chunk in provider.stream(messages, model=model, max_tokens=max_tokens, temperature=temperature):
                self.renderer.stream_chunk(chunk)
                content += chunk
            self.renderer.end_provider_stream()
            usage = provider.last_usage
        else:
            content, usage = await retry_with_backoff(
                provider.generate, messages, model=model, max_tokens=max_tokens, temperature=temperature
            )
            self.renderer.show_response(provider.name, content)

        return LLMResponse(
            provider=provider.name, model=actual_model, content=content, usage=usage
        )

    def _synthesis_provider_order(self) -> list[str]:
        """Return providers in preferred order for synthesis."""
        synth_name = self.config.synthesis_provider
        if synth_name and synth_name in self.providers:
            return [synth_name] + [n for n in self.providers if n != synth_name]

        ordered: list[str] = []
        for preferred in ("anthropic", "openai"):
            if preferred in self.providers:
                ordered.append(preferred)
        for name in self.providers:
            if name not in ordered:
                ordered.append(name)
        return ordered

    async def _synthesize(
        self, latest: dict[str, LLMResponse]
    ) -> tuple[str, str]:
        parts = [self._full_prompt_with_context()]
        parts.append("\n---\n\nFinal answers from each model after debate:\n")
        for name, resp in latest.items():
            parts.append(f"--- {name} ({resp.model}) ---\n{resp.content}\n")
        parts.append(
            "\nSynthesize the best possible answer from all models' responses. "
            "Re-read the original prompt and context above carefully.\n\n"
            "CRITICAL RULES:\n"
            "- Address EVERY numbered question or requirement in the original prompt.\n"
            "- When the prompt asks for examples, provide CONCRETE examples with "
            "real data, names, numbers, and specifics - not generic placeholders.\n"
            "- When the prompt or context mentions specific data (names, numbers, "
            "scores, versions), use that exact data in your answer.\n"
            "- When the prompt asks for code, prompts, schemas, or configs, "
            "provide complete, copy-pasteable output - not descriptions of what "
            "it would look like.\n"
            "- Prefer the most specific and detailed version of any point across "
            "the models. Never abstract a concrete example into a vague summary.\n"
            "- If models disagree, pick the version with the strongest reasoning "
            "and most specificity.\n\n"
            "Structure your response in two sections:\n\n"
            "## Resolution Analysis\n"
            "For each model, list which specific points you accepted and which "
            "you rejected, with reasoning tied to the original requirements.\n\n"
            "## Final Answer\n"
            "Provide the complete synthesized answer. Match the level of detail "
            "and specificity the original prompt demands."
        )

        system_msg = self._build_system()
        messages: list[Message] = []
        if system_msg:
            messages.append(system_msg)
        messages.append(Message(role="user", content="\n".join(parts)))

        # Try providers in order, falling back on failure
        ordered = self._synthesis_provider_order()
        last_error: Exception | None = None

        for name in ordered:
            provider = self.providers[name]
            model = self.config.model_overrides.get(name)
            try:
                self.renderer.start_round(0)
                resp = await self._get_response(provider, messages, model)

                content = resp.content
                if "## Final Answer" in content:
                    parts_split = content.split("## Final Answer", 1)
                    resolution = parts_split[0].strip()
                    final = parts_split[1].strip()
                else:
                    resolution = ""
                    final = content

                return final, resolution
            except Exception as e:
                last_error = e
                self.renderer.show_error(name, f"Synthesis failed: {e}")

        raise RuntimeError(
            f"All providers failed during synthesis. Last error: {last_error}"
        )

    @staticmethod
    def _has_converged(
        prev: dict[str, LLMResponse], curr: dict[str, LLMResponse]
    ) -> bool:
        common = set(prev) & set(curr)
        if not common:
            return False
        for name in common:
            ratio = difflib.SequenceMatcher(
                None, prev[name].content, curr[name].content
            ).ratio()
            if ratio < 0.9:
                return False
        return True
