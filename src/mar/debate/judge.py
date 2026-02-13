"""Judge debate strategy - one model evaluates all others."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from mar.debate.base import DebateStrategy
from mar.models import (
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

        # Step 1: All providers answer concurrently
        self.renderer.start_round(1)
        responses = await self._initial_round()
        debate_round = DebateRound(round_number=1, responses=responses)
        result.rounds.append(debate_round)
        self.writer.write_round(1, responses)

        # Step 2: Judge evaluates all responses
        self.renderer.start_round(2)
        judge_provider = self.providers[judge_name]
        judge_model = self.config.model_overrides.get(judge_name)

        judgment = await self._judge(judge_provider, judge_model, responses)

        # Parse judgment into resolution + final answer
        content = judgment.content
        if "## Final Answer" in content:
            parts = content.split("## Final Answer", 1)
            resolution = parts[0].strip()
            final = parts[1].strip()
        else:
            resolution = ""
            final = content

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

        responses: list[LLMResponse] = []

        if self.config.verbosity == Verbosity.VERBOSE:
            for name, provider in self.providers.items():
                model = self.config.model_overrides.get(name)
                try:
                    resp = await self._get_response(provider, messages, model)
                    responses.append(resp)
                except Exception as e:
                    self.renderer.show_error(name, str(e))
        else:
            tasks = []
            provider_names = []
            for name, provider in self.providers.items():
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
        parts.append(
            "\nYou are the judge. Re-read the original prompt and context above "
            "carefully. Evaluate each response against EVERY specific requirement "
            "in the original prompt.\n\n"
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

        return await self._get_response(judge, messages, model)

    def _full_prompt_with_context(self) -> str:
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
