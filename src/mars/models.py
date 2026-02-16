"""Pydantic data models for MARS."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


def provider_base_name(provider: str) -> str:
    """Extract base provider name from a participant ID.

    'vertex:claude-sonnet-4' -> 'vertex'
    'openai' -> 'openai'
    """
    return provider.split(":")[0]


class DebateMode(StrEnum):
    ROUND_ROBIN = "round-robin"
    JUDGE = "judge"


class Verbosity(StrEnum):
    VERBOSE = "verbose"
    QUIET = "quiet"


class Message(BaseModel):
    role: str
    content: str


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class LLMResponse(BaseModel):
    provider: str
    model: str
    content: str
    usage: TokenUsage = Field(default_factory=TokenUsage)


class Critique(BaseModel):
    author: str
    target: str
    content: str


class DebateRound(BaseModel):
    round_number: int
    responses: list[LLMResponse] = Field(default_factory=list)
    critiques: list[Critique] = Field(default_factory=list)


class DebateResult(BaseModel):
    prompt: str
    context: list[str] = Field(default_factory=list)
    mode: DebateMode
    rounds: list[DebateRound] = Field(default_factory=list)
    final_answer: str = ""
    convergence_reason: str = ""
    resolution_reasoning: str = ""


class DebateConfig(BaseModel):
    prompt: str
    context: list[str] = Field(default_factory=list)
    providers: list[str] = Field(default_factory=list)
    model_overrides: dict[str, str] = Field(default_factory=dict)
    mode: DebateMode = DebateMode.ROUND_ROBIN
    max_rounds: int = 3
    judge_provider: str | None = None
    synthesis_provider: str | None = None
    convergence_threshold: float = 0.85
    verbosity: Verbosity = Verbosity.QUIET
    max_tokens: int = 8192
    temperature: float | None = None
    output_dir: str = "./mars-output"

    @field_validator("max_rounds")
    @classmethod
    def max_rounds_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_rounds must be at least 1")
        return v

    @field_validator("convergence_threshold")
    @classmethod
    def threshold_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("convergence_threshold must be between 0.0 and 1.0")
        return v

    @field_validator("providers")
    @classmethod
    def providers_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one provider is required")
        return v


# --- Attribution & Cost models ---


class ProviderAttribution(BaseModel):
    provider: str
    model: str
    contribution_pct: float = 0.0
    contributed_sentences: int = 0
    total_final_sentences: int = 0
    survival_rate: float = 0.0
    survived_sentences: int = 0
    initial_sentences: int = 0
    influence_score: float = 0.0
    influence_details: dict[str, float] = Field(default_factory=dict)


class RoundDiff(BaseModel):
    provider: str
    from_round: int
    to_round: int
    similarity: float = 0.0
    sentences_added: int = 0
    sentences_removed: int = 0
    sentences_unchanged: int = 0


class AttributionReport(BaseModel):
    providers: list[ProviderAttribution] = Field(default_factory=list)
    similarity_threshold: float = 0.6
    sentence_count_final: int = 0
    novel_sentences: int = 0
    novel_pct: float = 0.0
    round_diffs: list[RoundDiff] = Field(default_factory=list)


class ProviderCost(BaseModel):
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    share_of_total: float = 0.0


class CostReport(BaseModel):
    providers: list[ProviderCost] = Field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
