"""Pydantic data models for MAR."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class DebateMode(str, Enum):
    ROUND_ROBIN = "round-robin"
    JUDGE = "judge"


class Verbosity(str, Enum):
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
    verbosity: Verbosity = Verbosity.QUIET
    max_tokens: int = 8192
    temperature: float | None = None
    output_dir: str = "./mar-output"


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


class AttributionReport(BaseModel):
    providers: list[ProviderAttribution] = Field(default_factory=list)
    similarity_threshold: float = 0.6
    sentence_count_final: int = 0


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
