"""Cost computation from token usage."""

from __future__ import annotations

from mars.models import CostReport, DebateResult, ProviderCost

# Pricing per 1M tokens: (input_cost, output_cost)
# Approximate as of early 2025. Ollama is local/free.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "o3": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    # Anthropic
    "claude-opus-4": (15.00, 75.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-3": (0.25, 1.25),
    # Google
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
}


def _lookup_price(model: str) -> tuple[float, float]:
    """Find pricing for a model, matching by prefix."""
    # Exact match first
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    # Prefix match (e.g. "claude-sonnet-4-20250514" matches "claude-sonnet-4")
    for key, price in MODEL_PRICING.items():
        if model.startswith(key):
            return price
    return 0.0, 0.0


def compute_costs(result: DebateResult) -> CostReport:
    """Sum token usage per provider across all rounds and compute costs."""
    # Accumulate tokens per (provider, model)
    totals: dict[str, dict[str, int | str]] = {}

    for rnd in result.rounds:
        for resp in rnd.responses:
            key = resp.provider
            if key not in totals:
                totals[key] = {
                    "model": resp.model,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            totals[key]["input_tokens"] += resp.usage.input_tokens  # type: ignore[operator]
            totals[key]["output_tokens"] += resp.usage.output_tokens  # type: ignore[operator]

    providers: list[ProviderCost] = []
    grand_input = 0
    grand_output = 0
    grand_cost = 0.0

    for provider_name, info in totals.items():
        model = str(info["model"])
        inp = int(info["input_tokens"])  # type: ignore[arg-type]
        out = int(info["output_tokens"])  # type: ignore[arg-type]
        inp_price, out_price = _lookup_price(model)
        inp_cost = (inp / 1_000_000) * inp_price
        out_cost = (out / 1_000_000) * out_price
        total = inp_cost + out_cost

        providers.append(
            ProviderCost(
                provider=provider_name,
                model=model,
                input_tokens=inp,
                output_tokens=out,
                total_tokens=inp + out,
                input_cost=round(inp_cost, 6),
                output_cost=round(out_cost, 6),
                total_cost=round(total, 6),
            )
        )
        grand_input += inp
        grand_output += out
        grand_cost += total

    # Compute share of total
    for pc in providers:
        pc.share_of_total = round((pc.total_cost / grand_cost) * 100, 1) if grand_cost > 0 else 0.0

    return CostReport(
        providers=providers,
        total_input_tokens=grand_input,
        total_output_tokens=grand_output,
        total_cost=round(grand_cost, 6),
    )
