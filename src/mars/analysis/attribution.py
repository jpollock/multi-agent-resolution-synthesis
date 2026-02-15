"""Post-debate attribution analysis using sentence-level similarity."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from mars.models import AttributionReport, DebateResult, ProviderAttribution, RoundDiff

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_MIN_SENTENCE_LEN = 20


def split_sentences(text: str) -> list[str]:
    raw = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in raw if len(s.strip()) >= _MIN_SENTENCE_LEN]


def _sentence_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _best_match(sentence: str, candidates: list[str]) -> tuple[int, float]:
    if not candidates:
        return -1, 0.0
    best_idx = -1
    best_score = 0.0
    for i, cand in enumerate(candidates):
        score = _sentence_similarity(sentence, cand)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx, best_score


@dataclass
class _ProviderText:
    provider: str
    model: str
    round_sentences: dict[int, list[str]] = field(default_factory=dict)


class AttributionAnalyzer:
    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def analyze(self, result: DebateResult) -> AttributionReport:
        provider_data = self._extract_provider_data(result)
        final_sentences = split_sentences(result.final_answer)

        attributions: list[ProviderAttribution] = []

        for name, data in provider_data.items():
            contribution_pct, contributed = self._contribution(final_sentences, data, provider_data)
            survival_rate, survived, initial_count = self._survival(final_sentences, data)
            influence_score, influence_details = self._influence(
                data, provider_data, list(provider_data.keys())
            )

            attributions.append(
                ProviderAttribution(
                    provider=name,
                    model=data.model,
                    contribution_pct=round(contribution_pct, 1),
                    contributed_sentences=contributed,
                    total_final_sentences=len(final_sentences),
                    survival_rate=round(survival_rate, 1),
                    survived_sentences=survived,
                    initial_sentences=initial_count,
                    influence_score=round(influence_score, 1),
                    influence_details={k: round(v, 1) for k, v in influence_details.items()},
                )
            )

        total_attributed = sum(a.contributed_sentences for a in attributions)
        novel = max(0, len(final_sentences) - total_attributed)
        novel_pct = (novel / len(final_sentences) * 100) if final_sentences else 0.0

        round_diffs = self._compute_round_diffs(provider_data)

        return AttributionReport(
            providers=attributions,
            similarity_threshold=self.threshold,
            sentence_count_final=len(final_sentences),
            novel_sentences=novel,
            novel_pct=round(novel_pct, 1),
            round_diffs=round_diffs,
        )

    def _extract_provider_data(self, result: DebateResult) -> dict[str, _ProviderText]:
        data: dict[str, _ProviderText] = {}
        for rnd in result.rounds:
            for resp in rnd.responses:
                if resp.provider not in data:
                    data[resp.provider] = _ProviderText(provider=resp.provider, model=resp.model)
                data[resp.provider].round_sentences[rnd.round_number] = split_sentences(
                    resp.content
                )
        return data

    def _contribution(
        self,
        final_sentences: list[str],
        provider: _ProviderText,
        all_providers: dict[str, _ProviderText],
    ) -> tuple[float, int]:
        if not final_sentences:
            return 0.0, 0

        attributed = 0
        for sent in final_sentences:
            best_provider = None
            best_score = 0.0
            for name, pdata in all_providers.items():
                all_sents = [s for sents in pdata.round_sentences.values() for s in sents]
                _, score = _best_match(sent, all_sents)
                if score > best_score:
                    best_score = score
                    best_provider = name
            if best_score >= self.threshold and best_provider == provider.provider:
                attributed += 1

        pct = (attributed / len(final_sentences)) * 100
        return pct, attributed

    def _survival(
        self,
        final_sentences: list[str],
        provider: _ProviderText,
    ) -> tuple[float, int, int]:
        round1 = provider.round_sentences.get(1, [])
        if not round1:
            return 0.0, 0, 0

        survived = 0
        for sent in round1:
            _, score = _best_match(sent, final_sentences)
            if score >= self.threshold:
                survived += 1

        pct = (survived / len(round1)) * 100
        return pct, survived, len(round1)

    def _influence(
        self,
        provider: _ProviderText,
        all_providers: dict[str, _ProviderText],
        provider_names: list[str],
    ) -> tuple[float, dict[str, float]]:
        other_names = [n for n in provider_names if n != provider.provider]
        if not other_names:
            return 0.0, {}

        rounds_sorted = sorted(provider.round_sentences.keys())
        details: dict[str, list[float]] = {n: [] for n in other_names}

        for rnd in rounds_sorted:
            next_rnd = rnd + 1
            src_sentences = provider.round_sentences.get(rnd, [])
            if not src_sentences:
                continue

            for other_name in other_names:
                other_data = all_providers.get(other_name)
                if other_data is None:
                    continue
                other_curr = other_data.round_sentences.get(rnd, [])
                other_next = other_data.round_sentences.get(next_rnd, [])
                if not other_next:
                    continue

                adopted = 0
                for sent in src_sentences:
                    _, score_next = _best_match(sent, other_next)
                    if score_next < self.threshold:
                        continue
                    _, score_curr = _best_match(sent, other_curr)
                    if score_curr >= self.threshold:
                        continue
                    adopted += 1

                adoption_rate = (adopted / len(src_sentences)) * 100
                details[other_name].append(adoption_rate)

        per_target: dict[str, float] = {}
        for name, rates in details.items():
            per_target[name] = sum(rates) / len(rates) if rates else 0.0

        overall = sum(per_target.values()) / len(per_target) if per_target else 0.0
        return overall, per_target

    def _compute_round_diffs(self, provider_data: dict[str, _ProviderText]) -> list[RoundDiff]:
        diffs: list[RoundDiff] = []
        for name, data in provider_data.items():
            rounds_sorted = sorted(data.round_sentences.keys())
            for i in range(len(rounds_sorted) - 1):
                r1 = rounds_sorted[i]
                r2 = rounds_sorted[i + 1]
                s1 = data.round_sentences[r1]
                s2 = data.round_sentences[r2]

                matched_s2: set[int] = set()
                unchanged = 0
                for sent in s1:
                    idx, score = _best_match(sent, s2)
                    if score >= self.threshold:
                        unchanged += 1
                        matched_s2.add(idx)

                removed = len(s1) - unchanged
                added = len(s2) - len(matched_s2)
                similarity = difflib.SequenceMatcher(None, "\n".join(s1), "\n".join(s2)).ratio()

                diffs.append(
                    RoundDiff(
                        provider=name,
                        from_round=r1,
                        to_round=r2,
                        similarity=round(similarity, 3),
                        sentences_added=added,
                        sentences_removed=removed,
                        sentences_unchanged=unchanged,
                    )
                )
        return diffs
