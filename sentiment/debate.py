"""Bull vs bear debate — runs only when signals genuinely diverge.

The LLM quota is saved by calling Claude ONLY on genuine divergence: when the
(keyless, pre-scored) sentiment and the deterministic signal point opposite
ways. A low or neutral sentiment is not a conflict and never triggers the
debate on its own — doing so used to fire an LLM call for nearly every coin
every cycle and burn through the Claude subscription quota.

When triggered, a single structured prompt asks Claude to weigh both sides
and return a final verdict.  If the client is unavailable or rate-limited,
the analyst result is returned unchanged.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

from sentiment.analyst import AnalystResult
from sentiment.claude_client import ClaudeClient

_log = logging.getLogger(__name__)

_DEBATE_TEMPLATE = """\
You are a senior portfolio risk manager arbitrating between two analysts on \
{symbol} for the next 24-48 hours.

SENTIMENT ANALYST says (score {analyst_score:+.2f}): {analyst_rationale}

QUANT MODEL says (deterministic signal score {det_score:+.2f}): \
momentum/volatility indicators are pointing {det_direction}.

{sources_text}

Weigh both views. Respond ONLY with a single JSON object (no markdown, no explanation):
{{"score": <float -1.0 to 1.0>, "rationale": "<one sentence, max 100 chars>"}}

Where -1.0 = strongly bearish, 0.0 = neutral, +1.0 = strongly bullish.\
"""

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


@dataclass
class DebateResult:
    """Output after optional bull/bear arbitration."""

    symbol: str
    score: float        # -1.0 … +1.0
    rationale: str      # brief explanation (≤ 100 chars)
    debated: bool       # True when the debate LLM call was made
    debated_at: int     # unix seconds


def should_debate(analyst_score: float, deterministic_score: float) -> bool:
    """Return True only on genuine divergence: sentiment and the deterministic
    signal point opposite ways (one positive, one negative).

    A low/neutral sentiment is NOT a conflict and does not trigger the debate
    on its own — that previously fired an LLM call almost every cycle and
    exhausted the Claude subscription quota.
    """
    return (analyst_score * deterministic_score) < 0


def _parse_response(text: str) -> tuple[float, str] | None:
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        score = float(data["score"])
        score = max(-1.0, min(1.0, score))
        rationale = str(data.get("rationale", ""))[:100]
        return score, rationale
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        _log.debug("debate: failed to parse LLM JSON: %s", exc)
        return None


def debate(
    sources_text: str,
    analyst_result: AnalystResult,
    deterministic_score: float,
    client: ClaudeClient,
) -> DebateResult:
    """Run bull/bear debate if signals diverge; otherwise promote analyst result.

    Parameters
    ----------
    sources_text:
        Pre-formatted context block (same one passed to analyst.analyse).
    analyst_result:
        Output of sentiment.analyst.analyse().
    deterministic_score:
        Aggregated quant signal in [-1, 1] for the same symbol.
    client:
        Initialised ClaudeClient.
    """
    symbol = analyst_result.symbol

    if not should_debate(analyst_result.score, deterministic_score):
        _log.debug("debate: no divergence for %s — skipping debate", symbol)
        return DebateResult(
            symbol=symbol,
            score=analyst_result.score,
            rationale=analyst_result.rationale,
            debated=False,
            debated_at=int(time.time()),
        )

    _log.info(
        "debate: divergence detected for %s (analyst=%.2f det=%.2f) — running debate",
        symbol, analyst_result.score, deterministic_score,
    )

    prompt = _DEBATE_TEMPLATE.format(
        symbol=symbol,
        analyst_score=analyst_result.score,
        analyst_rationale=analyst_result.rationale,
        det_score=deterministic_score,
        det_direction="bearish" if deterministic_score < 0 else "bullish",
        sources_text=sources_text,
    )

    response = client.query(prompt)

    if response is not None:
        parsed = _parse_response(response)
        if parsed is not None:
            score, rationale = parsed
            return DebateResult(
                symbol=symbol,
                score=score,
                rationale=rationale,
                debated=True,
                debated_at=int(time.time()),
            )
        _log.warning("debate: unparseable response for %s; using analyst score", symbol)

    return DebateResult(
        symbol=symbol,
        score=analyst_result.score,
        rationale=analyst_result.rationale,
        debated=False,
        debated_at=int(time.time()),
    )
