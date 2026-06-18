"""Single-pass LLM sentiment analyst.

Sends a formatted sources block to Claude (via ClaudeClient) and parses
a structured JSON response into an AnalystResult.  Falls back to a
caller-supplied heuristic score if the LLM is unavailable or returns
an unparseable response.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

from sentiment.claude_client import ClaudeClient

_log = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are a financial analyst. Rate the short-term (24-48 h) sentiment for {symbol}.

{sources_text}

Respond ONLY with a single JSON object (no markdown, no explanation):
{{"score": <float -1.0 to 1.0>, "rationale": "<one sentence, max 100 chars>"}}

Where -1.0 = strongly bearish, 0.0 = neutral, +1.0 = strongly bullish.\
"""


@dataclass
class AnalystResult:
    """Output of a single-pass sentiment analysis run."""

    symbol: str
    score: float        # -1.0 (very bearish) … +1.0 (very bullish)
    rationale: str      # brief explanation (≤ 100 chars)
    analysed_at: int    # unix seconds
    llm_used: bool      # False when falling back to heuristic


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Matches a JSON object anywhere in the LLM response text.
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_llm_response(text: str) -> tuple[float, str] | None:
    """Extract (score, rationale) from LLM text; return None on failure."""
    match = _JSON_RE.search(text)
    if not match:
        _log.debug("analyst: no JSON found in LLM response")
        return None
    try:
        data = json.loads(match.group())
        score = float(data["score"])
        score = max(-1.0, min(1.0, score))
        rationale = str(data.get("rationale", ""))[:100]
        return score, rationale
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        _log.debug("analyst: failed to parse LLM JSON: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def heuristic_score(price_change_24h_pct: float | None) -> float:
    """Normalise a 24 h price change to a [-1, 1] sentiment proxy.

    Maps ±10 % → ±1.0, clamped at the extremes.
    """
    if price_change_24h_pct is None:
        return 0.0
    return max(-1.0, min(1.0, price_change_24h_pct / 10.0))


def analyse(
    symbol: str,
    sources_text: str,
    client: ClaudeClient,
    *,
    fallback_score: float = 0.0,
) -> AnalystResult:
    """Run a single-pass LLM sentiment analysis for *symbol*.

    Parameters
    ----------
    symbol:
        Asset identifier (e.g. ``"BTC/USDT"`` or ``"AAPL"``).
    sources_text:
        Pre-formatted context block produced by ``to_prompt_text()`` from
        either ``sources_crypto`` or ``sources_stocks``.
    client:
        Initialised ``ClaudeClient``.  Its ``query()`` method is called once.
    fallback_score:
        Score returned when the LLM is unavailable or response is
        unparseable.  Callers should pass ``heuristic_score(price_change)``
        rather than leaving this as 0.0 when price data is available.
    """
    prompt = _PROMPT_TEMPLATE.format(symbol=symbol, sources_text=sources_text)
    response = client.query(prompt)

    if response is not None:
        parsed = _parse_llm_response(response)
        if parsed is not None:
            score, rationale = parsed
            return AnalystResult(
                symbol=symbol,
                score=score,
                rationale=rationale,
                analysed_at=int(time.time()),
                llm_used=True,
            )
        _log.warning("analyst: unparseable LLM response for %s; using fallback", symbol)

    return AnalystResult(
        symbol=symbol,
        score=fallback_score,
        rationale="deterministic fallback (LLM unavailable)",
        analysed_at=int(time.time()),
        llm_used=False,
    )
