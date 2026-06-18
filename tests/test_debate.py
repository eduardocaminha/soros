"""Tests for sentiment/debate.py."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from sentiment.analyst import AnalystResult
from sentiment.debate import (
    LOW_CONVICTION_THRESHOLD,
    DebateResult,
    _parse_response,
    debate,
    should_debate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _analyst(score: float, rationale: str = "some rationale", symbol: str = "BTC/USDT") -> AnalystResult:
    return AnalystResult(
        symbol=symbol,
        score=score,
        rationale=rationale,
        analysed_at=int(time.time()),
        llm_used=True,
    )


def _client(response: str | None) -> MagicMock:
    m = MagicMock()
    m.query.return_value = response
    return m


# ---------------------------------------------------------------------------
# should_debate
# ---------------------------------------------------------------------------

class TestShouldDebate:
    def test_opposite_signs_triggers_debate(self):
        assert should_debate(0.5, -0.5) is True

    def test_opposite_signs_small_values_triggers(self):
        assert should_debate(0.3, -0.1) is True

    def test_same_sign_high_conviction_no_debate(self):
        assert should_debate(0.8, 0.6) is False

    def test_same_sign_no_debate(self):
        assert should_debate(-0.7, -0.4) is False

    def test_low_conviction_analyst_triggers_debate(self):
        assert should_debate(LOW_CONVICTION_THRESHOLD - 0.01, 0.5) is True

    def test_exactly_at_threshold_no_debate(self):
        assert should_debate(LOW_CONVICTION_THRESHOLD, 0.5) is False

    def test_zero_analyst_score_triggers_debate(self):
        assert should_debate(0.0, 0.5) is True

    def test_zero_det_score_no_opposite_signs(self):
        # 0.5 * 0.0 = 0, not < 0, so only low_conviction triggers
        assert should_debate(0.5, 0.0) is False

    def test_both_zero_triggers_low_conviction(self):
        assert should_debate(0.0, 0.0) is True


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_parses_clean_json(self):
        result = _parse_response('{"score": 0.4, "rationale": "Balanced view"}')
        assert result is not None
        score, rationale = result
        assert score == pytest.approx(0.4)
        assert rationale == "Balanced view"

    def test_clamps_above_one(self):
        result = _parse_response('{"score": 3.0, "rationale": "very bullish"}')
        assert result is not None
        assert result[0] == pytest.approx(1.0)

    def test_clamps_below_minus_one(self):
        result = _parse_response('{"score": -2.5, "rationale": "very bearish"}')
        assert result is not None
        assert result[0] == pytest.approx(-1.0)

    def test_truncates_long_rationale(self):
        result = _parse_response(f'{{"score": 0.1, "rationale": "{"x" * 150}"}}')
        assert result is not None
        assert len(result[1]) == 100

    def test_returns_none_on_no_json(self):
        assert _parse_response("no json here") is None

    def test_returns_none_on_missing_score(self):
        assert _parse_response('{"rationale": "no score"}') is None

    def test_parses_embedded_json(self):
        result = _parse_response('Verdict:\n{"score": -0.3, "rationale": "Lean bearish"}\n')
        assert result is not None
        assert result[0] == pytest.approx(-0.3)


# ---------------------------------------------------------------------------
# debate
# ---------------------------------------------------------------------------

class TestDebate:
    def test_no_debate_when_aligned_returns_analyst_score(self):
        a = _analyst(0.7)
        result = debate("sources", a, 0.5, _client("should not be called"))
        assert result.debated is False
        assert result.score == pytest.approx(0.7)
        assert result.rationale == a.rationale

    def test_no_debate_client_not_queried(self):
        a = _analyst(0.8)
        client = _client("should not be called")
        debate("sources", a, 0.6, client)
        client.query.assert_not_called()

    def test_debate_triggered_on_opposite_signs(self):
        a = _analyst(0.6)
        client = _client('{"score": 0.1, "rationale": "Mixed signals"}')
        result = debate("sources", a, -0.4, client)
        assert result.debated is True
        assert result.score == pytest.approx(0.1)
        assert result.rationale == "Mixed signals"
        client.query.assert_called_once()

    def test_debate_triggered_on_low_conviction(self):
        a = _analyst(0.1)  # below threshold
        client = _client('{"score": 0.4, "rationale": "Slight bullish edge"}')
        result = debate("sources", a, 0.5, client)
        assert result.debated is True

    def test_fallback_to_analyst_when_client_returns_none(self):
        a = _analyst(0.5)
        result = debate("sources", a, -0.3, _client(None))
        assert result.debated is False
        assert result.score == pytest.approx(a.score)
        assert result.rationale == a.rationale

    def test_fallback_when_response_unparseable(self):
        a = _analyst(0.5)
        result = debate("sources", a, -0.3, _client("Sorry, cannot parse"))
        assert result.debated is False
        assert result.score == pytest.approx(a.score)

    def test_result_has_correct_symbol(self):
        a = _analyst(0.6, symbol="ETH/USDT")
        client = _client('{"score": 0.2, "rationale": "ok"}')
        result = debate("sources", a, -0.5, client)
        assert result.symbol == "ETH/USDT"

    def test_result_is_debate_result_instance(self):
        a = _analyst(0.8)
        result = debate("sources", a, 0.6, _client("unused"))
        assert isinstance(result, DebateResult)

    def test_debated_at_is_recent(self):
        a = _analyst(0.8)
        before = int(time.time())
        result = debate("sources", a, 0.6, _client("unused"))
        after = int(time.time())
        assert before <= result.debated_at <= after

    def test_prompt_includes_symbol_and_sources(self):
        a = _analyst(0.5, symbol="SOL/USDT")
        client = _client('{"score": 0.0, "rationale": "neutral"}')
        debate("my sources text", a, -0.4, client)
        prompt = client.query.call_args[0][0]
        assert "SOL/USDT" in prompt
        assert "my sources text" in prompt
