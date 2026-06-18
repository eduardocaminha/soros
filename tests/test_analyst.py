"""Tests for sentiment/analyst.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sentiment.analyst import (
    AnalystResult,
    _parse_llm_response,
    analyse,
    heuristic_score,
)


# ---------------------------------------------------------------------------
# heuristic_score
# ---------------------------------------------------------------------------

class TestHeuristicScore:
    def test_none_returns_zero(self):
        assert heuristic_score(None) == 0.0

    def test_positive_change_positive_score(self):
        assert heuristic_score(5.0) == pytest.approx(0.5)

    def test_negative_change_negative_score(self):
        assert heuristic_score(-5.0) == pytest.approx(-0.5)

    def test_clamped_at_positive_one(self):
        assert heuristic_score(20.0) == pytest.approx(1.0)

    def test_clamped_at_negative_one(self):
        assert heuristic_score(-20.0) == pytest.approx(-1.0)

    def test_zero_change_zero_score(self):
        assert heuristic_score(0.0) == pytest.approx(0.0)

    def test_10_pct_maps_to_one(self):
        assert heuristic_score(10.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _parse_llm_response
# ---------------------------------------------------------------------------

class TestParseLlmResponse:
    def test_parses_clean_json(self):
        text = '{"score": 0.7, "rationale": "Strong bullish momentum"}'
        result = _parse_llm_response(text)
        assert result is not None
        score, rationale = result
        assert score == pytest.approx(0.7)
        assert rationale == "Strong bullish momentum"

    def test_parses_json_embedded_in_text(self):
        text = 'Here is my analysis:\n{"score": -0.4, "rationale": "Bearish signals"}\nDone.'
        result = _parse_llm_response(text)
        assert result is not None
        score, rationale = result
        assert score == pytest.approx(-0.4)

    def test_clamps_score_above_one(self):
        text = '{"score": 2.5, "rationale": "Very bullish"}'
        result = _parse_llm_response(text)
        assert result is not None
        score, _ = result
        assert score == pytest.approx(1.0)

    def test_clamps_score_below_minus_one(self):
        text = '{"score": -3.0, "rationale": "Very bearish"}'
        result = _parse_llm_response(text)
        assert result is not None
        score, _ = result
        assert score == pytest.approx(-1.0)

    def test_truncates_long_rationale(self):
        long_rationale = "x" * 150
        text = f'{{"score": 0.1, "rationale": "{long_rationale}"}}'
        result = _parse_llm_response(text)
        assert result is not None
        _, rationale = result
        assert len(rationale) == 100

    def test_returns_none_when_no_json(self):
        assert _parse_llm_response("No JSON here at all.") is None

    def test_returns_none_on_missing_score_key(self):
        assert _parse_llm_response('{"rationale": "No score"}') is None

    def test_returns_none_on_invalid_json(self):
        assert _parse_llm_response("{invalid json}") is None

    def test_score_zero_is_valid(self):
        text = '{"score": 0.0, "rationale": "Neutral market"}'
        result = _parse_llm_response(text)
        assert result is not None
        score, _ = result
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# analyse
# ---------------------------------------------------------------------------

def _make_client(response: str | None) -> MagicMock:
    client = MagicMock()
    client.query.return_value = response
    return client


class TestAnalyse:
    def test_returns_analyst_result_on_success(self):
        client = _make_client('{"score": 0.6, "rationale": "Positive momentum"}')
        result = analyse("BTC/USDT", "some sources", client)
        assert isinstance(result, AnalystResult)
        assert result.symbol == "BTC/USDT"
        assert result.score == pytest.approx(0.6)
        assert result.rationale == "Positive momentum"
        assert result.llm_used is True
        assert result.analysed_at > 0

    def test_uses_fallback_when_client_returns_none(self):
        client = _make_client(None)
        result = analyse("AAPL", "some sources", client, fallback_score=0.3)
        assert result.score == pytest.approx(0.3)
        assert result.llm_used is False
        assert "fallback" in result.rationale

    def test_uses_fallback_when_response_unparseable(self):
        client = _make_client("Sorry, I cannot provide a score.")
        result = analyse("AAPL", "some sources", client, fallback_score=-0.1)
        assert result.score == pytest.approx(-0.1)
        assert result.llm_used is False

    def test_default_fallback_score_is_zero(self):
        client = _make_client(None)
        result = analyse("AAPL", "sources", client)
        assert result.score == pytest.approx(0.0)

    def test_prompt_contains_symbol_and_sources(self):
        client = _make_client('{"score": 0.0, "rationale": "neutral"}')
        analyse("ETH/USDT", "eth sources text", client)
        prompt = client.query.call_args[0][0]
        assert "ETH/USDT" in prompt
        assert "eth sources text" in prompt

    def test_score_clamped_to_minus_one(self):
        client = _make_client('{"score": -5.0, "rationale": "very bearish"}')
        result = analyse("BTC/USDT", "sources", client)
        assert result.score == pytest.approx(-1.0)
        assert result.llm_used is True

    def test_score_clamped_to_plus_one(self):
        client = _make_client('{"score": 5.0, "rationale": "very bullish"}')
        result = analyse("BTC/USDT", "sources", client)
        assert result.score == pytest.approx(1.0)
        assert result.llm_used is True
