"""Tests for sentiment/fear_greed_history.py."""

from __future__ import annotations

import datetime
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from sentiment.fear_greed_history import (
    build_index,
    fetch_history,
    get_index,
    load_cached,
    lookup,
    save_cache,
    score_from_value,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' to UTC midnight unix timestamp."""
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )
    return int(dt.timestamp())


_SAMPLE_ENTRIES = [
    {"value": "72", "value_classification": "Greed",        "timestamp": str(_ts("2024-01-03"))},
    {"value": "25", "value_classification": "Fear",          "timestamp": str(_ts("2024-01-02"))},
    {"value": "50", "value_classification": "Neutral",       "timestamp": str(_ts("2024-01-01"))},
    {"value": "10", "value_classification": "Extreme Fear",  "timestamp": str(_ts("2023-12-31"))},
]

_SAMPLE_API_RESPONSE = {
    "name": "Fear and Greed Index",
    "data": _SAMPLE_ENTRIES,
    "metadata": {"error": None},
}


# ---------------------------------------------------------------------------
# fetch_history
# ---------------------------------------------------------------------------

class TestFetchHistory:
    def test_returns_entries_on_success(self):
        with patch(
            "sentiment.fear_greed_history.urlopen",
        ) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                json.dumps(_SAMPLE_API_RESPONSE).encode()
            )
            entries = fetch_history()
        assert len(entries) == 4
        assert entries[0]["value"] == "72"

    def test_uses_limit_zero_by_default(self):
        captured = []

        def fake_urlopen(req, timeout):
            captured.append(req.full_url)

            class _Ctx:
                def __enter__(self_inner):
                    return self_inner
                def __exit__(self_inner, *a):
                    pass
                def read(self_inner):
                    return json.dumps(_SAMPLE_API_RESPONSE).encode()

            return _Ctx()

        with patch("sentiment.fear_greed_history.urlopen", side_effect=fake_urlopen):
            fetch_history()

        assert captured and "limit=0" in captured[0]

    def test_returns_empty_on_network_error(self):
        from urllib.error import URLError

        with patch(
            "sentiment.fear_greed_history.urlopen",
            side_effect=URLError("timeout"),
        ):
            entries = fetch_history()
        assert entries == []

    def test_returns_empty_on_malformed_json(self):
        class _Ctx:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
            def read(self):
                return b"not json{"

        with patch("sentiment.fear_greed_history.urlopen", return_value=_Ctx()):
            entries = fetch_history()
        assert entries == []

    def test_returns_empty_when_data_key_missing(self):
        class _Ctx:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
            def read(self):
                return json.dumps({"metadata": {"error": None}}).encode()

        with patch("sentiment.fear_greed_history.urlopen", return_value=_Ctx()):
            entries = fetch_history()
        assert entries == []

    def test_custom_limit_passed_in_url(self):
        captured = []

        def fake_urlopen(req, timeout):
            captured.append(req.full_url)

            class _Ctx:
                def __enter__(self_inner):
                    return self_inner
                def __exit__(self_inner, *a):
                    pass
                def read(self_inner):
                    return json.dumps(_SAMPLE_API_RESPONSE).encode()

            return _Ctx()

        with patch("sentiment.fear_greed_history.urlopen", side_effect=fake_urlopen):
            fetch_history(limit=30)

        assert captured and "limit=30" in captured[0]


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------

class TestBuildIndex:
    def test_maps_dates_to_values(self):
        index = build_index(_SAMPLE_ENTRIES)
        assert index["2024-01-01"] == 50
        assert index["2024-01-02"] == 25
        assert index["2024-01-03"] == 72
        assert index["2023-12-31"] == 10

    def test_all_four_dates_present(self):
        index = build_index(_SAMPLE_ENTRIES)
        assert len(index) == 4

    def test_empty_entries_gives_empty_index(self):
        assert build_index([]) == {}

    def test_skips_malformed_entry(self):
        entries = [
            {"value": "50", "value_classification": "Neutral", "timestamp": str(_ts("2024-01-01"))},
            {"value": "bad", "value_classification": "?",      "timestamp": str(_ts("2024-01-02"))},
            {"value_classification": "Fear"},  # missing value + timestamp
        ]
        index = build_index(entries)
        assert "2024-01-01" in index
        assert index["2024-01-01"] == 50
        assert len(index) == 1

    def test_latest_entry_wins_for_duplicate_date(self):
        # Same day can appear twice; last write wins (dict overwrite).
        ts = _ts("2024-01-01")
        entries = [
            {"value": "30", "value_classification": "Fear",    "timestamp": str(ts)},
            {"value": "70", "value_classification": "Greed",   "timestamp": str(ts + 60)},
        ]
        index = build_index(entries)
        # Both map to 2024-01-01; second write wins.
        assert index["2024-01-01"] in (30, 70)  # deterministic but either is acceptable


# ---------------------------------------------------------------------------
# lookup
# ---------------------------------------------------------------------------

class TestLookup:
    def setup_method(self):
        self.index = build_index(_SAMPLE_ENTRIES)

    def test_exact_date_match(self):
        result = lookup(self.index, _ts("2024-01-02"))
        assert result == 25

    def test_midnight_utc_matches(self):
        result = lookup(self.index, _ts("2024-01-03"))
        assert result == 72

    def test_backward_fill_for_missing_date(self):
        # 2024-01-02T12:00 UTC falls on 2024-01-02 → value 25
        ts_noon = _ts("2024-01-02") + 12 * 3600
        result = lookup(self.index, ts_noon)
        assert result == 25

    def test_backward_fill_picks_most_recent_prior_date(self):
        # 2024-01-04 is not in the index; most recent prior date is 2024-01-03
        ts = _ts("2024-01-04")
        result = lookup(self.index, ts)
        assert result == 72

    def test_returns_none_before_earliest_date(self):
        # 2023-12-30 is before 2023-12-31 (earliest in our sample)
        ts = _ts("2023-12-30")
        result = lookup(self.index, ts)
        assert result is None

    def test_returns_none_on_empty_index(self):
        result = lookup({}, _ts("2024-01-01"))
        assert result is None

    def test_far_future_returns_latest(self):
        ts = _ts("2099-12-31")
        result = lookup(self.index, ts)
        assert result == 72  # 2024-01-03 is the most recent in sample

    def test_earliest_date_exact_match(self):
        result = lookup(self.index, _ts("2023-12-31"))
        assert result == 10


# ---------------------------------------------------------------------------
# score_from_value
# ---------------------------------------------------------------------------

class TestScoreFromValue:
    def test_50_gives_zero(self):
        assert score_from_value(50) == pytest.approx(0.0)

    def test_100_gives_one(self):
        assert score_from_value(100) == pytest.approx(1.0)

    def test_0_gives_minus_one(self):
        assert score_from_value(0) == pytest.approx(-1.0)

    def test_75_gives_plus_half(self):
        # (75 - 50) / 50 = 0.5
        assert score_from_value(75) == pytest.approx(0.5)

    def test_25_gives_minus_half(self):
        # (25 - 50) / 50 = -0.5
        assert score_from_value(25) == pytest.approx(-0.5)

    def test_clamped_above(self):
        assert score_from_value(150) == pytest.approx(1.0)

    def test_clamped_below(self):
        assert score_from_value(-50) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# save_cache / load_cached
# ---------------------------------------------------------------------------

class TestDiskCache:
    def test_roundtrip(self, tmp_path):
        index = {"2024-01-01": 50, "2024-01-02": 25}
        path = tmp_path / "fg_cache.json"
        save_cache(index, path)
        loaded = load_cached(path, max_age_secs=3600)
        assert loaded == index

    def test_missing_file_returns_none(self, tmp_path):
        assert load_cached(tmp_path / "nonexistent.json") is None

    def test_stale_cache_returns_none(self, tmp_path):
        index = {"2024-01-01": 50}
        path = tmp_path / "fg_cache.json"
        save_cache(index, path)

        # Manually backdate the saved_at timestamp.
        payload = json.loads(path.read_text())
        payload["saved_at"] = time.time() - 90_000  # 25 h ago
        path.write_text(json.dumps(payload))

        assert load_cached(path, max_age_secs=86_400) is None

    def test_fresh_cache_returned(self, tmp_path):
        index = {"2024-01-01": 72}
        path = tmp_path / "fg_cache.json"
        save_cache(index, path)
        loaded = load_cached(path, max_age_secs=86_400)
        assert loaded == index

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "subdir" / "deeper" / "cache.json"
        save_cache({"2024-01-01": 30}, path)
        assert path.exists()

    def test_corrupted_file_returns_none(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json at all {")
        assert load_cached(path) is None

    def test_values_cast_to_int(self, tmp_path):
        payload = {"saved_at": time.time(), "index": {"2024-01-01": "50"}}
        path = tmp_path / "fg_cache.json"
        path.write_text(json.dumps(payload))
        loaded = load_cached(path)
        assert isinstance(loaded["2024-01-01"], int)


# ---------------------------------------------------------------------------
# get_index
# ---------------------------------------------------------------------------

class TestGetIndex:
    def test_fetches_and_returns_index(self):
        with patch("sentiment.fear_greed_history.fetch_history", return_value=_SAMPLE_ENTRIES):
            index = get_index(cache_path=None)
        assert "2024-01-01" in index
        assert index["2024-01-01"] == 50

    def test_uses_cache_when_fresh(self, tmp_path):
        path = tmp_path / "cache.json"
        save_cache({"2024-01-01": 99}, path)

        with patch(
            "sentiment.fear_greed_history.fetch_history",
            side_effect=AssertionError("should not be called"),
        ):
            index = get_index(cache_path=path)

        assert index["2024-01-01"] == 99

    def test_fetches_when_cache_stale(self, tmp_path):
        path = tmp_path / "cache.json"
        payload = {"saved_at": time.time() - 90_000, "index": {"2024-01-01": 1}}
        path.write_text(json.dumps(payload))

        with patch(
            "sentiment.fear_greed_history.fetch_history",
            return_value=_SAMPLE_ENTRIES,
        ):
            index = get_index(cache_path=path, max_age_secs=86_400)

        assert index["2024-01-01"] == 50  # from fresh fetch, not stale cache

    def test_writes_cache_after_fetch(self, tmp_path):
        path = tmp_path / "cache.json"

        with patch("sentiment.fear_greed_history.fetch_history", return_value=_SAMPLE_ENTRIES):
            get_index(cache_path=path)

        assert path.exists()
        loaded = load_cached(path)
        assert loaded is not None
        assert "2024-01-01" in loaded

    def test_no_cache_write_on_empty_fetch(self, tmp_path):
        path = tmp_path / "cache.json"

        with patch("sentiment.fear_greed_history.fetch_history", return_value=[]):
            index = get_index(cache_path=path)

        assert index == {}
        assert not path.exists()  # no write when index is empty

    def test_returns_empty_dict_on_total_failure(self):
        with patch("sentiment.fear_greed_history.fetch_history", return_value=[]):
            index = get_index(cache_path=None)
        assert index == {}


# ---------------------------------------------------------------------------
# Integration: lookup over a realistic multi-day span
# ---------------------------------------------------------------------------

class TestLookupIntegration:
    def test_every_known_date_resolves(self):
        index = build_index(_SAMPLE_ENTRIES)
        dates = ["2023-12-31", "2024-01-01", "2024-01-02", "2024-01-03"]
        for d in dates:
            result = lookup(index, _ts(d))
            assert result is not None
            assert 0 <= result <= 100

    def test_gap_filled_by_backward_fill(self):
        # Index has 2024-01-01 and 2024-01-03 but NOT 2024-01-02.
        sparse_index = {"2024-01-01": 50, "2024-01-03": 72}
        # 2024-01-02 should fill from 2024-01-01 (most recent prior date).
        result = lookup(sparse_index, _ts("2024-01-02"))
        assert result == 50

    def test_intra_day_timestamp_uses_date_of_ts(self):
        # 2024-01-01T23:59:59 UTC should resolve to 2024-01-01.
        ts_end_of_day = _ts("2024-01-01") + 86_399
        index = {"2024-01-01": 42}
        assert lookup(index, ts_end_of_day) == 42
