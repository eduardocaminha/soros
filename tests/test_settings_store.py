"""Tests for database/settings_store.py — precedence, allowlist, validation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import config
from database.settings_store import (
    InvalidValueError,
    LockedKeyError,
    delete_override,
    get_all_overrides,
    get_override,
    resolve,
    set_override,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def temp_db(tmp_path: Path) -> str:
    db_file = str(tmp_path / "test.db")
    schema = (Path(__file__).parent.parent / "database" / "schema.sql").read_text()
    conn = sqlite3.connect(db_file)
    conn.executescript(schema)
    conn.commit()
    conn.close()
    return db_file


@pytest.fixture(autouse=True)
def _patch_db(temp_db: str, monkeypatch):
    import database.db as db_module

    class _FakeDB:
        def connect(self):
            c = sqlite3.connect(temp_db)
            c.row_factory = sqlite3.Row
            return c

    monkeypatch.setattr(db_module, "_db", _FakeDB())


# ---------------------------------------------------------------------------
# ALLOWLIST membership
# ---------------------------------------------------------------------------

class TestAllowlist:
    def test_execution_toggles_not_in_allowlist(self):
        for key in ("CRYPTO_LIVE", "STOCKS_LIVE", "SENTIMENT_ENABLED"):
            assert key not in config.SETTINGS_ALLOWLIST, f"{key} must not be editable"

    def test_hard_risk_limits_not_in_allowlist(self):
        for key in ("MAX_DRAWDOWN_PCT", "MAX_OPEN_POSITIONS"):
            assert key not in config.SETTINGS_ALLOWLIST, f"{key} must not be editable"

    def test_tunable_keys_present(self):
        expected = {
            "LOOP_INTERVAL_SECONDS",
            "SIGNAL_THRESHOLD",
            "SCREENER_TOP_N",
            "GEM_TOP_N",
            "POSITION_SIZE_PCT",
            "GEM_TRAILING_STOP_PCT",
            "IGNITION_WEIGHT",
        }
        for key in expected:
            assert key in config.SETTINGS_ALLOWLIST, f"{key} should be editable"

    def test_locked_set_contains_execution_and_risk(self):
        for key in ("CRYPTO_LIVE", "STOCKS_LIVE", "SENTIMENT_ENABLED",
                    "MAX_DRAWDOWN_PCT", "MAX_OPEN_POSITIONS"):
            assert key in config.SETTINGS_LOCKED


# ---------------------------------------------------------------------------
# set_override — allowlist enforcement
# ---------------------------------------------------------------------------

class TestSetOverrideAllowlist:
    def test_locked_key_raises_locked_key_error(self):
        with pytest.raises(LockedKeyError):
            set_override("CRYPTO_LIVE", "true")

    def test_locked_key_stocks_live_raises(self):
        with pytest.raises(LockedKeyError):
            set_override("STOCKS_LIVE", "true")

    def test_locked_key_sentiment_enabled_raises(self):
        with pytest.raises(LockedKeyError):
            set_override("SENTIMENT_ENABLED", "true")

    def test_locked_key_max_drawdown_raises(self):
        with pytest.raises(LockedKeyError):
            set_override("MAX_DRAWDOWN_PCT", "0.5")

    def test_locked_key_max_open_positions_raises(self):
        with pytest.raises(LockedKeyError):
            set_override("MAX_OPEN_POSITIONS", "10")

    def test_unknown_key_raises_locked_key_error(self):
        with pytest.raises(LockedKeyError):
            set_override("TOTALLY_UNKNOWN_KEY", "value")

    def test_allowlisted_key_succeeds(self):
        set_override("SIGNAL_THRESHOLD", "0.3")
        assert get_override("SIGNAL_THRESHOLD") == "0.3"


# ---------------------------------------------------------------------------
# set_override — value validation
# ---------------------------------------------------------------------------

class TestSetOverrideValidation:
    def test_invalid_type_raises(self):
        with pytest.raises(InvalidValueError):
            set_override("SIGNAL_THRESHOLD", "not_a_float")

    def test_below_minimum_raises(self):
        with pytest.raises(InvalidValueError):
            set_override("LOOP_INTERVAL_SECONDS", "30")  # min=60

    def test_above_maximum_raises(self):
        with pytest.raises(InvalidValueError):
            set_override("SIGNAL_THRESHOLD", "1.5")  # max=1.0

    def test_at_minimum_accepted(self):
        set_override("LOOP_INTERVAL_SECONDS", "60")
        assert get_override("LOOP_INTERVAL_SECONDS") == "60"

    def test_at_maximum_accepted(self):
        set_override("SIGNAL_THRESHOLD", "1.0")
        assert get_override("SIGNAL_THRESHOLD") == "1.0"

    def test_bool_true_accepted(self):
        set_override("SCREENER_ENABLED", "true")
        assert get_override("SCREENER_ENABLED") == "true"

    def test_bool_false_accepted(self):
        set_override("SCREENER_ENABLED", "false")
        assert get_override("SCREENER_ENABLED") == "false"

    def test_bool_invalid_raises(self):
        with pytest.raises(InvalidValueError):
            set_override("SCREENER_ENABLED", "yes")

    def test_upsert_updates_existing(self):
        set_override("SIGNAL_THRESHOLD", "0.3")
        set_override("SIGNAL_THRESHOLD", "0.4")
        assert get_override("SIGNAL_THRESHOLD") == "0.4"


# ---------------------------------------------------------------------------
# get_override / delete_override
# ---------------------------------------------------------------------------

class TestGetDeleteOverride:
    def test_get_returns_none_when_not_set(self):
        assert get_override("SIGNAL_THRESHOLD") is None

    def test_delete_clears_existing(self):
        set_override("SIGNAL_THRESHOLD", "0.3")
        delete_override("SIGNAL_THRESHOLD")
        assert get_override("SIGNAL_THRESHOLD") is None

    def test_delete_nonexistent_is_noop(self):
        delete_override("SIGNAL_THRESHOLD")  # must not raise

    def test_get_all_empty_when_no_overrides(self):
        assert get_all_overrides() == {}

    def test_get_all_returns_all(self):
        set_override("SIGNAL_THRESHOLD", "0.3")
        set_override("GEM_TOP_N", "7")
        result = get_all_overrides()
        assert result["SIGNAL_THRESHOLD"] == "0.3"
        assert result["GEM_TOP_N"] == "7"


# ---------------------------------------------------------------------------
# resolve — env > settings > default precedence
# ---------------------------------------------------------------------------

class TestResolve:
    def test_env_wins_over_settings(self):
        set_override("SIGNAL_THRESHOLD", "0.3")
        result = resolve("SIGNAL_THRESHOLD", env_raw="0.5", default=0.25)
        assert result == pytest.approx(0.5)

    def test_settings_used_when_env_absent(self):
        set_override("SIGNAL_THRESHOLD", "0.3")
        result = resolve("SIGNAL_THRESHOLD", env_raw=None, default=0.25)
        assert result == pytest.approx(0.3)

    def test_default_used_when_both_absent(self):
        result = resolve("SIGNAL_THRESHOLD", env_raw=None, default=0.25)
        assert result == pytest.approx(0.25)

    def test_env_wins_over_settings_int(self):
        set_override("LOOP_INTERVAL_SECONDS", "120")
        result = resolve("LOOP_INTERVAL_SECONDS", env_raw="300", default=3600)
        assert result == 300

    def test_settings_override_int(self):
        set_override("LOOP_INTERVAL_SECONDS", "120")
        result = resolve("LOOP_INTERVAL_SECONDS", env_raw=None, default=3600)
        assert result == 120

    def test_settings_override_bool_true(self):
        set_override("SCREENER_ENABLED", "true")
        result = resolve("SCREENER_ENABLED", env_raw=None, default=False)
        assert result is True

    def test_settings_override_bool_false(self):
        set_override("SCREENER_ENABLED", "false")
        result = resolve("SCREENER_ENABLED", env_raw=None, default=True)
        assert result is False

    def test_env_bool_parsing(self):
        result = resolve("SCREENER_ENABLED", env_raw="true", default=False)
        assert result is True

    def test_corrupt_override_falls_back_to_default(self, temp_db):
        conn = sqlite3.connect(temp_db)
        conn.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("SIGNAL_THRESHOLD", "not_a_number", 0),
        )
        conn.commit()
        conn.close()
        result = resolve("SIGNAL_THRESHOLD", env_raw=None, default=0.25)
        assert result == pytest.approx(0.25)

    def test_locked_key_env_wins(self):
        result = resolve("CRYPTO_LIVE", env_raw="true", default=False)
        assert result is True

    def test_locked_key_default_used_without_env(self):
        result = resolve("CRYPTO_LIVE", env_raw=None, default=False)
        assert result is False
