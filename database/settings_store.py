"""Runtime settings store: CRUD for the settings table with allowlist enforcement.

Precedence (highest to lowest): env > settings table > hardcoded default.

Only keys in config.SETTINGS_ALLOWLIST may be written here; all others
(execution toggles, hard risk limits) raise LockedKeyError.
"""

from __future__ import annotations

import time
from typing import Any

import config
from database.db import get_connection


class LockedKeyError(ValueError):
    """Raised when a write targets a key absent from SETTINGS_ALLOWLIST."""


class InvalidValueError(ValueError):
    """Raised when a value fails type or range validation."""


def _parse_value(key: str, raw: str) -> Any:
    """Parse *raw* against the allowlist spec for *key*.

    Returns the typed Python value.
    Raises InvalidValueError on type mismatch or out-of-range.
    """
    spec = config.SETTINGS_ALLOWLIST[key]
    typ = spec["type"]

    if typ is bool:
        if raw.lower() in ("true", "1"):
            value: Any = True
        elif raw.lower() in ("false", "0"):
            value = False
        else:
            raise InvalidValueError(f"{key}: expected bool ('true'/'false'), got {raw!r}")
    else:
        try:
            value = typ(raw)
        except (ValueError, TypeError) as exc:
            raise InvalidValueError(
                f"{key}: cannot convert {raw!r} to {typ.__name__}"
            ) from exc

    min_val = spec.get("min")
    max_val = spec.get("max")
    if min_val is not None and value < min_val:
        raise InvalidValueError(f"{key}: {value} is below minimum {min_val}")
    if max_val is not None and value > max_val:
        raise InvalidValueError(f"{key}: {value} exceeds maximum {max_val}")

    return value


def get_override(key: str) -> str | None:
    """Return the raw stored override for *key*, or None if not set."""
    conn = get_connection()
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def set_override(key: str, value: str) -> None:
    """Write a runtime override, enforcing the allowlist and value constraints.

    Raises LockedKeyError if *key* is absent from SETTINGS_ALLOWLIST.
    Raises InvalidValueError if *value* fails type/range validation.
    """
    if key not in config.SETTINGS_ALLOWLIST:
        raise LockedKeyError(
            f"{key!r} is not in SETTINGS_ALLOWLIST and cannot be overridden at runtime"
        )
    _parse_value(key, value)  # validate; raise on bad value

    conn = get_connection()
    conn.execute(
        """
        INSERT INTO settings (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
        """,
        (key, value, int(time.time())),
    )
    conn.commit()


def delete_override(key: str) -> None:
    """Remove a runtime override (falls back to env/default on next reload)."""
    conn = get_connection()
    conn.execute("DELETE FROM settings WHERE key = ?", (key,))
    conn.commit()


def get_all_overrides() -> dict[str, str]:
    """Return all current settings overrides as {key: raw_value}."""
    conn = get_connection()
    rows = conn.execute("SELECT key, value FROM settings").fetchall()
    return {row[0]: row[1] for row in rows}


def resolve(key: str, env_raw: str | None, default: Any) -> Any:
    """Apply env > settings > default precedence for an allowlisted key.

    - *env_raw*: the raw string from os.environ, or None if unset.
    - *default*: the hardcoded fallback value (already the right type).

    Returns the effective typed value.  If the settings override exists but
    is invalid (e.g. corrupt DB row), logs a warning and falls back to default.
    """
    spec = config.SETTINGS_ALLOWLIST.get(key)
    if spec is None:
        # Locked key: no settings layer
        if env_raw is not None:
            typ = type(default)
            if typ is bool:
                return env_raw.lower() == "true"
            try:
                return typ(env_raw)
            except (ValueError, TypeError):
                return default
        return default

    typ = spec["type"]

    if env_raw is not None:
        if typ is bool:
            return env_raw.lower() == "true"
        return typ(env_raw)

    override = get_override(key)
    if override is not None:
        try:
            return _parse_value(key, override)
        except InvalidValueError:
            return default

    return default
