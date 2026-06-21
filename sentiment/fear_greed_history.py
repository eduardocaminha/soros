"""Historical Fear & Greed Index from alternative.me.

Fetches the full history (limit=0 returns all available dates, ~2000+ days
back to 2018), builds a date-indexed lookup, and exposes helpers the backtest
A/B runner can use to align F&G values to OHLCV bar timestamps.

Design:
- Pure functions; no module-level state or I/O side effects at import time.
- Missing dates: backward-fill with the most recent available value so every
  bar gets a reading.  If the bar precedes the earliest record, returns None.
- Disk cache (JSON) is optional but avoids hitting the API on every backtest run.

Public API
----------
fetch_history(limit)          -> list[dict]   raw API response entries
build_index(entries)          -> dict[str,int] YYYY-MM-DD -> F&G value
lookup(index, ts)             -> int | None   F&G value for a unix ts
score_from_value(value)       -> float        value -> [-1, 1]
load_cached(path, max_age_s)  -> dict|None    load index from cache if fresh
save_cache(index, path)       -> None         persist index to JSON
get_index(cache_path, ...)    -> dict[str,int] fetch+cache convenience
"""

from __future__ import annotations

import datetime
import json
import logging
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

_log = logging.getLogger(__name__)

_FNG_URL = "https://api.alternative.me/fng/?limit={limit}&format=json"
_TIMEOUT = 15  # seconds

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def fetch_history(limit: int = 0) -> list[dict]:
    """Fetch Fear & Greed history from alternative.me.

    Parameters
    ----------
    limit:
        Number of entries to fetch.  0 means "all available" (API convention).

    Returns
    -------
    List of raw entry dicts, newest-first.  Each dict has keys:
        ``value``               str  e.g. "25"
        ``value_classification``str  e.g. "Fear"
        ``timestamp``           str  unix epoch seconds
    Returns empty list on any network / parse error.
    """
    url = _FNG_URL.format(limit=limit)
    try:
        req = Request(url, headers={"User-Agent": "soros-bot/1.0"})
        with urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except (URLError, ValueError, TimeoutError, OSError) as exc:
        _log.warning("fear_greed_history: fetch failed: %s", exc)
        return []

    if not isinstance(data, dict):
        return []
    entries = data.get("data")
    if not isinstance(entries, list):
        return []
    return entries


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def build_index(entries: list[dict]) -> dict[str, int]:
    """Build a YYYY-MM-DD → int index from raw API entries.

    Skips entries with missing or unparseable fields; logs a warning per skip.

    Parameters
    ----------
    entries:
        Raw list returned by :func:`fetch_history`.

    Returns
    -------
    Dict mapping ISO date strings to integer F&G values (0–100).
    """
    index: dict[str, int] = {}
    for entry in entries:
        try:
            ts_int = int(entry["timestamp"])
            value = int(entry["value"])
            date = datetime.datetime.utcfromtimestamp(ts_int).strftime("%Y-%m-%d")
            index[date] = value
        except (KeyError, ValueError, TypeError, OSError):
            _log.debug("fear_greed_history: skipping malformed entry: %s", entry)
    return index


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def lookup(index: dict[str, int], ts: int) -> int | None:
    """Return the Fear & Greed value for the UTC date of *ts*.

    Uses backward fill: if the exact date is missing, the most recent
    available date before *ts* is returned.  Returns ``None`` only when
    *ts* precedes every date in *index* (no prior data available).

    Parameters
    ----------
    index:
        Date index built by :func:`build_index`.
    ts:
        Unix epoch seconds representing the bar timestamp to look up.
    """
    if not index:
        return None

    target = datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

    # Fast path: exact date match.
    if target in index:
        return index[target]

    # Backward fill: find the most recent date that is <= target.
    sorted_dates = sorted(index.keys())
    # Binary-search equivalent: find rightmost date <= target.
    lo, hi = 0, len(sorted_dates) - 1
    best: str | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if sorted_dates[mid] <= target:
            best = sorted_dates[mid]
            lo = mid + 1
        else:
            hi = mid - 1

    if best is None:
        return None
    return index[best]


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------


def score_from_value(value: int) -> float:
    """Convert a F&G value (0–100) to a sentiment score in [-1, 1].

    Mirrors the formula used in ``pre_score()`` in ``sources_crypto``:
        score = (value - 50) / 50
    """
    return max(-1.0, min(1.0, (value - 50) / 50.0))


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


def load_cached(
    path: str | Path,
    max_age_secs: int = 86_400,
) -> dict[str, int] | None:
    """Load the F&G index from a JSON cache file if it is still fresh.

    Parameters
    ----------
    path:
        File path to the cache JSON (written by :func:`save_cache`).
    max_age_secs:
        Maximum age in seconds before the cache is considered stale.
        Default: 86 400 s (24 h) — F&G updates once per day.

    Returns
    -------
    The cached index dict, or ``None`` if the file is absent, too old, or
    cannot be parsed.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text())
        saved_at: float = payload["saved_at"]
        if time.time() - saved_at > max_age_secs:
            return None
        raw: dict = payload["index"]
        # Validate: all keys str, all values int.
        return {k: int(v) for k, v in raw.items()}
    except (KeyError, ValueError, TypeError, OSError):
        return None


def save_cache(index: dict[str, int], path: str | Path) -> None:
    """Persist *index* to a JSON cache file at *path*.

    Creates parent directories as needed.  Overwrites existing file atomically
    on POSIX via write-then-rename.

    Parameters
    ----------
    index:
        Date index to persist.
    path:
        Destination file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"saved_at": time.time(), "index": index}
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(p)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def get_index(
    cache_path: str | Path | None = None,
    limit: int = 0,
    max_age_secs: int = 86_400,
) -> dict[str, int]:
    """Return the F&G index, using a disk cache when *cache_path* is given.

    Workflow:
    1. If *cache_path* is set and the cache is fresh, return cached index.
    2. Otherwise fetch from alternative.me, build the index, optionally save
       the cache, and return it.

    Parameters
    ----------
    cache_path:
        Optional path for the cache JSON file.  Pass ``None`` to always
        fetch live (useful in tests).
    limit:
        Passed to :func:`fetch_history`.  0 = all available history.
    max_age_secs:
        Cache freshness threshold.
    """
    if cache_path is not None:
        cached = load_cached(cache_path, max_age_secs)
        if cached is not None:
            return cached

    entries = fetch_history(limit=limit)
    index = build_index(entries)

    if cache_path is not None and index:
        try:
            save_cache(index, cache_path)
        except OSError as exc:
            _log.warning("fear_greed_history: could not write cache %s: %s", cache_path, exc)

    return index
