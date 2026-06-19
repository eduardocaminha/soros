"""Universe assembler: market-cap base ∪ gem candidates with origin tagging.

Combines:
  - data.universe.get_base_universe()  — top-N by market cap
  - data.gem_scanner.scan_gems()       — CEX ignition candidates
  - data.dex_scanner.get_dex_trending_symbols() — DEX boost for gem scores

Origin values (stored in screener_runs.origin for the dashboard):
  'base'        — from CoinGecko market-cap tier
  'gem'         — from ignition scanner (volume surge + ROC)
  'dex_boosted' — gem with DEX trending signal applied

Usage::

    from data.assembler import assemble_universe
    uni = assemble_universe()
    # uni.all_symbols  — base + gems, deduped
    # uni.origins      — symbol → 'base' | 'gem' | 'dex_boosted'
    # uni.gem_candidates — GemCandidate list (for risk sizing in next step)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data.dex_scanner import get_dex_trending_symbols
from data.gem_scanner import GemCandidate, scan_gems
from data.universe import get_base_universe
from database.db import get_logger

_log = get_logger(__name__)


@dataclass
class AssembledUniverse:
    """Result of one universe assembly pass."""

    base_symbols: list[str]
    gem_symbols: list[str]
    dex_boosted_symbols: frozenset[str]
    all_symbols: list[str]          # base + gems deduped, base order preserved
    origins: dict[str, str]         # symbol → 'base' | 'gem' | 'dex_boosted'
    gem_candidates: list[GemCandidate] = field(default_factory=list)


def assemble_universe(
    exchange: Any | None = None,
    n: int | None = None,
    refresh_secs: int | None = None,
) -> AssembledUniverse:
    """Build the full crypto universe: market-cap base ∪ ignition gem candidates.

    Parameters
    ----------
    exchange:
        ccxt exchange instance for gem scanning. Defaults to a keyless
        Binance spot instance created inside scan_gems().
    n:
        Number of market-cap base symbols. Defaults to config.MARKETCAP_TOP_N.
    refresh_secs:
        Cache TTL for the market-cap base. Defaults to config.MARKETCAP_REFRESH_SECS.

    Returns
    -------
    AssembledUniverse
        Combined universe with per-symbol origin tagging and the raw
        GemCandidate list (used for risk sizing in the risk manager).
    """
    # 1. Market-cap base tier (cached; degrades gracefully on API failure)
    base = get_base_universe(n=n, refresh_secs=refresh_secs)
    base_set = set(base)

    # 2. DEX trending symbols — used only as a score boost, not for execution
    dex_trending = get_dex_trending_symbols()

    # 3. Gem scanner — base_symbols excluded to prevent duplicates
    gem_candidates = scan_gems(
        exchange=exchange,
        base_symbols=base_set,
        dex_trending_symbols=dex_trending,
    )
    gem_symbols = [c.symbol for c in gem_candidates]
    dex_boosted: frozenset[str] = frozenset(c.symbol for c in gem_candidates if c.dex_boost)

    # 4. Build origins map and all_symbols (base order first, then gems)
    origins: dict[str, str] = {}
    all_symbols: list[str] = []

    for sym in base:
        origins[sym] = "base"
        all_symbols.append(sym)

    for sym in gem_symbols:
        if sym not in base_set:
            origins[sym] = "dex_boosted" if sym in dex_boosted else "gem"
            all_symbols.append(sym)

    _log.info(
        "universe assembled: base=%d gems=%d dex_boosted=%d total=%d",
        len(base),
        len(gem_symbols),
        len(dex_boosted),
        len(all_symbols),
    )

    return AssembledUniverse(
        base_symbols=base,
        gem_symbols=gem_symbols,
        dex_boosted_symbols=dex_boosted,
        all_symbols=all_symbols,
        origins=origins,
        gem_candidates=gem_candidates,
    )
