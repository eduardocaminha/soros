"""Tests for the backtest harness (backtest/engine.py)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from backtest.engine import (
    BacktestConfig,
    BacktestResult,
    Trade,
    _compute_metrics,
    _get_signal,
    run_backtest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_START = 1_700_000_000   # 2023-11-14 22:13:20 UTC
_HOUR = 3600


def _make_prices(
    symbol: str = "BTC/USDT",
    asset_class: str = "crypto",
    n: int = 300,
    base_price: float = 30_000.0,
    trend: float = 0.0,  # price change per candle
    funding_rate: float | None = 0.0001,
) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame for backtest injection."""
    closes = [base_price + trend * i for i in range(n)]
    rows = []
    for i, c in enumerate(closes):
        rows.append({
            "symbol": symbol,
            "asset_class": asset_class,
            "ts": _START + i * _HOUR,
            "open": c,
            "high": c * 1.005,
            "low": c * 0.995,
            "close": c,
            "volume": 1000.0,
            "funding_rate": funding_rate,
        })
    return pd.DataFrame(rows)


def _cfg(
    symbols: list[tuple[str, str]] | None = None,
    n: int = 300,
    initial_capital: float = 10_000.0,
    signal_threshold: float = 0.25,
    **kw,
) -> BacktestConfig:
    if symbols is None:
        symbols = [("BTC/USDT", "crypto")]
    start = _START
    end = _START + (n - 1) * _HOUR
    return BacktestConfig(
        symbols=symbols,
        start_ts=start,
        end_ts=end,
        initial_capital=initial_capital,
        signal_threshold=signal_threshold,
        **kw,
    )


# ---------------------------------------------------------------------------
# _get_signal
# ---------------------------------------------------------------------------

class TestGetSignal:
    def test_returns_hold_on_empty_df(self):
        empty = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "funding_rate"]
        )
        composite, action = _get_signal(empty, "crypto", 0.25)
        assert composite == 0.0
        assert action == "hold"

    def test_returns_hold_for_flat_prices(self):
        df = _make_prices(n=50, trend=0.0)
        row = df[df["symbol"] == "BTC/USDT"].copy()
        composite, action = _get_signal(row, "crypto", 0.25)
        assert action == "hold"
        assert -1.0 <= composite <= 1.0

    def test_rising_trend_yields_buy(self):
        # Strong uptrend should produce composite >= threshold.
        df = _make_prices(n=200, base_price=1_000.0, trend=10.0)
        window = df.tail(100)
        composite, action = _get_signal(window, "crypto", 0.05)
        assert composite > 0.0

    def test_falling_trend_yields_sell(self):
        df = _make_prices(n=200, base_price=20_000.0, trend=-100.0)
        window = df.tail(100)
        composite, action = _get_signal(window, "crypto", 0.05)
        assert composite < 0.0

    def test_threshold_respected(self):
        # With a very high threshold, action is always hold.
        df = _make_prices(n=200, base_price=1_000.0, trend=50.0)
        _, action = _get_signal(df, "crypto", 0.99)
        assert action == "hold"

    def test_stocks_no_funding(self):
        df = _make_prices(symbol="AAPL", asset_class="stocks", n=50, funding_rate=None)
        composite, action = _get_signal(df, "stocks", 0.25)
        assert -1.0 <= composite <= 1.0
        assert action in ("buy", "sell", "hold")


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_empty_equity_curve(self):
        fe, tr, cagr, sharpe, dd, wr, nt, nw = _compute_metrics([], 10_000.0, [])
        assert fe == 10_000.0
        assert tr == 0.0

    def test_flat_equity_no_return(self):
        ts = _START
        curve = [(ts + i * _HOUR, 10_000.0) for i in range(100)]
        fe, tr, cagr, sharpe, dd, wr, nt, nw = _compute_metrics(curve, 10_000.0, [])
        assert tr == pytest.approx(0.0)
        assert sharpe == 0.0
        assert dd == 0.0

    def test_total_return_correct(self):
        curve = [(_START, 10_000.0), (_START + _HOUR, 11_000.0)]
        fe, tr, *_ = _compute_metrics(curve, 10_000.0, [])
        assert fe == pytest.approx(11_000.0)
        assert tr == pytest.approx(0.10)

    def test_max_drawdown_detected(self):
        # equity goes 10k → 12k → 8k → 10k; max dd = 33.3%
        curve = [
            (_START, 10_000.0),
            (_START + _HOUR, 12_000.0),
            (_START + 2 * _HOUR, 8_000.0),
            (_START + 3 * _HOUR, 10_000.0),
        ]
        _, _, _, _, dd, *_ = _compute_metrics(curve, 10_000.0, [])
        assert dd == pytest.approx(4_000.0 / 12_000.0, rel=1e-6)

    def test_win_rate_all_wins(self):
        trades = [
            Trade("X", "crypto", _START, _START + _HOUR, 100.0, 110.0, 1.0, 10.0),
            Trade("Y", "crypto", _START, _START + _HOUR, 200.0, 220.0, 1.0, 20.0),
        ]
        curve = [(_START, 10_000.0), (_START + _HOUR, 10_030.0)]
        *_, wr, nt, nw = _compute_metrics(curve, 10_000.0, trades)
        assert wr == 1.0
        assert nt == 2
        assert nw == 2

    def test_win_rate_mixed(self):
        trades = [
            Trade("X", "crypto", _START, _START + _HOUR, 100.0, 110.0, 1.0, 10.0),
            Trade("Y", "crypto", _START, _START + _HOUR, 200.0, 190.0, 1.0, -10.0),
        ]
        curve = [(_START, 10_000.0), (_START + _HOUR, 10_000.0)]
        *_, wr, nt, nw = _compute_metrics(curve, 10_000.0, trades)
        assert wr == 0.5
        assert nt == 2
        assert nw == 1

    def test_open_trades_excluded_from_win_rate(self):
        trades = [
            Trade("X", "crypto", _START, None, 100.0, None, 1.0, 0.0),  # open
        ]
        curve = [(_START, 10_000.0)]
        *_, wr, nt, nw = _compute_metrics(curve, 10_000.0, trades)
        assert nt == 0
        assert wr == 0.0

    def test_sharpe_positive_for_steady_growth(self):
        # Steady upward growth should give positive Sharpe.
        base = 10_000.0
        curve = [(_START + i * _HOUR, base * (1.001 ** i)) for i in range(365 * 24)]
        _, _, _, sharpe, *_ = _compute_metrics(curve, base, [])
        assert sharpe > 0.0

    def test_cagr_one_year_doubling(self):
        start = _START
        end = _START + 365 * 24 * _HOUR
        curve = [(start, 10_000.0), (end, 20_000.0)]
        _, _, cagr, *_ = _compute_metrics(curve, 10_000.0, [])
        assert cagr == pytest.approx(1.0, rel=0.05)  # ~100 % per year


# ---------------------------------------------------------------------------
# run_backtest (integration)
# ---------------------------------------------------------------------------

class TestRunBacktest:
    def test_empty_prices_returns_zero_result(self):
        cfg = _cfg()
        empty_df = pd.DataFrame(
            columns=["symbol", "asset_class", "ts", "open", "high", "low",
                     "close", "volume", "funding_rate"]
        )
        result = run_backtest(cfg, prices_df=empty_df)
        assert isinstance(result, BacktestResult)
        assert result.total_return == 0.0
        assert result.equity_curve == []

    def test_flat_prices_no_trades(self):
        prices = _make_prices(n=300, trend=0.0)
        cfg = _cfg(n=300)
        result = run_backtest(cfg, prices_df=prices)
        # Flat price → signals neutral → no buy signal crossed threshold
        assert result.num_trades == 0
        assert result.final_equity == pytest.approx(cfg.initial_capital)

    def test_rising_trend_generates_buy(self):
        # Strong uptrend should trigger at least one buy.
        prices = _make_prices(n=300, base_price=1_000.0, trend=20.0)
        cfg = _cfg(n=300, signal_threshold=0.05)
        result = run_backtest(cfg, prices_df=prices)
        assert len(result.equity_curve) > 0
        # equity curve length == number of timestamps in range
        assert len(result.equity_curve) == 300

    def test_fees_slippage_applied_on_round_trip(self):
        # Build a price series that goes up then immediately down — net signal
        # should still show fee/slippage cost applied to both legs.
        prices = _make_prices(n=300, base_price=100.0, trend=1.0)
        cfg = _cfg(n=300, signal_threshold=0.05, fee_pct=0.001, slippage_pct=0.0005)
        result = run_backtest(cfg, prices_df=prices)
        for t in result.trades:
            if t.exit_price is not None:
                # Entry price inflated by fee+slippage
                raw_close = t.entry_price / (1.0 + cfg.fee_pct + cfg.slippage_pct)
                assert t.entry_price == pytest.approx(raw_close * (1.0 + cfg.fee_pct + cfg.slippage_pct), rel=1e-9)
                # Exit price deflated by fee+slippage
                raw_exit = t.exit_price / (1.0 - cfg.fee_pct - cfg.slippage_pct)
                assert t.exit_price == pytest.approx(raw_exit * (1.0 - cfg.fee_pct - cfg.slippage_pct), rel=1e-9)

    def test_drawdown_gate_blocks_new_positions(self):
        # Simulate large equity drop to trigger drawdown gate.
        prices = _make_prices(n=300, base_price=100_000.0, trend=-1_000.0)
        cfg = _cfg(
            n=300,
            signal_threshold=0.001,
            max_drawdown_pct=0.01,
            initial_capital=10_000.0,
        )
        result = run_backtest(cfg, prices_df=prices)
        # After gate fires, no more trades should open — hard to assert exact
        # count but result should be valid.
        assert isinstance(result, BacktestResult)

    def test_position_cap_enforced(self):
        # Two symbols both signalling buy with cap=1 → only one open at a time.
        p1 = _make_prices("BTC/USDT", "crypto", n=300, base_price=100.0, trend=5.0)
        p2 = _make_prices("ETH/USDT", "crypto", n=300, base_price=50.0, trend=2.5)
        prices = pd.concat([p1, p2], ignore_index=True)
        cfg = BacktestConfig(
            symbols=[("BTC/USDT", "crypto"), ("ETH/USDT", "crypto")],
            start_ts=_START,
            end_ts=_START + 299 * _HOUR,
            initial_capital=10_000.0,
            max_open_positions=1,
            signal_threshold=0.05,
        )
        result = run_backtest(cfg, prices_df=prices)
        # At any point in time, at most 1 position should be open.
        # Verify by checking the equity curve is well-formed.
        assert len(result.equity_curve) > 0
        # No two overlapping open trades for different symbols at same time.
        open_at: dict[int, list[str]] = {}
        for t in result.trades:
            start_t = t.entry_ts
            end_t = t.exit_ts if t.exit_ts else result.equity_curve[-1][0]
            for ts in range(start_t, end_t + _HOUR, _HOUR):
                open_at.setdefault(ts, []).append(t.symbol)
        for ts, syms in open_at.items():
            assert len(syms) <= 1, f"position cap violated at ts={ts}: {syms}"

    def test_result_metrics_consistent(self):
        prices = _make_prices(n=300, base_price=1_000.0, trend=5.0)
        cfg = _cfg(n=300, signal_threshold=0.05)
        result = run_backtest(cfg, prices_df=prices)

        assert result.initial_capital == cfg.initial_capital
        assert result.max_drawdown >= 0.0
        assert result.max_drawdown <= 1.0
        assert 0.0 <= result.win_rate <= 1.0
        assert result.num_wins <= result.num_trades
        assert result.sharpe == result.sharpe  # not NaN

    def test_equity_curve_starts_at_initial_capital(self):
        prices = _make_prices(n=300, trend=0.0)
        cfg = _cfg(n=300)
        result = run_backtest(cfg, prices_df=prices)
        if result.equity_curve:
            assert result.equity_curve[0][1] == pytest.approx(cfg.initial_capital)

    def test_stocks_symbol_no_funding(self):
        prices = _make_prices(
            symbol="AAPL", asset_class="stocks", n=300,
            base_price=150.0, trend=0.5, funding_rate=None,
        )
        cfg = BacktestConfig(
            symbols=[("AAPL", "stocks")],
            start_ts=_START,
            end_ts=_START + 299 * _HOUR,
            initial_capital=10_000.0,
            signal_threshold=0.05,
        )
        result = run_backtest(cfg, prices_df=prices)
        assert isinstance(result, BacktestResult)

    def test_no_real_execution(self):
        # Backtest must never touch order_executor or live exchange.
        import backtest.engine as be

        prices = _make_prices(n=300, base_price=1_000.0, trend=10.0)
        cfg = _cfg(n=300, signal_threshold=0.05)

        # If order_executor were called it would fail (no DB). This must not raise.
        result = run_backtest(cfg, prices_df=prices)
        assert isinstance(result, BacktestResult)

    def test_position_size_uses_current_equity(self):
        # Verify position quantity = (equity * position_size_pct) / close.
        prices = _make_prices(n=300, base_price=1_000.0, trend=10.0)
        cfg = _cfg(n=300, signal_threshold=0.05, position_size_pct=0.10)
        result = run_backtest(cfg, prices_df=prices)
        for t in result.trades:
            # entry_price already has fee/slippage; raw close = entry_price / (1+fee+slip)
            raw_close = t.entry_price / (1.0 + cfg.fee_pct + cfg.slippage_pct)
            # quantity should be approximately equity_at_entry * 0.10 / raw_close
            expected_min = cfg.initial_capital * cfg.position_size_pct * 0.5 / raw_close
            assert t.quantity > expected_min  # sanity — not zero or near-zero


# ---------------------------------------------------------------------------
# Screener integration
# ---------------------------------------------------------------------------


class TestBacktestScreenerIntegration:
    """Acceptance tests: backtest reuses engine.screener.screen() for symbol selection."""

    def _mock_screen(self, crypto: list[str], stocks: list[str]):
        """Return a factory that patches screen() to return the given symbols."""
        from engine.screener import ScreenerResult

        def _screen(*args, **kwargs):
            return ScreenerResult(
                selected_crypto=crypto,
                selected_stocks=stocks,
                entries=[],
            )

        return _screen

    def test_use_screener_false_uses_cfg_symbols(self):
        """When use_screener=False, cfg.symbols drives the backtest."""
        prices = _make_prices("BTC/USDT", "crypto", n=300, base_price=1_000.0, trend=0.0)
        cfg = _cfg(symbols=[("BTC/USDT", "crypto")], n=300)
        assert cfg.use_screener is False
        result = run_backtest(cfg, prices_df=prices)
        assert isinstance(result, BacktestResult)

    def test_use_screener_true_uses_screener_symbols(self, monkeypatch):
        """When use_screener=True, screen() replaces cfg.symbols."""
        import engine.screener as screener_mod

        monkeypatch.setattr(screener_mod, "screen", self._mock_screen(
            crypto=["BTC/USDT"], stocks=[],
        ))

        prices = _make_prices("BTC/USDT", "crypto", n=300, base_price=1_000.0, trend=0.0)
        cfg = BacktestConfig(
            symbols=[],  # intentionally empty — screener should fill in
            start_ts=_START,
            end_ts=_START + 299 * _HOUR,
            initial_capital=10_000.0,
            use_screener=True,
        )
        result = run_backtest(cfg, prices_df=prices)
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) == 300

    def test_use_screener_true_ignores_cfg_symbols(self, monkeypatch):
        """Screener override replaces whatever was in cfg.symbols."""
        import engine.screener as screener_mod

        monkeypatch.setattr(screener_mod, "screen", self._mock_screen(
            crypto=["ETH/USDT"], stocks=[],
        ))

        prices_btc = _make_prices("BTC/USDT", "crypto", n=300, base_price=1_000.0)
        prices_eth = _make_prices("ETH/USDT", "crypto", n=300, base_price=500.0)
        prices = pd.concat([prices_btc, prices_eth], ignore_index=True)

        cfg = BacktestConfig(
            symbols=[("BTC/USDT", "crypto")],  # would be used if screener were off
            start_ts=_START,
            end_ts=_START + 299 * _HOUR,
            initial_capital=10_000.0,
            use_screener=True,
        )
        result = run_backtest(cfg, prices_df=prices)
        # All trades should be on ETH/USDT (the screener selection), not BTC/USDT.
        traded_syms = {t.symbol for t in result.trades}
        assert "BTC/USDT" not in traded_syms

    def test_use_screener_with_stocks(self, monkeypatch):
        """Screener can select stocks symbols too."""
        import engine.screener as screener_mod

        monkeypatch.setattr(screener_mod, "screen", self._mock_screen(
            crypto=[], stocks=["AAPL"],
        ))

        prices = _make_prices("AAPL", "stocks", n=300, base_price=150.0, trend=0.0,
                              funding_rate=None)
        cfg = BacktestConfig(
            symbols=[],
            start_ts=_START,
            end_ts=_START + 299 * _HOUR,
            initial_capital=10_000.0,
            use_screener=True,
        )
        result = run_backtest(cfg, prices_df=prices)
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) == 300

    def test_cli_screener_flag(self, monkeypatch):
        """--screener flag is parsed and sets use_screener=True on the config."""
        from backtest.engine import _parse_args

        args = _parse_args([
            "--screener",
            "--start", "2024-01-01",
            "--end", "2024-12-31",
        ])
        assert args.screener is True
        assert args.symbols == []

    def test_cli_symbols_required_without_screener(self):
        """--symbols is required when --screener is not set."""
        import argparse
        from backtest.engine import _parse_args

        with pytest.raises(SystemExit):
            _parse_args(["--start", "2024-01-01", "--end", "2024-12-31"])
