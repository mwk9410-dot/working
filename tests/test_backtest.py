import pandas as pd

from fibtrader import BotConfig
from fibtrader.backtest import backtest_fib_confluence
from fibtrader.features import add_features
from fibtrader.fib import attach_recent_swing_and_fib
from fibtrader.swing import zigzag_confirmed


def _pipeline(df, cfg):
    sw = zigzag_confirmed(df, pct=cfg.zigzag_pct, confirm_bars=cfg.confirm_bars)
    sw = attach_recent_swing_and_fib(sw, ratios=cfg.fib_ratios)
    return add_features(sw, cfg)


def test_backtest_returns_dataframe_and_stats(synth_df):
    cfg = BotConfig(zigzag_pct=0.05, min_confluence=1)  # loose to ensure trades
    feat = _pipeline(synth_df, cfg)
    trades, stats = backtest_fib_confluence(feat, cfg)
    assert isinstance(trades, pd.DataFrame)
    if stats is not None:
        assert {"trades", "win_rate", "max_drawdown"}.issubset(stats.keys())


def test_no_trades_when_confluence_unreachable(synth_df):
    cfg = BotConfig(min_confluence=99)
    feat = _pipeline(synth_df, cfg)
    trades, stats = backtest_fib_confluence(feat, cfg)
    assert trades.empty
    assert stats is None


def test_higher_costs_dont_improve_returns(synth_df):
    cfg_low = BotConfig(zigzag_pct=0.05, min_confluence=1, fee_bps=0, slippage_bps=0)
    cfg_high = BotConfig(zigzag_pct=0.05, min_confluence=1, fee_bps=50, slippage_bps=50)
    feat = _pipeline(synth_df, cfg_low)
    _, stats_low = backtest_fib_confluence(feat, cfg_low)
    _, stats_high = backtest_fib_confluence(feat, cfg_high)
    if stats_low and stats_high:
        assert stats_high["avg_ret"] <= stats_low["avg_ret"] + 1e-9
