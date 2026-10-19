"""Regla de continuación en extremos de RSI: una posición a la vez y retorno futuro sin mirar antes de la entrada."""
import duckdb
import numpy as np
import pandas as pd
import pytest

from clefbt import db, rules
from clefbt.features import compute_features

CFG = {"labeling": {"horizon_hours": 24, "atr_period": 14}, "costs": {"round_trip_cost_frac": 0.0011}}


def test_take_trades_holds_one_position_at_a_time():
    sig = np.array([1, 1, 1, 0, 1, 1, 0, 1])
    ret = np.ones(8)
    # entra en 0 y no mira señales hasta salir (hold=3): siguiente entrada posible en 3 => la primera señal desde ahí es la 4
    assert list(rules.take_trades(sig, ret, hold=3)) == [0, 4, 7]


def test_take_trades_skips_rows_without_a_forward_return():
    sig = np.array([1, 1, 1])
    ret = np.array([np.nan, 0.5, 0.5])
    assert list(rules.take_trades(sig, ret, hold=1)) == [1, 2]


def _db_with_candles(n=800, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = np.r_[close[0], close[:-1]]
    c = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=n, freq="h"), "open": open_,
                      "high": np.maximum(open_, close) * 1.002, "low": np.minimum(open_, close) * 0.998,
                      "close": close, "volume": 1.0})
    con = duckdb.connect(":memory:")
    con.execute(db.SCHEMA)
    con.register("_c", c)
    con.execute("INSERT INTO candles_1h SELECT 'BTC', ts, open, high, low, close, volume FROM _c")
    return con, c


def test_forward_return_starts_at_the_next_open_and_ends_24h_later():
    con, c = _db_with_candles()
    df = rules.candle_dataset(con, CFG, "BTC")
    t = 300
    ts = c["ts"].iloc[t]
    atr = compute_features(c)["atr"].iloc[t]
    expected = (c["open"].iloc[t + 25] - c["open"].iloc[t + 1]) / atr           # entrada: apertura de t+1; salida: apertura de t+25
    assert df.loc[ts, "ret_fwd_atr"] == pytest.approx(expected)
    assert df.loc[ts, "ret_pct_fwd"] == pytest.approx(c["open"].iloc[t + 25] / c["open"].iloc[t + 1] - 1)


def test_last_rows_have_no_forward_return():
    con, _ = _db_with_candles()
    df = rules.candle_dataset(con, CFG, "BTC")
    assert df["ret_fwd_atr"].iloc[-25:].isna().all()
    assert df["ret_fwd_atr"].iloc[:-25].notna().all()


def test_continuation_rule_goes_with_the_extreme():
    """RSI muy bajo => corto; si luego el precio cae, la regla gana (bruta positiva)."""
    idx = pd.date_range("2024-01-01", periods=2500, freq="h")      # >= 30 operaciones no solapadas
    df = pd.DataFrame({"rsi": 10.0, "ret_fwd_atr": -2.0, "cost_atr": 0.1}, index=idx)     # precio cae 2 ATR tras cada señal
    rows = rules._segment_rows(df, "sintético", 24, 0.0011)
    row = next(r for r in rows if r["banda"] == "30/70")
    assert row["bruta ATR"] == pytest.approx(2.0)
    df2 = df.assign(rsi=90.0, ret_fwd_atr=+2.0)                                           # RSI alto y el precio sigue subiendo
    row2 = next(r for r in rules._segment_rows(df2, "sintético", 24, 0.0011) if r["banda"] == "30/70")
    assert row2["bruta ATR"] == pytest.approx(2.0)
