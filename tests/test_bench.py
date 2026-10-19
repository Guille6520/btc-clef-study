"""Benchmark clásico, derivados y estrategias diarias: sin fuga temporal y con la contabilidad correcta."""
import duckdb
import numpy as np
import pandas as pd
import pytest

from clefbt import classic, daily, db, derivs

BENCH = {"holdout_start": "2025-10-14", "first_test_year": 2021, "embargo_hours": 25}


# ---------- walk-forward purgado ----------
class _Recorder:
    """Modelo falso que anota hasta qué instante vio datos de entrenamiento y a partir de cuándo predice."""
    def __init__(self, log):
        self.log = log

    def fit(self, X, y):
        self.max_train = X[:, 0].max()
        return self

    def predict_proba(self, X):
        self.log.append((self.max_train, X[:, 0].min()))
        return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])


ORIGIN = pd.Timestamp("2019-10-01")


def _hours(ts):
    """Horas desde ORIGIN; independiente de la resolución interna (ns/us) que use pandas."""
    return (pd.to_datetime(ts) - ORIGIN) / pd.Timedelta(hours=1)


def _frame():
    idx = pd.date_range(ORIGIN, "2026-03-01", freq="h")
    return pd.DataFrame({"t": np.asarray(_hours(idx), dtype=float), "y": np.random.default_rng(0).integers(0, 2, len(idx))},
                        index=idx)


def test_walk_forward_never_trains_closer_than_the_embargo_to_the_test_period():
    log = []
    df = _frame()
    classic.walk_forward(df, ["t"], "y", lambda: _Recorder(log), BENCH)
    assert len(log) == 5                                                   # 2021 ... 2025
    for max_train, min_test in log:
        assert min_test - max_train >= BENCH["embargo_hours"]           # unidades: horas


def test_walk_forward_never_predicts_inside_the_blind_holdout():
    df = _frame()
    pred, const = classic.walk_forward(df, ["t"], "y", lambda: _Recorder([]), BENCH)
    hold = pd.Timestamp(BENCH["holdout_start"])
    assert pred[pred.index >= hold].isna().all() and const[const.index >= hold].isna().all()
    assert pred[(pred.index >= "2021-01-01") & (pred.index < hold)].notna().all()


def test_walk_forward_training_set_excludes_the_holdout_even_in_the_final_year():
    log = []
    df = _frame()
    classic.walk_forward(df, ["t"], "y", lambda: _Recorder(log), BENCH)
    hold_t = _hours(pd.Timestamp(BENCH["holdout_start"]))
    assert all(max_train < hold_t for max_train, _ in log)


# ---------- derivados: sin mirar el futuro ----------
def test_snapshot_belongs_to_the_candle_that_closes_after_it():
    ts = pd.Series(pd.to_datetime(["2024-01-01 12:05:00", "2024-01-01 12:55:00", "2024-01-01 13:00:00",
                                   "2024-01-01 13:00:01"]))
    got = derivs.snapshot_candle_ts(ts).tolist()
    assert got == [pd.Timestamp("2024-01-01 12:00"), pd.Timestamp("2024-01-01 12:00"),
                   pd.Timestamp("2024-01-01 12:00"), pd.Timestamp("2024-01-01 13:00")]


def test_daily_context_uses_the_previous_days_value():
    con = duckdb.connect(":memory:")
    con.execute(db.SCHEMA)
    days = pd.date_range("2024-01-01", "2024-01-10")
    rows = []
    for name in ("fng", "VIXCLS", "NASDAQCOM", "DTWEXBGS", "DGS10", "SP500"):
        rows += [(name, d.date(), 100.0 + i) for i, d in enumerate(days)]
    con.register("_x", pd.DataFrame(rows, columns=["name", "date", "value"]))
    con.execute("INSERT INTO daily_ext SELECT name, date, value FROM _x")
    idx = pd.date_range("2024-01-05", periods=72, freq="h")
    f = pd.DataFrame({"close": 100.0, "rv168": 1.0}, index=idx)
    out = classic.add_ext_features(f, con, "BTC")
    # el 5-ene (índice 4 de la serie: 104) debe ver el valor del 4-ene (103) durante TODO el día, incluida la hora 00:00
    assert out.loc["2024-01-05 00:00", "fng"] == 103.0 and out.loc["2024-01-05 23:00", "fng"] == 103.0
    assert out.loc["2024-01-06 00:00", "fng"] == 104.0


def test_open_interest_features_are_backward_looking():
    con = duckdb.connect(":memory:")
    con.execute(db.SCHEMA)
    idx = pd.date_range("2024-01-01", periods=400, freq="h")
    oi = pd.DataFrame({"symbol": "BTC", "ts": idx, "oi": np.r_[np.full(300, 100.0), np.full(100, 200.0)],
                       "oi_value": 1.0, "ls_top_acc": 1.0, "ls_top_pos": 1.0, "ls_all": 1.0, "taker_ls_vol": 1.0})
    con.register("_o", oi)
    con.execute("INSERT INTO oi_1h SELECT * FROM _o")
    f = pd.DataFrame({"close": 100.0, "rv168": 1.0}, index=idx)
    out = classic.add_ext_features(f, con, "BTC")
    # el salto del OI ocurre en la hora 300: el cambio a 24 h solo puede verse A PARTIR de ahí
    assert out["oi_chg_24"].iloc[:300].fillna(0).eq(0).all()
    assert out["oi_chg_24"].iloc[300] == pytest.approx(1.0)


# ---------- estrategias diarias ----------
def _days(ret, funding=None):
    idx = pd.date_range("2024-01-01", periods=len(ret), freq="D")
    return pd.DataFrame({"ret": ret, "funding": funding if funding is not None else 0.0}, index=idx)


def test_position_is_earned_the_day_after_it_is_decided():
    d = _days([np.nan, 0.0, 0.5, 0.0, 0.0])
    pos = pd.Series([0.0, 0.0, 1.0, 1.0, 1.0], index=d.index)        # decidida al cierre del día 2 (la subida del día 2 NO se cobra)
    x = daily.pnl_series(d, pos)
    assert x.loc["2024-01-03"] == pytest.approx(0.0)                   # el día +50 % aún sin posición
    assert x.loc["2024-01-04"] == pytest.approx(-daily.SIDE_COST)      # entra al cierre del 3: paga coste, ret 0
    assert x.loc["2024-01-05"] == pytest.approx(0.0)


def test_long_pays_positive_funding_and_short_receives_it():
    d = _days([np.nan, 0.0, 0.0, 0.0], funding=0.001)
    long_ = daily.pnl_series(d, pd.Series(1.0, index=d.index))
    short = daily.pnl_series(d, pd.Series(-1.0, index=d.index))
    assert long_.iloc[-1] == pytest.approx(-0.001)
    assert short.iloc[-1] == pytest.approx(+0.001)


def test_metrics_for_a_constant_daily_return():
    x = pd.Series(0.001, index=pd.date_range("2024-01-01", periods=365))
    m = daily.metrics(x)
    assert m["CAGR"] == pytest.approx(1.001 ** 365 - 1, rel=1e-6)
    assert m["maxDD"] == pytest.approx(0.0)


def test_strategy_positions_need_their_warmup():
    close = pd.Series(np.linspace(100, 200, 400), index=pd.date_range("2023-01-01", periods=400))
    d = pd.DataFrame({"close": close, "ret": close.pct_change(), "funding": 0.0})
    pos = daily.positions(d)
    assert pos["tendencia SMA200 (largo/plano)"].iloc[:199].isna().all()          # sin 200 días de historia no hay señal
    assert pos["tendencia SMA200 (largo/plano)"].iloc[-1] == 1.0                    # tendencia alcista sostenida
