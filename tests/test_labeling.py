"""Etiquetado triple barrera con velas sintéticas de resultado conocido.

Historia plana (open=close=100, high=100.5, low=99.5) => ATR(14) = 1,0, de modo que con SL 1,5 ATR y TP 3 ATR
el stop de un largo está en ~98,5 y el TP en ~103. Cada escenario modifica unas pocas velas en una decisión
distinta (separadas 100 velas para que no se contaminen). Sin red ni claves: ensure_1m se neutraliza.
"""
import duckdb
import pandas as pd
import pytest

from clefbt import db, labeling
from clefbt.features import compute_features

H = 24
N = 760
T0 = pd.Timestamp("2024-01-01")
SYMBOL = "BTC"
TS = pd.date_range(T0, periods=N, freq="h")
CFG = {
    "labeling": {"horizon_hours": H, "atr_period": 14, "main_sl_atr": 1.5, "main_tp_atr": 3.0,
                 "grid_sl": [1.5], "grid_tp": [3.0], "resolve_ambiguity": "main"},
    "data": {"ccxt": {}},
}

# índice de la vela de decisión de cada escenario (ventanas de 24 velas que no se pisan)
I_TOUCH_TP, I_TOUCH_SL = 20, 50
I_TP, I_SL, I_TIE_1M, I_TIE_NO_1M, I_GAP, I_TIME = 100, 200, 300, 400, 500, 600


def _set(df, idx, **kw):
    for k, v in kw.items():
        df.loc[idx, k] = v


def _build_candles():
    c = pd.DataFrame({"ts": TS, "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1.0})
    _set(c, I_TP + 1, open=100.2)                       # la entrada es esta apertura, no el cierre de la señal
    _set(c, I_TP + 3, high=120.0)                       # toca TP del largo (y stop del corto)
    _set(c, I_SL + 3, low=80.0)                         # toca stop del largo (y TP del corto)
    _set(c, I_TIE_1M + 5, high=120.0, low=80.0)         # ambas barreras en la misma vela de 1h
    _set(c, I_TIE_NO_1M + 5, high=120.0, low=80.0)      # idem, pero sin velas de 1m
    _set(c, I_GAP + 2, open=95.0, high=96.0, low=94.0, close=95.0)       # gap a la baja
    _set(c, I_TIME + H + 1, open=101.0, high=101.5, close=101.0)         # apertura de salida por tiempo
    # bordes exactos: usan el mismo ATR que el etiquetador, derivado de las propias features
    atr = compute_features(c)["atr"].iloc[I_TOUCH_TP]
    _set(c, I_TOUCH_TP + 3, high=100.0 + 3.0 * atr)      # el máximo roza el TP exactamente: NO cuenta
    atr = compute_features(c)["atr"].iloc[I_TOUCH_SL]
    _set(c, I_TOUCH_SL + 3, low=100.0 - 1.5 * atr)       # el mínimo alcanza el stop exactamente: SÍ cuenta
    return c


def _build_1m():
    hour = TS[I_TIE_1M + 5]
    m = pd.DataFrame({"ts": pd.date_range(hour, periods=60, freq="min"),
                      "open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0, "volume": 1.0})
    m.loc[10, ["high", "low"]] = [110.0, 100.0]         # minuto 10: sube al TP del largo / stop del corto
    m.loc[30, ["high", "low"]] = [100.0, 85.0]          # minuto 30: cae al stop del largo / TP del corto
    return m


@pytest.fixture(scope="module")
def con():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(labeling.data, "ensure_1m", lambda *a, **k: 0)    # los tests jamás tocan la red
        con = duckdb.connect(":memory:")
        con.execute(db.SCHEMA)
        for table, df in (("candles_1h", _build_candles()), ("candles_1m", _build_1m())):
            con.register("_df", df)
            con.execute(f"INSERT INTO {table} SELECT '{SYMBOL}', ts, open, high, low, close, volume FROM _df")
            con.unregister("_df")
        # liquidaciones antes de entrar y en el instante exacto de la entrada (no cuentan) y una durante la operación (sí)
        fund = pd.DataFrame({"ts": [TS[I_TP], TS[I_TP + 1], TS[I_TP + 2]], "rate": [0.7, 0.5, 0.0001]})
        con.register("_f", fund)
        con.execute(f"INSERT INTO funding SELECT '{SYMBOL}', ts, rate FROM _f")
        con.unregister("_f")
        labeling.run_labeling(con, CFG, SYMBOL)
        yield con
        con.close()


def lab(con, i, direction):
    df = con.execute("SELECT * FROM labels WHERE symbol=? AND ts=? AND direction=?",
                     [SYMBOL, TS[i].to_pydatetime(), direction]).df()
    assert len(df) == 1, f"se esperaba 1 etiqueta en i={i} {direction}, hay {len(df)}"
    return df.iloc[0]


def r_from_price(row, px, side):
    """R esperado si se sale a `px` (side=+1 largo, -1 corto), con el riesgo de esa misma etiqueta."""
    return side * (px - row.entry_px) / (row.risk_frac * row.entry_px)


def test_entry_is_next_open_not_signal_close(con):
    assert lab(con, I_TP, "long").entry_px == pytest.approx(100.2)


def test_tp_first_for_long_is_stop_for_short(con):
    long_, short = lab(con, I_TP, "long"), lab(con, I_TP, "short")
    assert long_.outcome == "tp" and long_.r_gross == pytest.approx(2.0)
    assert short.outcome == "sl" and short.r_gross == pytest.approx(-1.0)


def test_sl_first_for_long_is_tp_for_short(con):
    long_, short = lab(con, I_SL, "long"), lab(con, I_SL, "short")
    assert long_.outcome == "sl" and long_.r_gross == pytest.approx(-1.0)
    assert short.outcome == "tp" and short.r_gross == pytest.approx(2.0)


def test_same_candle_tie_is_resolved_with_1m_data(con):
    long_, short = lab(con, I_TIE_1M, "long"), lab(con, I_TIE_1M, "short")
    assert (long_.outcome, long_.resolved_by) == ("tp", "1m")      # el 1m sube primero
    assert (short.outcome, short.resolved_by) == ("sl", "1m")
    assert long_.r_gross == pytest.approx(2.0) and short.r_gross == pytest.approx(-1.0)


@pytest.mark.parametrize("direction", ["long", "short"])
def test_unresolvable_tie_is_never_guessed(con, direction):
    row = lab(con, I_TIE_NO_1M, direction)
    assert row.outcome == "ambiguous" and pd.isna(row.resolved_by)
    assert row.r_gross == pytest.approx(-1.0)         # caso pesimista
    assert row.r_gross_opt == pytest.approx(2.0)      # caso optimista (TP/SL)
    assert (row.y_pess, row.y_opt) == (0, 1)


def test_gap_through_stop_fills_at_the_worse_open(con):
    long_, short = lab(con, I_GAP, "long"), lab(con, I_GAP, "short")
    assert long_.outcome == "sl"
    assert long_.r_gross == pytest.approx(r_from_price(long_, 95.0, +1))
    assert long_.r_gross < -1.0                       # peor que el stop teórico
    assert short.outcome == "tp"
    assert short.r_gross == pytest.approx(r_from_price(short, 95.0, -1))
    assert short.r_gross > 2.0                        # mejor que el TP teórico


def test_time_exit_closes_at_next_open_and_is_not_a_tp(con):
    long_, short = lab(con, I_TIME, "long"), lab(con, I_TIME, "short")
    assert long_.outcome == short.outcome == "time"
    assert long_.r_gross == pytest.approx(r_from_price(long_, 101.0, +1))
    assert short.r_gross == pytest.approx(r_from_price(short, 101.0, -1))
    assert 0 < long_.r_gross < 2.0
    assert (long_.y_pess, long_.y_opt) == (0, 0)      # cierre por tiempo cuenta como "no tocó el TP"


def test_tp_must_be_crossed_not_just_touched(con):
    assert lab(con, I_TOUCH_TP, "long").outcome == "time"


def test_stop_counts_as_soon_as_it_is_reached(con):
    assert lab(con, I_TOUCH_SL, "long").outcome == "sl"


def test_funding_counts_only_settlements_during_the_trade(con):
    # 0,7 y 0,5 se liquidaron antes/en el instante de entrar (no cuentan); 0,0001 durante la operación
    # (el largo paga, el corto cobra)
    assert lab(con, I_TP, "long").funding_frac == pytest.approx(0.0001)
    assert lab(con, I_TP, "short").funding_frac == pytest.approx(-0.0001)


def test_labels_cover_exactly_the_valid_decisions(con):
    n, first, last = con.execute("SELECT count(*), min(ts), max(ts) FROM labels").fetchone()
    assert n == 2 * (N - H - 1 - 13)                  # ATR válido desde la vela 13; 2 direcciones
    assert pd.Timestamp(first) == TS[13]
    assert pd.Timestamp(last) == TS[N - H - 2]        # hace falta la vela de salida por tiempo
