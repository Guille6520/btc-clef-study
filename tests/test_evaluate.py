"""Métricas y simulación: casos con respuesta conocida."""
import numpy as np
import pytest

from clefbt.evaluate import auc, brier, logloss, simulate, summarize_trades


def test_auc_known_value():
    assert auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == pytest.approx(0.75)


def test_auc_is_half_for_constant_scores():
    assert auc(np.full(6, 0.3), np.array([0, 1, 0, 1, 0, 1])) == pytest.approx(0.5)


def test_brier_and_logloss_known_values():
    assert brier(np.array([0.9, 0.1]), np.array([1, 0])) == pytest.approx(0.01)
    assert logloss(np.array([0.5, 0.5]), np.array([1, 0])) == pytest.approx(np.log(2))


def test_summarize_trades():
    s = summarize_trades(np.array([1.0, -1.0, 2.0]))
    assert s["n"] == 3
    assert s["media_R"] == pytest.approx(2 / 3)
    assert s["suma_R"] == pytest.approx(2.0)
    assert s["win"] == pytest.approx(2 / 3)
    assert s["pf"] == pytest.approx(3.0)
    assert s["max_dd_R"] == pytest.approx(1.0)


def test_summarize_trades_empty():
    s = summarize_trades(np.array([]))
    assert s["n"] == 0 and s["suma_R"] == 0.0


def test_simulation_keeps_one_position_at_a_time():
    """Señales a t, t+1h y t+2h. La primera sale a t+3h: la segunda se salta y la tercera (decisión en t+3h) entra."""
    ts = np.array(["2024-01-01T00", "2024-01-01T01", "2024-01-01T02"], dtype="datetime64[ns]")
    p_long = np.array([0.9, 0.9, 0.9])
    p_short = np.array([0.1, 0.1, 0.1])
    r_long = np.array([1.0, 10.0, 3.0])
    r_short = np.zeros(3)
    exit_long = np.array(["2024-01-01T03", "2024-01-01T04", "2024-01-01T05"], dtype="datetime64[ns]")
    trades = simulate(ts, p_long, p_short, r_long, r_short, exit_long, exit_long, thr=0.5)
    assert list(trades) == [1.0, 3.0]


def test_simulation_respects_threshold_and_picks_the_more_confident_side():
    ts = np.array(["2024-01-01T00", "2024-01-02T00"], dtype="datetime64[ns]")
    exits = np.array(["2024-01-01T05", "2024-01-02T05"], dtype="datetime64[ns]")
    p_long = np.array([0.3, 0.7])
    p_short = np.array([0.4, 0.8])
    r_long = np.array([1.0, 1.0])
    r_short = np.array([5.0, 5.0])
    trades = simulate(ts, p_long, p_short, r_long, r_short, exits, exits, thr=0.6)
    assert list(trades) == [5.0]        # la primera (máx 0,4) no llega al umbral; en la segunda gana el corto
