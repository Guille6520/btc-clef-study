"""Sin fuga temporal y estado anonimizado."""
import json
import re

import numpy as np
import pandas as pd

from clefbt.features import FEATURE_COLS, build_state, compute_features, valid_rows


def _random_walk(n=900, seed=0):
    rng = np.random.default_rng(seed)
    close = 30_000 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.003, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.003, n))
    ts = pd.date_range("2022-01-01", periods=n, freq="h")
    return pd.DataFrame({"ts": ts, "open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.uniform(100, 200, n)})


def _funding(c):
    ts = c["ts"][c["ts"].dt.hour % 8 == 0]
    return pd.DataFrame({"ts": ts, "rate": np.linspace(-1e-4, 1e-4, len(ts))})


def test_features_do_not_use_future_candles():
    c = _random_walk()
    t = 700
    base = compute_features(c, _funding(c))

    future_changed = c.copy()
    rng = np.random.default_rng(1)
    for col in ("open", "high", "low", "close", "volume"):
        future_changed.loc[t + 1:, col] = future_changed.loc[t + 1:, col] * rng.uniform(0.5, 2.0, len(c) - t - 1)
    altered = compute_features(future_changed, _funding(future_changed))

    # todo lo hasta la vela t debe ser idéntico, incluidos NaN
    pd.testing.assert_frame_equal(base.iloc[:t + 1], altered.iloc[:t + 1])


def test_state_is_anonymised_and_valid_json():
    c = _random_walk()
    f = valid_rows(compute_features(c, _funding(c)))
    assert len(f) > 100
    state = build_state(f.iloc[-1])
    data = json.loads(state)

    assert not re.search(r"20\d\d", state), "el estado no debe contener años/fechas"
    assert not any(k in data for k in ("ts", "date", "time", "timestamp"))

    numbers = []

    def walk(x):
        if isinstance(x, (int, float)):
            numbers.append(x)
        elif isinstance(x, list):
            for y in x:
                walk(y)
        elif isinstance(x, dict):
            for y in x.values():
                walk(y)

    walk(data)
    assert numbers and max(abs(v) for v in numbers) < 1000, "no debe haber precios absolutos (BTC cotiza por encima)"


def test_valid_rows_requires_all_feature_columns():
    c = _random_walk()
    f = valid_rows(compute_features(c, _funding(c)))
    assert not f[FEATURE_COLS].isna().any().any()
