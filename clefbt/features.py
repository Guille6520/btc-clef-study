"""Features y estado para Clef.

Convención temporal: la fila con ts = apertura de la vela T describe el mercado en su CIERRE (T+1h).
Todo lo calculado aquí usa solo datos hasta ese cierre. La entrada simulada es la apertura de T+1h.
El estado que se envía a Clef está anonimizado: sin fechas, sin nombre de moneda y con precios
normalizados en unidades de ATR, para reducir la memorización del histórico por parte del modelo.
"""
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "atr_pct", "vol_regime", "rsi", "dist_ema20", "dist_ema50", "dist_ema200",
    "ret_1", "ret_4", "ret_24", "ret_72", "range_pos_24", "vol_ratio",
    "funding_last_bps", "funding_24h_bps",
]
N_LAST = 6


def _wilder(x, n):
    return x.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def compute_features(c, funding=None, atr_period=14):
    """c: DataFrame [ts, open, high, low, close, volume] ordenado. Devuelve DataFrame indexado por ts."""
    c = c.sort_values("ts").reset_index(drop=True)
    close, high, low, vol = c["close"], c["high"], c["low"], c["volume"]
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, atr_period)

    f = pd.DataFrame({"ts": c["ts"], "atr": atr, "close": close})
    f["atr_pct"] = atr / close * 100
    f["vol_regime"] = f["atr_pct"] / f["atr_pct"].rolling(500, min_periods=500).mean()

    d = close.diff()
    up, dn = _wilder(d.clip(lower=0), 14), _wilder((-d).clip(lower=0), 14)
    f["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))

    for span in (20, 50, 200):
        ema = close.ewm(span=span, adjust=False, min_periods=span).mean()
        f[f"dist_ema{span}"] = (close - ema) / atr
    for k in (1, 4, 24, 72):
        f[f"ret_{k}"] = (close - close.shift(k)) / atr
    lo24, hi24 = low.rolling(24).min(), high.rolling(24).max()
    f["range_pos_24"] = (close - lo24) / (hi24 - lo24).replace(0, np.nan)
    f["vol_ratio"] = vol / vol.rolling(24).mean()

    # funding: se liquida en horas puntuales; la tasa de la hora T ya se conoce al cierre de la vela T
    if funding is not None and len(funding):
        fr = funding.drop_duplicates("ts").set_index("ts")["rate"]
        grid = f["ts"]
        settle = grid.map(fr).fillna(0.0).to_numpy()
        last = grid.map(fr).ffill().to_numpy()
        f["funding_last_bps"] = last * 1e4
        f["funding_24h_bps"] = pd.Series(settle).rolling(24, min_periods=24).sum().to_numpy() * 1e4
    else:
        f["funding_last_bps"] = np.nan
        f["funding_24h_bps"] = np.nan

    # últimas N velas relativas al último cierre, en unidades de ATR
    for k in range(N_LAST):
        for name, col in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close")):
            f[f"lc{k}_{name}"] = (c[col].shift(k) - close) / atr
    return f.set_index("ts")


def valid_rows(f, need_funding=True):
    cols = [x for x in FEATURE_COLS if need_funding or not x.startswith("funding")]
    return f.dropna(subset=cols + ["atr"])


def build_state(row):
    """Estado anonimizado (JSON compacto) para una fila de features."""
    import json

    def r(x, n=2):
        return None if pd.isna(x) else round(float(x), n)

    last = []
    for k in range(N_LAST):
        last.append([r(row[f"lc{k}_{n}"]) for n in ("o", "h", "l", "c")])
    state = {
        "note": "BTC perpetual futures, 1h candles, decision at candle close. Prices are in ATR units "
                "relative to the latest close (no dates or absolute prices).",
        "atr_pct_of_price": r(row["atr_pct"], 3),
        "volatility_vs_500h_avg": r(row["vol_regime"]),
        "rsi14": r(row["rsi"], 1),
        "close_minus_ema20_atr": r(row["dist_ema20"]),
        "close_minus_ema50_atr": r(row["dist_ema50"]),
        "close_minus_ema200_atr": r(row["dist_ema200"]),
        "return_1h_atr": r(row["ret_1"]),
        "return_4h_atr": r(row["ret_4"]),
        "return_24h_atr": r(row["ret_24"]),
        "return_72h_atr": r(row["ret_72"]),
        "position_in_24h_range_0_1": r(row["range_pos_24"]),
        "volume_vs_24h_avg": r(row["vol_ratio"]),
        "funding_last_bps": r(row["funding_last_bps"], 2),
        "funding_24h_sum_bps": r(row["funding_24h_bps"], 2),
        "last_6_candles_ohlc_atr_oldest_last": last[::-1],
    }
    return json.dumps(state, separators=(",", ":"))
