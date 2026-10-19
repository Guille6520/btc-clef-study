"""Estrategias de baja frecuencia (velas diarias) y análisis del carry de funding.

Todo se decide al cierre del día D (UTC) y se ejecuta a ese precio de cierre, con coste y slippage, y el
resultado se cobra el día D+1. Las reglas son estándar y con parámetros típicos FIJADOS de antemano (no se
ajustan): al ser reglas sin entrenamiento, el riesgo principal es el de probar demasiadas variantes, así que la
lista es corta y se declara entera. La reserva ciega (>= holdout_start) no se toca hasta `daily --final`.
"""
import io
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np
import pandas as pd

from . import data
from .config import REPORTS_DIR

SIDE_COST = 0.00055        # 0,045 % taker + 0,01 % slippage por lado y por unidad de nominal
DAYS = 365


def daily_frame(con, symbol):
    c = data.load_candles(con, symbol).set_index("ts")
    d = c.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    fund = data.load_funding(con, symbol).drop_duplicates("ts").set_index("ts")["rate"]
    d["funding"] = fund.resample("1D").sum().reindex(d.index).fillna(0.0)    # suma de las liquidaciones del día
    d["ret"] = d["close"].pct_change()
    return d


def positions(d):
    close, ret = d["close"], d["ret"]
    sma50, sma200 = close.rolling(50).mean(), close.rolling(200).mean()
    lev = (0.40 / (ret.rolling(30).std() * np.sqrt(DAYS))).clip(0, 2)
    return {
        "comprar y mantener": pd.Series(1.0, index=d.index),
        "tendencia SMA200 (largo/plano)": (close > sma200).astype(float).where(sma200.notna()),
        "tendencia SMA200 (largo/corto)": pd.Series(np.where(close > sma200, 1.0, -1.0), index=d.index).where(sma200.notna()),
        "cruce SMA50/200 (largo/plano)": (sma50 > sma200).astype(float).where(sma200.notna()),
        "momentum 90 d (largo/plano)": (close / close.shift(90) - 1 > 0).astype(float).where(close.shift(90).notna()),
        "momentum 180 d (largo/plano)": (close / close.shift(180) - 1 > 0).astype(float).where(close.shift(180).notna()),
        "vol-objetivo 40 % (siempre largo)": lev,
        "tendencia SMA200 + vol-objetivo 40 %": ((close > sma200).astype(float) * lev).where(sma200.notna() & lev.notna()),
    }


def pnl_series(d, pos):
    p = pos.shift(1)                                    # lo decidido ayer se cobra hoy
    gross = p * d["ret"]
    funding = -p * d["funding"]                         # el largo paga funding positivo, el corto lo cobra
    cost = -(p - p.shift(1)).abs() * SIDE_COST
    return (gross + funding + cost).dropna()


def metrics(x):
    eq = (1 + x).cumprod()
    years = len(x) / DAYS
    sd = x.std()
    return {"CAGR": eq.iloc[-1] ** (1 / years) - 1, "vol": sd * np.sqrt(DAYS),
            "Sharpe": x.mean() / sd * np.sqrt(DAYS) if sd > 0 else np.nan,
            "maxDD": (eq / eq.cummax() - 1).min(), "dias": len(x)}


def boot_sharpe(x, block=20, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = x.to_numpy()
    N = len(v)
    nb = int(np.ceil(N / block))
    out = np.empty(n)
    for b in range(n):
        s = rng.integers(0, N - block + 1, nb)
        w = np.concatenate([v[a:a + block] for a in s])[:N]
        out[b] = w.mean() / w.std() * np.sqrt(DAYS) if w.std() > 0 else np.nan
    return np.nanpercentile(out, [5, 95])


def run_daily(con, cfg, symbol, final=False):
    holdout = pd.Timestamp(cfg["bench"]["holdout_start"])
    d = daily_frame(con, symbol)
    if not final:
        d = d[d.index < holdout]
    print(f"=== ESTRATEGIAS DIARIAS | {symbol} | {d.index.min().date()} -> {d.index.max().date()} | {len(d)} días | "
          f"{'RESERVA CIEGA incluida (final)' if final else f'sin reserva ciega (>= {holdout.date()} no se toca)'} ===")
    print(f"Decisión al cierre del día, ejecución a ese cierre; coste {SIDE_COST:.3%} por lado + funding real. "
          f"8 variantes, todas declaradas (parámetros típicos, no ajustados).\n")

    pos = positions(d)
    pnl = {name: pnl_series(d, p) for name, p in pos.items()}
    common = None
    for x in pnl.values():
        common = x.index if common is None else common.intersection(x.index)
    bh = pnl["comprar y mantener"].loc[common]

    rows = []
    for name, x in pnl.items():
        x = x.loc[common]
        m = metrics(x)
        lo, hi = boot_sharpe(x)
        p = pos[name].loc[common]
        diff = (x - bh)
        rows.append({"estrategia": name, "CAGR": f"{m['CAGR']:+.1%}", "vol": f"{m['vol']:.0%}", "Sharpe": round(m["Sharpe"], 2),
                     "IC90 Sharpe": f"[{lo:.2f}, {hi:.2f}]", "maxDD": f"{m['maxDD']:.0%}",
                     "en mercado": f"{(p.abs() > 0).mean():.0%}",
                     "rot./año": round(float((p.diff().abs().sum()) / (len(x) / DAYS)), 1)})
    print(f"--- PERIODO COMÚN {common.min().date()} -> {common.max().date()} ({len(common)} días) ---")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n--- Por periodo (Sharpe) ---")
    bounds = [("2018-2019", "2018-01-01", "2020-01-01"), ("2020-2021", "2020-01-01", "2022-01-01"),
              ("2022", "2022-01-01", "2023-01-01"), ("2023-2025", "2023-01-01", "2030-01-01")]
    rows = []
    for name, x in pnl.items():
        r = {"estrategia": name}
        for label, a, b in bounds:
            s = x.loc[common][(x.loc[common].index >= a) & (x.loc[common].index < b)]
            r[label] = round(metrics(s)["Sharpe"], 2) if len(s) > 60 else ""
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n--- Retorno por año ---")
    rows = []
    for name, x in pnl.items():
        y = x.loc[common]
        r = {"estrategia": name}
        for yr, g in y.groupby(y.index.year):
            r[str(yr)] = f"{(1 + g).prod() - 1:+.0%}"
        rows.append(r)
    print(pd.DataFrame(rows).fillna("").to_string(index=False))

    # ---- carry de funding ----
    print("\n--- CARRY DE FUNDING (largo spot + corto perpetuo, 1x, sin predecir nada) ---")
    f = d["funding"]
    rows = []
    for yr, g in f.groupby(f.index.year):
        rows.append({"año": yr, "funding anualizado": f"{g.mean() * DAYS:+.1%}", "días con funding < 0": f"{(g < 0).mean():.0%}",
                     "peor mes": f"{g.rolling(30).sum().min():+.2%}"})
    print(pd.DataFrame(rows).to_string(index=False))
    carry = f - 2 * SIDE_COST * 2 / (len(f))          # coste de abrir y cerrar las dos patas, repartido
    eq = (1 + carry).cumprod()
    yrs = len(carry) / DAYS
    print(f"\nCarry acumulado en el periodo: CAGR {eq.iloc[-1] ** (1 / yrs) - 1:+.1%}, "
          f"Sharpe {carry.mean() / carry.std() * np.sqrt(DAYS):.2f}, peor caída del capital en carry {(eq / eq.cummax() - 1).min():.1%}")
    print("Ojo: no incluye la base spot-perpetuo en las salidas, ni el capital extra para margen (el corto a 1x se liquida si BTC se duplica),")
    print("ni el riesgo de exchange / contrato. Rendimiento sobre el nominal de UNA pata; sobre el capital total aproximado es menor.")
    print("\nNOTAS: ~8 años de datos diarios: el IC del Sharpe es ancho (±0,5 aprox.). 2017-2021 fue un mercado alcista excepcional;")
    print("       ninguna conclusión sobre el futuro. Menos variantes y menos rotación = menos riesgo de sobreajuste.")


def run_daily_to_file(con, cfg, symbol, final=False):
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_daily(con, cfg, symbol, final)
    txt = buf.getvalue()
    print(txt)
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"daily_{'final' if final else 'cv'}_{symbol}_{datetime.now():%Y%m%d_%H%M%S}.txt"
    path.write_text(txt, encoding="utf-8")
    print(f"Informe guardado en {path}")
