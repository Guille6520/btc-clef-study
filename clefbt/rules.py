"""Regla contraria por RSI: una hipótesis concreta salida del benchmark (la reversión por RSI fue el mejor AUC direccional).

Se trata como hipótesis EXPLORATORIA: surgió de mirar resultados, así que no vale como prueba. Lo único que la validaría es la
reserva ciega (`contrarian --final`), que se evalúa UNA sola vez con la regla ya fijada.

Regla: si RSI(14) de 1 h < umbral inferior => largo; > umbral superior => corto. Entrada en la apertura siguiente, salida 24 h
después, una posición a la vez. Resultado en múltiplos de ATR (bruto y neto de costes bajo tres escenarios de comisiones).
"""
import io
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np
import pandas as pd

from .classic import build_dataset
from .config import REPORTS_DIR

BANDS = [(30, 70), (25, 75), (20, 80), (15, 85)]
COST_SCENARIOS = {"taker 0,11 %": 0.0011, "mixto 0,07 %": 0.0007, "maker 0,04 %": 0.0004}


def take_trades(sig, ret, hold):
    """Una posición a la vez: tras entrar, no se mira otra señal hasta que sale."""
    taken, i, n = [], 0, len(sig)
    while i < n:
        if sig[i] != 0 and np.isfinite(ret[i]):
            taken.append(i)
            i += hold
        else:
            i += 1
    return np.array(taken, dtype=int)


def run_contrarian(con, cfg, symbol, final=False):
    H = cfg["labeling"]["horizon_hours"]
    holdout = pd.Timestamp(cfg["bench"]["holdout_start"])
    df = build_dataset(con, cfg, symbol)
    base_cost = cfg["costs"]["round_trip_cost_frac"]
    if final:
        df = df[df.index >= holdout]
    else:
        df = df[df.index < holdout]
    print(f"=== REGLA CONTRARIA POR RSI | {symbol} | {df.index.min()} -> {df.index.max()} | "
          f"{'RESERVA CIEGA (final, una sola vez)' if final else f'sin reserva ciega (>= {holdout.date()} no se toca)'} ===")
    print(f"Entrada en la apertura siguiente, salida a {H} h, una posición a la vez. Resultados en ATR por operación.\n")
    rsi, ret = df["rsi"].to_numpy(), df["ret_fwd_atr"].to_numpy()
    cost_base_atr = df["cost_atr"].to_numpy()                      # coste con el supuesto de config (0,11 %)
    years = df.index.year.to_numpy()

    rows = []
    for lo, hi in BANDS:
        sig = np.where(rsi < lo, 1, np.where(rsi > hi, -1, 0))
        idx = take_trades(sig, ret, H + 1)
        if len(idx) < 30:
            continue
        gross = sig[idx] * ret[idx]
        row = {"RSI <": lo, "RSI >": hi, "n": len(idx), "aciertos": f"{(gross > 0).mean():.0%}",
               "bruta ATR": round(gross.mean(), 3)}
        for name, c in COST_SCENARIOS.items():
            net = gross - cost_base_atr[idx] * (c / base_cost)
            se = net.std(ddof=1) / np.sqrt(len(net))
            row[f"neta {name}"] = f"{net.mean():+.3f} [{net.mean() - 1.645 * se:+.3f}, {net.mean() + 1.645 * se:+.3f}]"
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print("(neta = media [IC90]; con una posición a la vez las operaciones no se solapan: IC razonable)")

    lo, hi = 30, 70
    sig = np.where(rsi < lo, 1, np.where(rsi > hi, -1, 0))
    idx = take_trades(sig, ret, H + 1)
    gross = sig[idx] * ret[idx]
    net_mix = gross - cost_base_atr[idx] * (COST_SCENARIOS["mixto 0,07 %"] / base_cost)
    print(f"\n--- Por año, banda {lo}/{hi} (bruta y neta con comisión 'mixta') ---")
    rows = []
    for yr in sorted(set(years[idx])):
        m = years[idx] == yr
        rows.append({"año": yr, "n": int(m.sum()), "aciertos": f"{(gross[m] > 0).mean():.0%}",
                     "bruta ATR": round(gross[m].mean(), 3), "neta mixto": round(net_mix[m].mean(), 3)})
    print(pd.DataFrame(rows).to_string(index=False))
    for side, name in ((1, "largos (RSI bajo)"), (-1, "cortos (RSI alto)")):
        m = sig[idx] == side
        if m.sum() > 20:
            print(f"{name}: n={int(m.sum())}, bruta {gross[m].mean():+.3f} ATR, neta mixto {net_mix[m].mean():+.3f} ATR")
    print("\nNOTAS: hipótesis exploratoria (salió de mirar resultados). 4 bandas y 3 escenarios de coste = 12 celdas: no elijas la mejor.")
    print("       Las comisiones 'maker' exigen que las órdenes límite se ejecuten; aquí no se modela la selección adversa.")


def candle_dataset(con, cfg, symbol):
    """Dataset solo con velas (sin funding ni etiquetas): permite probar reglas de RSI en periodos y activos que el benchmark no vio."""
    from . import data
    from .features import compute_features
    H = cfg["labeling"]["horizon_hours"]
    c = data.load_candles(con, symbol)
    f = compute_features(c, None, cfg["labeling"]["atr_period"])
    o = c.sort_values("ts").set_index("ts")["open"]
    entry, exit_ = o.shift(-1), o.shift(-(H + 1))
    idx = pd.Series(o.index, index=o.index)
    ok = (idx.shift(-(H + 1)) - idx) == pd.Timedelta(hours=H + 1)
    f["ret_fwd_atr"] = ((exit_ - entry) / f["atr"]).where(ok)
    f["ret_pct_fwd"] = (exit_ / entry - 1).where(ok)            # retorno en % de precio de un largo
    f["cost_atr"] = cfg["costs"]["round_trip_cost_frac"] / (f["atr_pct"] / 100)
    return f.dropna(subset=["rsi", "atr", "atr_pct"])


# Regla FIJADA antes de la replicación: continuación en los extremos de RSI (la inversa de la contraria).
EXT_BANDS = [(30, 70), (25, 75), (20, 80)]
EXT_MAIN = (30, 70)


def _segment_rows(df, label, H, base_cost):
    rsi, ret = df["rsi"].to_numpy(), df["ret_fwd_atr"].to_numpy()
    cost_base = df["cost_atr"].to_numpy()
    years = df.index.year.to_numpy()
    rows = []
    for lo, hi in EXT_BANDS:
        sig = np.where(rsi < lo, -1, np.where(rsi > hi, 1, 0))            # continuación: RSI bajo => corto, alto => largo
        idx = take_trades(sig, ret, H + 1)
        if len(idx) < 30:
            rows.append({"segmento": label, "banda": f"{lo}/{hi}", "n": len(idx)})
            continue
        gross = sig[idx] * ret[idx]
        yrs = sorted(set(years[idx]))
        pos_years = sum(1 for y in yrs if gross[years[idx] == y].mean() > 0 and (years[idx] == y).sum() >= 10)
        n_years = sum(1 for y in yrs if (years[idx] == y).sum() >= 10)
        row = {"segmento": label, "banda": f"{lo}/{hi}", "n": len(idx), "aciertos": f"{(gross > 0).mean():.0%}",
               "bruta ATR": round(gross.mean(), 3), "años>0": f"{pos_years}/{n_years}"}
        for name, c in COST_SCENARIOS.items():
            net = gross - cost_base[idx] * (c / base_cost)
            se = net.std(ddof=1) / np.sqrt(len(net))
            row[f"neta {name}"] = f"{net.mean():+.3f} [{net.mean() - 1.645 * se:+.2f}, {net.mean() + 1.645 * se:+.2f}]"
        rows.append(row)
    return rows


def _robustness(df, label, H, lo, hi, slip_extra=(0.0, 0.001, 0.002), base_cost=0.0011):
    """¿Depende de pocas operaciones? ¿Qué pasa en % reales, con drawdown, y con peor deslizamiento en momentos de pánico?"""
    rsi, ret, rp = df["rsi"].to_numpy(), df["ret_fwd_atr"].to_numpy(), df["ret_pct_fwd"].to_numpy()
    cost_atr = df["cost_atr"].to_numpy()
    sig = np.where(rsi < lo, -1, np.where(rsi > hi, 1, 0))
    idx = take_trades(sig, ret, H + 1)
    if len(idx) < 30:
        return
    g = sig[idx] * ret[idx]
    pct = sig[idx] * rp[idx]
    years = (df.index[idx][-1] - df.index[idx][0]).days / 365.25
    order = np.argsort(g)[::-1]
    top10 = g[order[:10]].sum() / g.sum() if g.sum() > 0 else np.nan
    lo_cut, hi_cut = np.percentile(g, [5, 95])
    trimmed = g[(g >= lo_cut) & (g <= hi_cut)].mean()
    print(f"\n[{label}] banda {lo}/{hi}: {len(idx)} operaciones en {years:.1f} años ({len(idx) / years:.0f}/año)")
    print(f"  media bruta {g.mean():+.3f} ATR | mediana {np.median(g):+.3f} | media recortada (5 % por cada extremo) {trimmed:+.3f} "
          f"| las 10 mejores operaciones = {top10:.0%} del total bruto | peor operación {g.min():+.2f} ATR ({pct.min():+.1%} del precio)")
    for slip in slip_extra:
        net_pct = pct - base_cost - slip
        eq = np.cumprod(1 + net_pct)
        dd = (eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min()
        sharpe = net_pct.mean() / net_pct.std() * np.sqrt(len(idx) / years)
        print(f"  a 1x de nominal, comisión {base_cost:.2%} + deslizamiento extra {slip:.1%}: media {net_pct.mean():+.3%}/operación, "
              f"anual compuesto {eq[-1] ** (1 / years) - 1:+.1%}, Sharpe {sharpe:.2f}, peor caída {dd:.0%}")
    # sin el año que más aporta
    ys = df.index[idx].year.to_numpy()
    contrib = {y: g[ys == y].sum() for y in set(ys)}
    best = max(contrib, key=contrib.get)
    rest = g[ys != best]
    print(f"  quitando el año que más aporta ({best}): media bruta {rest.mean():+.3f} ATR sobre {len(rest)} operaciones")


def run_extremes(con, cfg, final=False):
    H = cfg["labeling"]["horizon_hours"]
    base_cost = cfg["costs"]["round_trip_cost_frac"]
    holdout = pd.Timestamp(cfg["bench"]["holdout_start"])
    seen_start = pd.Timestamp("2019-09-10")            # el benchmark y la regla contraria miraron BTC desde aquí hasta el holdout
    print(f"=== CONTINUACIÓN EN EXTREMOS DE RSI (regla fijada: RSI<lo => corto, RSI>hi => largo; {H} h; una posición a la vez) ===")
    print("Hipótesis nacida de ver que la regla CONTRARIA perdía en los 7 años observados. Se replica donde NO se ha mirado.\n")
    rows = []
    segments = []
    if final:
        for sym in ("BTC",):
            df = candle_dataset(con, cfg, sym)
            segments.append((f"{sym} RESERVA CIEGA (final)", df[df.index >= holdout]))
    else:
        btc = candle_dataset(con, cfg, "BTC")
        segments.append(("BTC 2017-09..2019-09 (NO visto)", btc[(btc.index >= "2017-09-01") & (btc.index < seen_start)]))
        segments.append(("BTC 2019-09..hold-out (visto)", btc[(btc.index >= seen_start) & (btc.index < holdout)]))
        for sym in ("ETH", "SOL"):
            try:
                d = candle_dataset(con, cfg, sym)
            except Exception:
                continue
            if len(d) > 5000:
                segments.append((f"{sym} 2018..hold-out (NO visto)", d[(d.index >= "2018-01-01") & (d.index < holdout)]))
    for label, df in segments:
        rows += _segment_rows(df, label, H, base_cost)
    print(pd.DataFrame(rows).fillna("").to_string(index=False))
    print("\n--- ROBUSTEZ de la banda principal (¿depende de pocas operaciones? ¿en % reales? ¿con peor ejecución?) ---")
    for label, df in segments:
        _robustness(df, label, H, EXT_MAIN[0], EXT_MAIN[1], base_cost=base_cost)
    print("\n(neta = media [IC90] en ATR por operación; 'años>0' = años con >= 10 operaciones y media bruta positiva)")
    print("NOTAS: el segmento 'visto' no es evidencia nueva. Solo cuentan los 'NO visto': otro periodo y otros activos (ETH/SOL se mueven")
    print("       con BTC, así que no son 100 % independientes). Con 3 bandas y 3 escenarios de coste: mira la banda principal 30/70.")


def run_extremes_to_file(con, cfg, final=False):
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_extremes(con, cfg, final)
    txt = buf.getvalue()
    print(txt)
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"extremes_{'final' if final else 'cv'}_{datetime.now():%Y%m%d_%H%M%S}.txt"
    path.write_text(txt, encoding="utf-8")
    print(f"Informe guardado en {path}")


def run_contrarian_to_file(con, cfg, symbol, final=False):
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_contrarian(con, cfg, symbol, final)
    txt = buf.getvalue()
    print(txt)
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"contrarian_{'final' if final else 'cv'}_{symbol}_{datetime.now():%Y%m%d_%H%M%S}.txt"
    path.write_text(txt, encoding="utf-8")
    print(f"Informe guardado en {path}")
