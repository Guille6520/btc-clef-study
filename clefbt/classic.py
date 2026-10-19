"""Benchmark de ML clásico con validación purgada (walk-forward por años).

Compara, sobre las mismas velas y los mismos costes que el estudio de Clef:
  * heurísticas simples (tendencia, reversión), regresión logística y gradient boosting;
  * tres objetivos: dirección a 24 h, TP-antes-que-SL (largo y corto) y volatilidad futura.

Protocolo (fijado antes de ver resultados):
  * Se entrena con todo lo anterior a cada año de test menos un embargo de `embargo_hours`, para que ninguna
    etiqueta de entrenamiento (ventana de 24 h) solape con el periodo de test.
  * Hiperparámetros fijos y conservadores: no se ajustan mirando el test.
  * Reserva ciega: nada a partir de `holdout_start` se usa hasta `bench --final`.
  * Costes y funding idénticos a los del resto del proyecto.
"""
import io
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data
from .config import REPORTS_DIR
from .evaluate import auc, block_bootstrap_ci, brier, logloss, simulate, summarize_trades
from .features import FEATURE_COLS, compute_features

BASE = [c for c in FEATURE_COLS if not c.startswith("funding")]
FUND = ["funding_last_bps", "funding_24h_bps", "funding_7d_bps"]
VOL = ["rv24", "rv168", "rv_ratio", "park24"]
STRUCT = ["ret_168", "dist_hi_24", "dist_lo_24", "dist_hi_72", "dist_lo_72", "dist_hi_168", "dist_lo_168", "vol_z168"]
CAL = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
ALL = BASE + FUND + VOL + STRUCT + CAL
PERP = ["taker_ratio_z168", "taker_ratio_24", "premium_bps", "premium_24h", "premium_chg_24", "trades_z168"]
OI = ["oi_chg_24", "oi_chg_72", "oi_chg_168", "ls_top_pos", "ls_all", "taker_ls_log", "oi_price_div"]
IV = ["dvol", "dvol_chg_24", "iv_rv_spread"]
EXT = ["fng", "vix", "vix_chg_5", "nasdaq_ret_1", "nasdaq_ret_5", "dxy_ret_5", "us10y", "us10y_chg_5", "spx_ret_5"]

FEATURE_SETS = {"base": BASE, "base+fund": BASE + FUND, "todas": ALL}
# Conjuntos con datos de derivados: se comparan sobre el periodo COMÚN (solo donde todos tienen datos)
DERIV_SETS = {"v2": ALL, "v2+perp": ALL + PERP, "v2+perp+oi": ALL + PERP + OI, "v2+perp+oi+iv": ALL + PERP + OI + IV,
              "v2+ext": ALL + EXT, "todo": ALL + PERP + OI + IV + EXT}


# ---------- datos ----------
def features_v2(c, fund, atr_period=14):
    f = compute_features(c, fund, atr_period)
    c = c.sort_values("ts").set_index("ts")
    close, high, low, vol = c["close"], c["high"], c["low"], c["volume"]
    logret = np.log(close).diff()
    f["rv24"] = logret.rolling(24).std() * 100
    f["rv168"] = logret.rolling(168).std() * 100
    f["rv_ratio"] = f["rv24"] / f["rv168"]
    f["park24"] = np.sqrt((np.log(high / low) ** 2 / (4 * np.log(2))).rolling(24).mean()) * 100
    f["ret_168"] = (close - close.shift(168)) / f["atr"]
    for w in (24, 72, 168):
        f[f"dist_hi_{w}"] = (close - high.rolling(w).max()) / f["atr"]
        f[f"dist_lo_{w}"] = (close - low.rolling(w).min()) / f["atr"]
    f["vol_z168"] = (vol - vol.rolling(168).mean()) / vol.rolling(168).std()
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * f.index.hour / 24), np.cos(2 * np.pi * f.index.hour / 24)
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * f.index.dayofweek / 7), np.cos(2 * np.pi * f.index.dayofweek / 7)
    if fund is not None and len(fund):
        fr = fund.drop_duplicates("ts").set_index("ts")["rate"]
        settle = pd.Series(f.index, index=f.index).map(fr).fillna(0.0)
        f["funding_7d_bps"] = settle.rolling(168).sum() * 1e4
    else:
        f["funding_7d_bps"] = np.nan
    return f


def add_ext_features(f, con, symbol):
    """Añade derivados y contexto. Cada valor es el último conocido antes del cierre de la vela; los diarios, con 1 día de retraso."""
    idx = f.index
    close = f["close"]
    perp = con.execute("SELECT ts, taker_buy_ratio, trades, premium FROM perp_1h WHERE symbol=? ORDER BY ts", [symbol]).df()
    if len(perp):
        perp = perp.set_index("ts").reindex(idx)
        tb = perp["taker_buy_ratio"]
        f["taker_ratio_z168"] = (tb - tb.rolling(168).mean()) / tb.rolling(168).std()
        f["taker_ratio_24"] = tb.rolling(24).mean()
        pm = perp["premium"] * 1e4
        f["premium_bps"], f["premium_24h"] = pm, pm.rolling(24).mean()
        f["premium_chg_24"] = pm - pm.shift(24)
        tr = perp["trades"]
        f["trades_z168"] = (tr - tr.rolling(168).mean()) / tr.rolling(168).std()
    oi = con.execute("SELECT ts, oi, ls_top_pos, ls_all, taker_ls_vol FROM oi_1h WHERE symbol=? ORDER BY ts", [symbol]).df()
    if len(oi):
        oi = oi.set_index("ts").reindex(idx).ffill(limit=6)
        for w in (24, 72, 168):
            f[f"oi_chg_{w}"] = oi["oi"] / oi["oi"].shift(w) - 1
        f["ls_top_pos"], f["ls_all"] = oi["ls_top_pos"], oi["ls_all"]
        f["taker_ls_log"] = np.log(oi["taker_ls_vol"].where(oi["taker_ls_vol"] > 0))
        f["oi_price_div"] = f["oi_chg_24"] - (close / close.shift(24) - 1)
    dv = con.execute("SELECT ts, dvol FROM dvol_1h ORDER BY ts").df()
    if len(dv) and symbol == "BTC":
        d = dv.set_index("ts")["dvol"].reindex(idx).ffill(limit=3)
        f["dvol"], f["dvol_chg_24"] = d, d - d.shift(24)
        f["iv_rv_spread"] = d - f["rv168"] * np.sqrt(24 * 365)
    de = con.execute("SELECT name, date, value FROM daily_ext").df()
    if len(de):
        piv = de.pivot(index="date", columns="name", values="value")
        piv.index = pd.to_datetime(piv.index)
        piv = piv.reindex(pd.date_range(piv.index.min(), piv.index.max())).ffill()
        feats = pd.DataFrame(index=piv.index)
        feats["fng"] = piv.get("fng")
        feats["vix"], feats["vix_chg_5"] = piv.get("VIXCLS"), piv.get("VIXCLS") - piv.get("VIXCLS").shift(5)
        feats["nasdaq_ret_1"], feats["nasdaq_ret_5"] = piv["NASDAQCOM"].pct_change(1), piv["NASDAQCOM"].pct_change(5)
        feats["dxy_ret_5"] = piv["DTWEXBGS"].pct_change(5)
        feats["us10y"], feats["us10y_chg_5"] = piv["DGS10"], piv["DGS10"] - piv["DGS10"].shift(5)
        feats["spx_ret_5"] = piv["SP500"].pct_change(5)
        day = idx.normalize() - pd.Timedelta(days=1)            # el dato del día D-1 para todas las horas del día D
        look = feats.reindex(day)
        look.index = idx
        for col in feats.columns:
            f[col] = look[col]
    return f


def build_dataset(con, cfg, symbol, ext=False):
    H = cfg["labeling"]["horizon_hours"]
    cost = cfg["costs"]["round_trip_cost_frac"]
    c = data.load_candles(con, symbol)
    f = features_v2(c, data.load_funding(con, symbol), cfg["labeling"]["atr_period"])
    cs = c.sort_values("ts").set_index("ts")
    o, high, low = cs["open"], cs["high"], cs["low"]
    entry, exit_ = o.shift(-1), o.shift(-(H + 1))
    idx = pd.Series(cs.index, index=cs.index)
    ok = (idx.shift(-(H + 1)) - idx) == pd.Timedelta(hours=H + 1)

    f["up"] = (exit_ > entry).astype(float).where(ok & exit_.notna())
    f["ret_fwd_atr"] = ((exit_ - entry) / f["atr"]).where(ok)
    fwd_hi = high.shift(-1).rolling(H).max().shift(-(H - 1))
    fwd_lo = low.shift(-1).rolling(H).min().shift(-(H - 1))
    f["log_range_fwd"] = np.log(((fwd_hi - fwd_lo) / f["atr"]).where(ok).clip(lower=1e-3))
    f["cost_atr"] = cost / (f["atr_pct"] / 100)

    lab = con.execute(
        "SELECT ts, direction, y_pess, r_gross, funding_frac, risk_frac, exit_ts FROM labels "
        "WHERE symbol=? AND sl_atr=? AND tp_atr=? AND horizon_h=?",
        [symbol, cfg["labeling"]["main_sl_atr"], cfg["labeling"]["main_tp_atr"], H]).df()
    for d in ("long", "short"):
        L = lab[lab["direction"] == d].set_index("ts")
        f[f"y_tp_{d}"] = L["y_pess"].astype(float)
        f[f"r_{d}"] = L["r_gross"] - (cost + L["funding_frac"]) / L["risk_frac"]
        f[f"exit_{d}"] = L["exit_ts"]
    if ext:
        f = add_ext_features(f, con, symbol)
    f = f.replace([np.inf, -np.inf], np.nan)         # divisiones por cero en ratios => dato ausente, no infinito
    return f.dropna(subset=ALL + ["atr"])


# ---------- modelos ----------
def make_logit():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500))


def make_hgb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=200,
                                          l2_regularization=1.0, random_state=0)


def make_ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=100.0))


def make_hgb_reg():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=200,
                                         l2_regularization=1.0, random_state=0)


CLASSIFIERS = {"logit": make_logit, "hgb": make_hgb}
HEURISTICS = {"tendencia(dist_ema50)": lambda d: d["dist_ema50"], "reversión(-rsi)": lambda d: -d["rsi"],
              "momentum(ret_24)": lambda d: d["ret_24"]}


def walk_forward(df, cols, target, factory, bench, regression=False, until=None):
    """Predicciones fuera de muestra, año a año, con embargo. Devuelve (serie de predicciones, serie de P base)."""
    holdout = pd.Timestamp(bench["holdout_start"])
    end_all = until or holdout
    pred = pd.Series(np.nan, index=df.index)
    const = pd.Series(np.nan, index=df.index)
    for yr in range(bench["first_test_year"], end_all.year + 1):
        t0, t1 = pd.Timestamp(f"{yr}-01-01"), min(pd.Timestamp(f"{yr + 1}-01-01"), end_all)
        if t0 >= t1:
            continue
        tr = df[(df.index < t0 - pd.Timedelta(hours=bench["embargo_hours"])) & (df.index < holdout)].dropna(subset=cols + [target])
        te = df[(df.index >= t0) & (df.index < t1)].dropna(subset=cols)
        if len(tr) < 2000 or te.empty:
            continue
        m = factory()
        m.fit(tr[cols].to_numpy(), tr[target].to_numpy())
        pred.loc[te.index] = m.predict(te[cols].to_numpy()) if regression else m.predict_proba(te[cols].to_numpy())[:, 1]
        const.loc[te.index] = tr[target].mean()
    return pred, const


# ---------- métricas ----------
def boot_auc(p, y, block, n, rng, q=(5, 95)):
    N = len(p)
    nb = int(np.ceil(N / block))
    out = np.empty(n)
    for b in range(n):
        s = rng.integers(0, N - block + 1, nb)
        i = np.concatenate([np.arange(a, a + block) for a in s])[:N]
        out[b] = auc(p[i], y[i])
    return np.percentile(out, q)


def r2_oos(y, p, base):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - base) ** 2))


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def common_mask(scores, const):
    """Filas OOS donde TODOS los modelos tienen predicción (para comparar conjuntos con distinta disponibilidad)."""
    m = const.notna()
    for name, s in scores.items():
        if name.split(" ")[0] in CLASSIFIERS or name.split(" ")[0] in ("ridge", "hgb"):
            m &= s.notna()
    return m


def _cls_table(df, scores, target, const, bench, rng, title, mask=None):
    rows = []
    sub = df[target].notna() & const.notna()
    if mask is not None:
        sub &= mask
    y = df.loc[sub, target].to_numpy()
    yrs = df.index[sub].year.to_numpy()
    for name, s in scores.items():
        p = s[sub].to_numpy()
        ok = ~np.isnan(p)
        if ok.sum() < 500:
            continue
        lo, hi = boot_auc(p[ok], y[ok], bench["block_hours"], bench["n_bootstrap"], rng)
        row = {"modelo": name, "n": int(ok.sum()), "AUC": round(auc(p[ok], y[ok]), 4), "IC90": f"[{lo:.3f}, {hi:.3f}]"}
        if name in CLASSIFIERS:
            pc = const[sub].to_numpy()
            row["Brier vs base"] = f"{(1 - brier(p[ok], y[ok]) / brier(pc[ok], y[ok])) * 100:+.2f}%"
            row["LogLoss"] = round(logloss(p[ok], y[ok]), 4)
        for yr in sorted(set(yrs[ok])):
            m = ok & (yrs == yr)
            row[str(yr)] = round(auc(p[m], y[m]), 3)
        rows.append(row)
    print(f"\n--- {title} ---")
    print(pd.DataFrame(rows).fillna("").to_string(index=False))


def run_bench(con, cfg, symbol, final=False, derivs=False):
    bench = dict(cfg["bench"])
    rng = np.random.default_rng(0)
    sets = DERIV_SETS if derivs else FEATURE_SETS
    if derivs:
        bench["first_test_year"] = bench["first_test_year_derivs"]
    last_set = list(sets)[-1]
    sim_models = [f"hgb [{s}]" for s in sets] + [f"logit [{last_set}]"]
    df = build_dataset(con, cfg, symbol, ext=derivs)
    holdout = pd.Timestamp(bench["holdout_start"])
    until = None
    if final:
        until = df.index.max() + pd.Timedelta(hours=1)
    else:
        df = df[df.index < holdout]           # la reserva ciega ni se carga
    print(f"=== BENCHMARK ML CLÁSICO | {symbol} | {df.index.min()} -> {df.index.max()} | {len(df)} velas | "
          f"{'RESERVA CIEGA (final)' if final else f'sin reserva ciega (>= {holdout.date()} no se toca)'} ===")
    print(f"Test walk-forward por años desde {bench['first_test_year']}, embargo {bench['embargo_hours']} h, "
          f"coste ida y vuelta {cfg['costs']['round_trip_cost_frac']:.2%} + funding real.")

    # ---- 1) DIRECCIÓN a 24 h ----
    scores, const = {}, None
    for fs_name, cols in sets.items():
        for mname, fac in CLASSIFIERS.items():
            p, c0 = walk_forward(df, cols, "up", fac, bench, until=until)
            scores[f"{mname} [{fs_name}]"], const = p, c0
    for hname, h in HEURISTICS.items():
        scores[hname] = h(df)
    dmask = common_mask(scores, const) if derivs else None
    if derivs:
        print(f"\n[Periodo COMÚN a todos los modelos: {df.index[dmask][0]} -> {df.index[dmask][-1]}, {int(dmask.sum())} velas]")
    _cls_table(df, scores, "up", const, bench, rng, "DIRECCIÓN a 24 h (¿precio de salida > entrada?)", dmask)
    pooled = df.loc[const.notna() & (dmask if dmask is not None else True), "up"]
    print(f"tasa base OOS: {pooled.mean():.3f}")

    # operar el signo con el mejor clasificador de cada familia, en ATR
    print("\nOperar el signo (largo si P>0,5+m, corto si P<0,5-m), neto de costes, en ATR:")
    rows = []
    for name in sim_models:
        p = scores[name]
        for m in bench["margins_dir"]:
            sel = (p - 0.5).abs() >= max(m, 1e-9) if m > 0 else p.notna()
            sel &= const.notna()
            if dmask is not None:
                sel &= dmask
            sel &= df["ret_fwd_atr"].notna()
            g = (np.sign(p - 0.5) * df["ret_fwd_atr"])[sel]
            net = g - df["cost_atr"][sel]
            if len(net) < 50:
                continue
            se = net.std(ddof=1) / np.sqrt(len(net))
            rows.append({"modelo": name, "margen": m, "n": len(net), "bruta_ATR": round(g.mean(), 3),
                         "neta_ATR": round(net.mean(), 3), "IC90": f"[{net.mean() - 1.645 * se:.3f}, {net.mean() + 1.645 * se:.3f}]"})
    print(pd.DataFrame(rows).to_string(index=False))
    print("(las ventanas de 24 h se solapan: el IC real es mayor que el nominal)")

    # ---- 2) TP antes que SL, largo y corto ----
    tp_scores = {}
    for d in ("long", "short"):
        scs, c0 = {}, None
        for fs_name, cols in sets.items():
            for mname, fac in CLASSIFIERS.items():
                p, c0 = walk_forward(df, cols, f"y_tp_{d}", fac, bench, until=until)
                scs[f"{mname} [{fs_name}]"] = p
        tp_scores[d] = (scs, c0)
        for hname, h in HEURISTICS.items():
            scs[hname] = h(df) if d == "long" else -h(df)
        tp_scores[d] = (scs, c0, common_mask(scs, c0) if derivs else None)
        _cls_table(df, scs, f"y_tp_{d}", c0, bench, rng, f"TP antes que SL ({d.upper()}; SL 1,5 ATR, TP 3 ATR, 24 h)",
                   tp_scores[d][2])

    print("\n--- SIMULACIÓN TP/SL (una posición a la vez, costes y funding incluidos; elige el lado de mayor P) ---")
    oos = tp_scores["long"][1].notna()
    if derivs:
        oos &= tp_scores["long"][2] & tp_scores["short"][2]
    oos &= df["r_long"].notna() & df["r_short"].notna()      # huecos de velas del exchange => sin etiqueta
    d_oos = df[oos]
    ts = d_oos.index.to_numpy()
    rl, rs = d_oos["r_long"].to_numpy(), d_oos["r_short"].to_numpy()
    el, es = d_oos["exit_long"].to_numpy(), d_oos["exit_short"].to_numpy()
    print(f"referencias (todas las velas OOS, solapadas): siempre largo {np.nanmean(rl):+.3f} R | siempre corto {np.nanmean(rs):+.3f} R")
    for name in sim_models:
        pl = tp_scores["long"][0][name][oos].to_numpy()
        ps = tp_scores["short"][0][name][oos].to_numpy()
        rows = []
        for thr in bench["thresholds_tp"]:
            tr = simulate(ts, pl, ps, rl, rs, el, es, thr)
            s = summarize_trades(tr)
            lo, hi = block_bootstrap_ci(tr, 5, 1000, rng)
            perm = []
            for _ in range(bench["n_permutations"] if len(tr) else 0):
                i = rng.permutation(len(ts))
                t2 = simulate(ts, pl[i], ps[i], rl, rs, el, es, thr)
                perm.append(t2.mean() if len(t2) else np.nan)
            pv = float(np.nanmean(np.array(perm) >= s["media_R"])) if perm and not np.isnan(s["media_R"]) else np.nan
            rows.append({"umbral": thr, "n": s["n"], "media_R": round(s["media_R"], 3) if s["n"] else np.nan,
                         "suma_R": round(s["suma_R"], 1), "win": round(s["win"], 3) if s["n"] else np.nan,
                         "IC90": f"[{lo:.2f}, {hi:.2f}]" if s["n"] else "", "p_perm": round(pv, 3) if perm else np.nan})
        print(f"\n[{name}]")
        print(pd.DataFrame(rows).to_string(index=False))

    # ---- 3) VOLATILIDAD futura ----
    print("\n--- VOLATILIDAD futura: log(rango de las próximas 24 h / ATR) ---")
    yv = df["log_range_fwd"]
    vpreds = {}
    for fs_name, cols in sets.items():
        for mname, fac in (("ridge", make_ridge), ("hgb", make_hgb_reg)):
            vpreds[f"{mname} [{fs_name}]"] = walk_forward(df, cols, "log_range_fwd", fac, bench, regression=True, until=until)
    vmask = yv.notna()
    if derivs:
        for p, _ in vpreds.values():
            vmask &= p.notna()
    rows = []
    for name, (p, c0) in vpreds.items():
        ok = p.notna() & vmask
        yrs = df.index[ok].year
        row = {"modelo": name, "n": int(ok.sum()),
               "R2 OOS": round(r2_oos(yv[ok].to_numpy(), p[ok].to_numpy(), c0[ok].to_numpy()), 4),
               "Spearman": round(spearman(p[ok], yv[ok]), 3)}
        for yr in sorted(set(yrs)):
            m = ok & (df.index.year == yr)
            row[str(yr)] = round(spearman(p[m], yv[m]), 3)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print("(R2 OOS frente a predecir la media de entrenamiento; Spearman por año en las columnas de años)")
    print("\nNOTAS: IC por bootstrap de bloques semanales; p_perm = fracción de barajados con media >= la real.")
    print("NOTAS: hiperparámetros fijos, no ajustados con el test. Cada variante probada aumenta el riesgo de falso positivo.")


def run_bench_to_file(con, cfg, symbol, final=False, derivs=False):
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_bench(con, cfg, symbol, final, derivs)
    txt = buf.getvalue()
    print(txt)
    REPORTS_DIR.mkdir(exist_ok=True)
    kind = ("final" if final else "cv") + ("_derivs" if derivs else "")
    path = REPORTS_DIR / f"bench_{kind}_{symbol}_{datetime.now():%Y%m%d_%H%M%S}.txt"
    path.write_text(txt, encoding="utf-8")
    print(f"Informe guardado en {path}")
