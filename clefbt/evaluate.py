"""Evaluación de las señales de Clef contra lo que pasó de verdad.

Qué responde:
  1. ¿Está calibrado?  (curva de fiabilidad, Brier, log-loss)
  2. ¿Discrimina?      (AUC) y ¿mejora a la tasa base y a un modelo clásico walk-forward?
  3. ¿Gana dinero?     (simulación con una posición a la vez, costes y funding, umbral por rejilla)
  4. ¿Es suerte?       (test de permutación: mismas probabilidades barajadas en el tiempo; IC por bootstrap de bloques)
"""
import io
import json
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np
import pandas as pd

from . import clef as clefmod
from . import data
from .config import REPORTS_DIR
from .features import FEATURE_COLS, compute_features, valid_rows


# ---------- métricas básicas ----------
def brier(p, y):
    return float(np.mean((p - y) ** 2))


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(p, y):
    y = np.asarray(y)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(p).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def calibration_table(p, y, bins=(0, .2, .3, .4, .5, .6, .7, .8, 1.0001)):
    cut = pd.cut(p, bins, right=False)
    t = pd.DataFrame({"p": p, "y": y, "bin": cut}).groupby("bin", observed=True).agg(
        n=("y", "size"), p_media=("p", "mean"), acierto_real=("y", "mean"))
    return t.round(3)


# ---------- carga ----------
def load_dataset(con, cfg, symbol, model, source=None, start=None, end=None):
    lab = cfg["labeling"]
    keys = clefmod.main_keys(cfg)
    calls = con.execute(
        "SELECT ts, source, probs FROM clef_calls WHERE symbol=? AND model=? AND state_version=? AND qset_version=?",
        [symbol, model, cfg["clef"]["state_version"], clefmod.qset_version(cfg)]).df()
    if calls.empty:
        raise SystemExit("No hay llamadas a Clef guardadas para esta configuración. Ejecuta 'run'.")
    pr = calls["probs"].map(json.loads)
    calls["p_long"] = pr.map(lambda d: d[keys["long"]])
    calls["p_short"] = pr.map(lambda d: d[keys["short"]])
    calls = calls.drop(columns="probs")
    if source:
        calls = calls[calls["source"] == source]
    if start:
        calls = calls[calls["ts"] >= pd.Timestamp(start)]
    if end:
        calls = calls[calls["ts"] < pd.Timestamp(end)]

    labels = con.execute(
        "SELECT * FROM labels WHERE symbol=? AND sl_atr=? AND tp_atr=? AND horizon_h=?",
        [symbol, lab["main_sl_atr"], lab["main_tp_atr"], lab["horizon_hours"]]).df()
    if labels.empty:
        raise SystemExit("No hay etiquetas. Ejecuta 'label'.")
    costs = cfg["costs"]["round_trip_cost_frac"]
    wide = calls.set_index("ts")
    for d in ("long", "short"):
        L = labels[labels["direction"] == d].set_index("ts")
        net_cost = costs / L["risk_frac"] + L["funding_frac"] / L["risk_frac"]
        wide[f"y_pess_{d}"] = L["y_pess"]
        wide[f"y_opt_{d}"] = L["y_opt"]
        wide[f"outcome_{d}"] = L["outcome"]
        wide[f"r_pess_{d}"] = L["r_gross"] - net_cost
        wide[f"r_opt_{d}"] = L["r_gross_opt"] - net_cost
        wide[f"exit_{d}"] = L["exit_ts"]
    wide = wide.dropna(subset=["y_pess_long", "y_pess_short"]).sort_index()

    f = compute_features(data.load_candles(con, symbol), data.load_funding(con, symbol), cfg["labeling"]["atr_period"])
    f = valid_rows(f)[FEATURE_COLS]
    return wide.join(f, how="inner")


# ---------- baseline clásico walk-forward ----------
def walk_forward_lr(df, direction, min_years=1, ycol=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = df[ycol or f"y_pess_{direction}"].to_numpy()
    X = df[FEATURE_COLS].to_numpy()
    years = df.index.year.to_numpy()
    pred = np.full(len(df), np.nan)
    ys = sorted(set(years))
    for yr in ys[min_years:]:
        tr, te = years < yr, years == yr
        if tr.sum() < 500 or len(np.unique(y[tr])) < 2:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.1))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return pred


# ---------- simulación ----------
def simulate(ts, p_long, p_short, r_long, r_short, exit_long, exit_short, thr):
    """Una posición a la vez. Decisión al cierre de la vela (ts+1h); no se abre hasta que la anterior sale."""
    busy = np.datetime64("1900-01-01")
    one_h = np.timedelta64(1, "h")
    taken = []
    for i in range(len(ts)):
        if ts[i] + one_h < busy:
            continue
        pl, ps = p_long[i], p_short[i]
        if max(pl, ps) < thr:
            continue
        if pl >= ps:
            taken.append(r_long[i])
            busy = exit_long[i]
        else:
            taken.append(r_short[i])
            busy = exit_short[i]
    return np.array(taken)


def summarize_trades(r):
    if len(r) == 0:
        return dict(n=0, media_R=np.nan, suma_R=0.0, win=np.nan, pf=np.nan, max_dd_R=0.0)
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    eq = np.cumsum(r)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0], eq])) - np.concatenate([[0], eq])))
    return dict(n=len(r), media_R=float(r.mean()), suma_R=float(r.sum()), win=float((r > 0).mean()),
                pf=float(wins / losses) if losses > 0 else np.inf, max_dd_R=dd)


def block_bootstrap_ci(r, block, n_boot, rng, q=(5, 95)):
    if len(r) < block * 2:
        return (np.nan, np.nan)
    nb = int(np.ceil(len(r) / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, len(r) - block + 1, nb)
        means[b] = np.concatenate([r[s:s + block] for s in starts])[:len(r)].mean()
    return tuple(np.percentile(means, q))


# ---------- informe ----------
def report(con, cfg, symbol, model=None, source=None, start=None, end=None):
    model = model or cfg["clef"]["model"]
    ev = cfg["eval"]
    rng = np.random.default_rng(0)
    df = load_dataset(con, cfg, symbol, model, source, start, end)
    lab = cfg["labeling"]
    print(f"=== {symbol} | modelo {model} | SL {lab['main_sl_atr']} ATR, TP {lab['main_tp_atr']} ATR, "
          f"{lab['horizon_hours']}h | {df.index.min()} -> {df.index.max()} | {len(df)} velas ===")
    print(f"Coste ida y vuelta supuesto: {cfg['costs']['round_trip_cost_frac']:.4%} del nominal + funding real\n")

    for d in ("long", "short"):
        yp, yo = df[f"y_pess_{d}"].to_numpy(), df[f"y_opt_{d}"].to_numpy()
        p = df[f"p_{d}"].to_numpy()
        n_amb = int((df[f"outcome_{d}"] == "ambiguous").sum())
        print(f"--- {d.upper()} ---  ambiguas sin resolver: {n_amb}")
        base = yp.mean()
        print(f"tasa base (TP antes que SL): {base:.3f}   P media de Clef: {p.mean():.3f}")
        print(f"Brier Clef: {brier(p, yp):.4f}  (tasa base constante: {brier(np.full_like(p, base), yp):.4f})  "
              f"LogLoss: {logloss(p, yp):.4f}  AUC: {auc(p, yp):.4f}")
        if n_amb:
            print(f"  [optimista] Brier {brier(p, yo):.4f}  AUC {auc(p, yo):.4f}")
        print("Calibración (¿cuando dice X%, acierta X%?):")
        print(calibration_table(p, yp).to_string())

        lr = walk_forward_lr(df, d, ev["min_train_years"])
        m = ~np.isnan(lr)
        if m.sum() > 100:
            print(f"Baseline logístico walk-forward (mismas filas, {m.sum()}): AUC {auc(lr[m], yp[m]):.4f}  "
                  f"Brier {brier(lr[m], yp[m]):.4f}   |  Clef en esas filas: AUC {auc(p[m], yp[m]):.4f}  "
                  f"Brier {brier(p[m], yp[m]):.4f}")
        print()

    print("=== SIMULACIÓN (una posición a la vez, costes y funding incluidos) ===")
    ts = df.index.to_numpy()
    args = (ts, df["p_long"].to_numpy(), df["p_short"].to_numpy())
    for variant in ("pess", "opt"):
        if variant == "opt" and not (df["outcome_long"].eq("ambiguous").any() or df["outcome_short"].eq("ambiguous").any()):
            continue
        rl, rs = df[f"r_{variant}_long"].to_numpy(), df[f"r_{variant}_short"].to_numpy()
        el, es = df["exit_long"].to_numpy(), df["exit_short"].to_numpy()
        print(f"[caso {'pesimista' if variant == 'pess' else 'optimista'} para velas ambiguas]")
        rows = []
        for thr in ev["thresholds"]:
            tr = simulate(*args, rl, rs, el, es, thr)
            s = summarize_trades(tr)
            lo, hi = block_bootstrap_ci(tr, ev["bootstrap_block"], ev["n_bootstrap"], rng)
            # permutación: barajar las probabilidades en el tiempo destruye la relación con el resultado
            perm = []
            for _ in range(ev["n_permutations"] if len(tr) else 0):
                idx = rng.permutation(len(ts))
                t2 = simulate(ts, args[1][idx], args[2][idx], rl, rs, el, es, thr)
                perm.append(t2.mean() if len(t2) else np.nan)
            perm = np.array(perm)
            pval = float(np.nanmean(perm >= s["media_R"])) if len(perm) and not np.isnan(s["media_R"]) else np.nan
            rows.append({"umbral": thr, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()},
                         "IC90_mediaR": f"[{lo:.2f}, {hi:.2f}]", "p_perm": round(pval, 3)})
        print(pd.DataFrame(rows).to_string(index=False))
        print()

    # referencias
    rl, rs = df["r_pess_long"].to_numpy(), df["r_pess_short"].to_numpy()
    print("Referencias sin Clef (todas las velas, sin filtro, solapadas; solo para contexto):")
    print(f"  siempre largo: media {np.nanmean(rl):+.3f} R | siempre corto: media {np.nanmean(rs):+.3f} R")
    print("\nNOTA: p_perm = fracción de barajados aleatorios con media >= la real. Si es grande, es compatible con suerte.")
    print("NOTA: con pocas operaciones los IC son anchos; no concluyas con n < ~100 por umbral.")


# ---------- pregunta direccional: ¿el precio a H horas es mayor que la entrada? ----------
def load_dir_dataset(con, cfg, symbol, model, source=None, start=None, end=None):
    H = cfg["labeling"]["horizon_hours"]
    calls = con.execute(
        "SELECT ts, source, probs FROM clef_calls WHERE symbol=? AND model=? AND state_version=? AND qset_version=?",
        [symbol, model, cfg["clef"]["state_version"], clefmod.qset_version(cfg)]).df()
    if calls.empty:
        raise SystemExit("No hay llamadas del conjunto 'dir'. Ejecuta: run --qset dir --sample 2000")
    calls["p_up"] = calls["probs"].map(lambda s: json.loads(s)[clefmod.DIR_KEY])
    calls = calls.drop(columns="probs")
    if source:
        calls = calls[calls["source"] == source]
    if start:
        calls = calls[calls["ts"] >= pd.Timestamp(start)]
    if end:
        calls = calls[calls["ts"] < pd.Timestamp(end)]

    c = data.load_candles(con, symbol).set_index("ts")
    idx = pd.Series(c.index, index=c.index)
    entry, exit_ = c["open"].shift(-1), c["open"].shift(-(H + 1))
    contiguous = (idx.shift(-(H + 1)) - idx) == pd.Timedelta(hours=H + 1)
    f = valid_rows(compute_features(c.reset_index(), data.load_funding(con, symbol), cfg["labeling"]["atr_period"]))
    out = calls.set_index("ts").join(f[FEATURE_COLS + ["atr"]], how="inner")
    out["entry"], out["exit"], out["ok"] = entry, exit_, contiguous
    out = out[out["ok"].fillna(False)].dropna(subset=["entry", "exit"])
    out["ret_atr"] = (out["exit"] - out["entry"]) / out["atr"]
    out["up"] = (out["exit"] > out["entry"]).astype(int)
    out["cost_atr"] = cfg["costs"]["round_trip_cost_frac"] / (out["atr_pct"] / 100)
    return out.sort_index()


def report_dir(con, cfg, symbol, model=None, source=None, start=None, end=None):
    model = model or cfg["clef"]["model"]
    H = cfg["labeling"]["horizon_hours"]
    df = load_dir_dataset(con, cfg, symbol, model, source, start, end)
    p, y = df["p_up"].to_numpy(), df["up"].to_numpy()
    print(f"=== {symbol} | modelo {model} | PREGUNTA DIRECCIONAL: ¿precio a {H}h > entrada? | "
          f"{df.index.min()} -> {df.index.max()} | {len(df)} velas ===\n")
    base = y.mean()
    print(f"tasa base (sube): {base:.3f}   P media de Clef: {p.mean():.3f}")
    print(f"Brier Clef: {brier(p, y):.4f}  (constante: {brier(np.full_like(p, base), y):.4f})  "
          f"LogLoss: {logloss(p, y):.4f}  AUC: {auc(p, y):.4f}")
    rho = df["p_up"].rank().corr(df["ret_atr"].rank())
    print(f"Correlación de rangos P(sube) vs retorno a {H}h (en ATR): {rho:+.4f}  (±{1.96 / np.sqrt(len(df)):.3f} ~95% si fuera azar)")
    print("Calibración (¿cuando dice X%, sube X%?):")
    print(calibration_table(p, y, bins=(0, .3, .4, .45, .5, .55, .6, .7, 1.0001)).to_string())

    lr = walk_forward_lr(df, None, cfg["eval"]["min_train_years"], ycol="up")
    m = ~np.isnan(lr)
    if m.sum() > 100:
        print(f"\nBaseline logístico walk-forward ({m.sum()} filas): AUC {auc(lr[m], y[m]):.4f} "
              f"Brier {brier(lr[m], y[m]):.4f}  |  Clef en esas filas: AUC {auc(p[m], y[m]):.4f} Brier {brier(p[m], y[m]):.4f}")

    print("\n--- Operar el signo (largo si P>0,5, corto si P<0,5), en unidades de ATR ---")
    print(f"Coste ida y vuelta medio ≈ {df['cost_atr'].mean():.3f} ATR (supuesto {cfg['costs']['round_trip_cost_frac']:.2%} del nominal)")
    side = np.sign(p - 0.5)
    gross = side * df["ret_atr"].to_numpy()
    net = gross - df["cost_atr"].to_numpy()
    rows = []
    for lo in (0.0, 0.03, 0.05, 0.10, 0.15, 0.20):
        sel = np.abs(p - 0.5) >= lo
        if sel.sum() < 20:
            continue
        g, nn = gross[sel], net[sel]
        se = nn.std(ddof=1) / np.sqrt(sel.sum())
        rows.append({"|P-0.5|>=": lo, "n": int(sel.sum()), "aciertos": round(float((g > 0).mean()), 3),
                     "media_bruta_ATR": round(float(g.mean()), 3), "media_neta_ATR": round(float(nn.mean()), 3),
                     "IC90_neta": f"[{nn.mean() - 1.645 * se:.3f}, {nn.mean() + 1.645 * se:.3f}]"})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nNOTA: muestra aleatoria y dispersa, así que cada fila es casi independiente; el IC es aproximado.")
    print("NOTA: un edge real debería tener AUC claramente > 0,5, correlación fuera de ±(1,96/√n) y media neta > 0 con IC por encima de 0.")


def report_to_file(con, cfg, symbol, **kw):
    buf = io.StringIO()
    qs = clefmod.question_set(cfg)
    with redirect_stdout(buf):
        (report_dir if qs == "dir" else report)(con, cfg, symbol, **kw)
    txt = buf.getvalue()
    print(txt)
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"eval_{qs}_{symbol}_{datetime.now():%Y%m%d_%H%M%S}.txt"
    path.write_text(txt, encoding="utf-8")
    print(f"Informe guardado en {path}")
