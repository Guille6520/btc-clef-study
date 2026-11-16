"""Clef frente a ML clásico sobre EXACTAMENTE las mismas velas.

El informe de `evaluate` y el de `bench` usan muestras distintas, así que sus cifras no son comparables entre sí.
Aquí se toman solo las velas en las que Clef tiene respuesta y el ML tiene predicción fuera de muestra
(walk-forward por años con embargo, igual que `bench`), y se mide todo sobre ese conjunto común:
  * AUC de cada modelo con IC por bootstrap de semanas, y la diferencia pareada AUC(ML) - AUC(Clef);
  * simulación TP/SL (una posición a la vez, costes y funding) y operar el signo de la pregunta direccional.
La reserva ciega (>= holdout_start) no se toca.
"""
import io
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np
import pandas as pd

from . import classic, evaluate
from . import clef as clefmod
from .config import REPORTS_DIR
from .evaluate import auc, simulate, summarize_trades

ML_MODELS = {"logit": classic.make_logit, "hgb": classic.make_hgb}
WEEK = pd.Timedelta(hours=168)


# ---------- estadística ----------
def week_ids(index):
    """Entero por semana (las velas de una misma semana comparten resultado de mercado: se remuestrean juntas)."""
    return np.asarray((pd.DatetimeIndex(index) - pd.Timestamp("2017-01-01")) // WEEK)


def cluster_ci(stat, groups, n, rng, q=(5, 95)):
    """IC por bootstrap de grupos: remuestrea semanas enteras con reemplazo y recalcula `stat(índices de fila)`."""
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    out = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        out[b] = stat(np.concatenate([members[k] for k in pick]))
    return tuple(np.percentile(out, q))


def paired_auc(y, p_ref, p_other, groups, n, rng):
    """AUC de ambos y su diferencia (other - ref), con IC90 de la diferencia sobre las mismas semanas remuestreadas."""
    a_ref, a_oth = auc(p_ref, y), auc(p_other, y)
    lo, hi = cluster_ci(lambda i: auc(p_other[i], y[i]) - auc(p_ref[i], y[i]), groups, n, rng)
    return a_ref, a_oth, a_oth - a_ref, lo, hi


# ---------- piezas ----------
def _with_qset(cfg, qset):
    return {**cfg, "clef": {**cfg["clef"], "question_set": qset}}


def _load_or_none(loader, *a, **kw):
    try:
        return loader(*a, **kw)
    except SystemExit as e:          # no hay llamadas de ese conjunto: se omite la sección, no se aborta
        print(f"  [omitido] {e}")
        return None


def _ml_predictions(ds, targets, bench):
    return {(t, m): classic.walk_forward(ds, classic.ALL, t, fac, bench)[0]
            for t in targets for m, fac in ML_MODELS.items()}


def _auc_section(title, y, scores, groups, years, n_boot, rng):
    """`scores`: {nombre: p}; el primero es Clef y sirve de referencia para la diferencia pareada."""
    names = list(scores)
    ref = names[0]
    rows = []
    for nm in names:
        p = scores[nm]
        lo, hi = cluster_ci(lambda i, p=p: auc(p[i], y[i]), groups, n_boot, rng)
        row = {"modelo": nm, "n": len(y), "AUC": round(auc(p, y), 4), "IC90": f"[{lo:.3f}, {hi:.3f}]"}
        if nm != ref:
            _, _, d, dlo, dhi = paired_auc(y, scores[ref], p, groups, n_boot, rng)
            row[f"AUC - {ref}"] = f"{d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
        for yr in sorted(set(years)):
            m = years == yr
            row[str(yr)] = round(auc(p[m], y[m]), 3) if m.sum() >= 100 and 0 < y[m].sum() < m.sum() else ""
        rows.append(row)
    print(f"\n--- {title} ---")
    print(pd.DataFrame(rows).fillna("").to_string(index=False))


def _sim_section(d, probs, bench, rng):
    """Simulación TP/SL sobre las filas comunes. `probs`: {nombre: (p_long, p_short)}."""
    ts = d.index.to_numpy()
    rl, rs = d["r_pess_long"].to_numpy(), d["r_pess_short"].to_numpy()
    el, es = d["exit_long"].to_numpy(), d["exit_short"].to_numpy()
    print(f"referencias (todas las velas comunes, solapadas): siempre largo {np.nanmean(rl):+.3f} R | "
          f"siempre corto {np.nanmean(rs):+.3f} R")
    for name, (pl, ps) in probs.items():
        rows = []
        for thr in bench["thresholds_tp"]:
            tr = simulate(ts, pl, ps, rl, rs, el, es, thr)
            s = summarize_trades(tr)
            lo, hi = evaluate.block_bootstrap_ci(tr, 5, 1000, rng)
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


def _dir_sign_section(d, probs, margins, groups, n_boot, rng):
    """Operar el signo de P(sube) - 0,5, neto de costes, en ATR. `probs`: {nombre: p}."""
    ret, cost = d["ret_atr"].to_numpy(), d["cost_atr"].to_numpy()
    rows = []
    for name, p in probs.items():
        for m in margins:
            sel = np.flatnonzero(np.abs(p - 0.5) >= max(m, 1e-9)) if m > 0 else np.arange(len(p))
            if len(sel) < 50:
                continue
            gross = np.sign(p[sel] - 0.5) * ret[sel]
            net = gross - cost[sel]
            g = groups[sel]
            lo, hi = cluster_ci(lambda i: net[i].mean(), g, n_boot, rng)
            rows.append({"modelo": name, "margen": m, "n": len(sel), "bruta_ATR": round(gross.mean(), 3),
                         "neta_ATR": round(net.mean(), 3), "IC90": f"[{lo:.3f}, {hi:.3f}]"})
    print(pd.DataFrame(rows).to_string(index=False))


# ---------- informe ----------
def run_compare(con, cfg, symbol, model=None):
    model = model or cfg["clef"]["model"]
    bench = dict(cfg["bench"])
    rng = np.random.default_rng(0)
    holdout = pd.Timestamp(bench["holdout_start"])
    t0 = pd.Timestamp(f"{bench['first_test_year']}-01-01")
    n_boot = bench["n_bootstrap"]

    ds = classic.build_dataset(con, cfg, symbol)
    ds = ds[ds.index < holdout]                          # la reserva ciega ni se carga
    candidates = int(((ds.index >= t0)).sum())
    print(f"=== CLEF ({model}) vs ML CLÁSICO, MISMAS VELAS | {symbol} | {t0.date()} -> {holdout.date()} (sin reserva ciega) ===")
    print(f"ML: walk-forward por años desde {bench['first_test_year']}, embargo {bench['embargo_hours']} h, "
          f"hiperparámetros fijos, variables 'todas'. Coste {cfg['costs']['round_trip_cost_frac']:.2%} + funding real.")
    print(f"Velas posibles en el rango: {candidates}. Las secciones usan solo las velas con respuesta de Clef.\n")

    # ---- TP antes que SL ----
    print("## TP antes que SL (SL 1,5 ATR, TP 3 ATR, 24 h)")
    tp = _load_or_none(evaluate.load_dataset, con, _with_qset(cfg, "tpsl"), symbol, model, end=holdout)
    if tp is not None:
        tp = tp[tp.index >= t0]
        ml = _ml_predictions(ds, ["y_tp_long", "y_tp_short"], bench)
        ok = pd.Series(True, index=tp.index)
        for v in ml.values():
            ok &= v.reindex(tp.index).notna()
        ok &= tp[["r_pess_long", "r_pess_short"]].notna().all(axis=1) & tp[["exit_long", "exit_short"]].notna().all(axis=1)
        d = tp[ok]
        print(f"Velas comunes: {len(d)} ({len(d) / max(candidates, 1):.1%} del rango), {d.index.min()} -> {d.index.max()}")
        if len(d) >= 200:
            groups, years = week_ids(d.index), d.index.year.to_numpy()
            probs = {f"clef {model}": (d["p_long"].to_numpy(), d["p_short"].to_numpy())}
            for side in ("long", "short"):
                y = d[f"y_pess_{side}"].to_numpy().astype(int)
                scores = {f"clef {model}": d[f"p_{side}"].to_numpy()}
                for m in ML_MODELS:
                    scores[f"{m} [todas]"] = ml[(f"y_tp_{side}", m)].reindex(d.index).to_numpy()
                print(f"tasa base ({side}): {y.mean():.3f}")
                _auc_section(f"AUC, TP antes que SL ({side.upper()})", y, scores, groups, years, n_boot, rng)
            for m in ML_MODELS:
                probs[f"{m} [todas]"] = (ml[("y_tp_long", m)].reindex(d.index).to_numpy(),
                                         ml[("y_tp_short", m)].reindex(d.index).to_numpy())
            print("\nSimulación (una posición a la vez; elige el lado de mayor P; mismo umbral para todos):")
            _sim_section(d, probs, bench, rng)
        else:
            print("Muy pocas velas comunes para comparar (< 200).")

    # ---- dirección a 24 h ----
    print("\n## Dirección a 24 h (¿el precio de salida es mayor que el de entrada?)")
    dr = _load_or_none(evaluate.load_dir_dataset, con, _with_qset(cfg, "dir"), symbol, model, end=holdout)
    if dr is not None:
        dr = dr[dr.index >= t0]
        ml = _ml_predictions(ds, ["up"], bench)
        ok = pd.Series(True, index=dr.index)
        for v in ml.values():
            ok &= v.reindex(dr.index).notna()
        d = dr[ok]
        print(f"Velas comunes: {len(d)} ({len(d) / max(candidates, 1):.1%} del rango), {d.index.min()} -> {d.index.max()}")
        if len(d) >= 200:
            groups, years = week_ids(d.index), d.index.year.to_numpy()
            y = d["up"].to_numpy().astype(int)
            scores = {f"clef {model}": d["p_up"].to_numpy()}
            for m in ML_MODELS:
                scores[f"{m} [todas]"] = ml[("up", m)].reindex(d.index).to_numpy()
            print(f"tasa base (sube): {y.mean():.3f}")
            _auc_section("AUC, dirección a 24 h", y, scores, groups, years, n_boot, rng)
            print("\nOperar el signo (largo si P > 0,5 + m, corto si P < 0,5 - m), neto de costes, en ATR:")
            _dir_sign_section(d, scores, bench["margins_dir"], groups, n_boot, rng)
        else:
            print("Muy pocas velas comunes para comparar (< 200).")

    print("\nNOTAS: IC90 por bootstrap de semanas completas. La diferencia 'AUC - clef' es pareada (mismas semanas remuestreadas):")
    print("NOTAS: si su IC incluye 0, no hay diferencia demostrable. Hiperparámetros del ML fijos, no ajustados con el test.")


def run_compare_to_file(con, cfg, symbol, model=None):
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_compare(con, cfg, symbol, model)
    txt = buf.getvalue()
    print(txt)
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"compare_{symbol}_{datetime.now():%Y%m%d_%H%M%S}.txt"
    path.write_text(txt, encoding="utf-8")
    print(f"Informe guardado en {path}")
