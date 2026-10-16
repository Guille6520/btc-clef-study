"""Etiquetado triple barrera (TP / SL / tiempo) sin sesgo de empate.

Reglas (todas deliberadamente realistas):
  * La decisión se toma al cierre de la vela T; la entrada es la APERTURA de la vela T+1.
  * La vela de entrada ya cuenta para stop y TP.
  * Gap: si la apertura de una vela ya está más allá del stop, el fill es a esa apertura (peor que el stop).
    Si está más allá del TP, el fill es a la apertura (mejor).
  * El TP (orden límite) solo se considera tocado si el precio lo ATRAVIESA (high > TP en largos).
    El stop se considera tocado si el precio lo alcanza (low <= SL en largos).
  * Si TP y SL caen en la misma vela de 1h: se reproduce esa hora con velas de 1m y gana lo que toque primero.
    Si ni el 1m lo decide (o no hay datos), queda 'ambiguous' y se guardan los dos casos (pesimista/optimista).
    Nunca se descarta ni se asigna a ciegas.
  * Si pasan `horizon_hours` sin tocar nada: salida a mercado en la apertura de la vela siguiente.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import data
from .features import compute_features

OUT = np.array(["time", "sl", "tp", "ambiguous"])
T_TIME, T_SL, T_TP, T_AMB = 0, 1, 2, 3


def _combos(cfg):
    lab = cfg["labeling"]
    return [(sl, tp) for sl in lab["grid_sl"] for tp in lab["grid_tp"]]


def _resolve_1m(m, slp, tpp, direction):
    """m: DataFrame 1m de una hora, ordenado. Devuelve ('sl'|'tp'|'ambiguous'|None, fill_px)."""
    for o, h, l in zip(m["open"].to_numpy(), m["high"].to_numpy(), m["low"].to_numpy()):
        if direction == 1:
            sl_gap, tp_gap = o <= slp, o > tpp
            sl_hit, tp_hit = l <= slp, h > tpp
        else:
            sl_gap, tp_gap = o >= slp, o < tpp
            sl_hit, tp_hit = h >= slp, l < tpp
        if sl_gap:
            return "sl", o
        if tp_gap:
            return "tp", o
        if sl_hit and tp_hit:
            return "ambiguous", None
        if sl_hit:
            return "sl", slp
        if tp_hit:
            return "tp", tpp
    return None, None


def run_labeling(con, cfg, symbol):
    lab = cfg["labeling"]
    H = int(lab["horizon_hours"])
    c = data.load_candles(con, symbol)
    if len(c) < H + 600:
        raise SystemExit(f"{symbol}: histórico insuficiente ({len(c)} velas). Ejecuta 'download'.")
    fund = data.load_funding(con, symbol)
    feats = compute_features(c, fund, lab["atr_period"])

    ts = c["ts"].to_numpy()
    o, h, l = c["open"].to_numpy(), c["high"].to_numpy(), c["low"].to_numpy()
    atr_all = feats["atr"].to_numpy()
    n = len(c)
    N = n - H - 1                       # decisiones posibles: i = 0..N-1
    atr = atr_all[:N]
    entry = o[1:N + 1]
    Wo = sliding_window_view(o, H)[1:N + 1]
    Wh = sliding_window_view(h, H)[1:N + 1]
    Wl = sliding_window_view(l, H)[1:N + 1]
    exit_open = o[H + 1:H + 1 + N]

    # funding acumulado por vela (liquidaciones con ts <= ts[j])
    if len(fund):
        fr = fund.drop_duplicates("ts").set_index("ts")["rate"]
        cf = c["ts"].map(fr).fillna(0.0).cumsum().to_numpy()
    else:
        cf = np.zeros(n)

    # solo ventanas contiguas (sin huecos de velas) y con ATR válido
    span = (ts[H + 1:H + 1 + N] - ts[:N]) == np.timedelta64(H + 1, "h")
    valid = span & np.isfinite(atr) & (atr > 0)
    rows = np.arange(N)
    out_frames = []

    resolve_all = lab["resolve_ambiguity"] == "all"
    main = (float(lab["main_sl_atr"]), float(lab["main_tp_atr"]))

    for d, dname in ((1, "long"), (-1, "short")):
        for sl, tp in _combos(cfg):
            risk = sl * atr
            if d == 1:
                slp, tpp = entry - risk, entry + tp * atr
                sl_hit, tp_hit = Wl <= slp[:, None], Wh > tpp[:, None]
                sl_gap, tp_gap = Wo <= slp[:, None], Wo > tpp[:, None]
            else:
                slp, tpp = entry + risk, entry - tp * atr
                sl_hit, tp_hit = Wh >= slp[:, None], Wl < tpp[:, None]
                sl_gap, tp_gap = Wo >= slp[:, None], Wo < tpp[:, None]

            ks = np.where(sl_hit.any(1), sl_hit.argmax(1), H)
            kt = np.where(tp_hit.any(1), tp_hit.argmax(1), H)
            k = np.minimum(ks, kt)
            kc = np.minimum(k, H - 1)
            none = k == H
            same = (ks == kt) & ~none
            sg, tg = sl_gap[rows, kc], tp_gap[rows, kc]

            sl_first = (ks < kt) | (same & sg)
            tp_first = (kt < ks) | (same & ~sg & tg)
            amb = same & ~sg & ~tg
            code = np.full(N, T_TIME)
            code[sl_first], code[tp_first], code[amb] = T_SL, T_TP, T_AMB

            fill_sl = np.where(sg, Wo[rows, kc], slp)
            fill_tp = np.where(tg, Wo[rows, kc], tpp)
            exit_px = np.where(code == T_TIME, exit_open,
                               np.where(code == T_SL, fill_sl,
                                        np.where(code == T_TP, fill_tp, slp)))
            r_gross = d * (exit_px - entry) / risk
            r_opt = r_gross.copy()
            r_gross = np.where(code == T_AMB, -1.0, r_gross)       # pesimista
            r_opt = np.where(code == T_AMB, tp / sl, r_opt)         # optimista
            exit_idx = np.where(code == T_TIME, rows + H + 1, rows + k + 2)
            exit_idx = np.minimum(exit_idx, n - 1)
            fund_frac = d * (cf[exit_idx] - cf[rows + 1])
            resolved_by = np.where(code == T_AMB, None, "1h").astype(object)

            # --- resolver ambiguos con 1m (solo configuración principal salvo resolve_ambiguity='all')
            if (sl, tp) == main or resolve_all:
                ai = np.where(valid & (code == T_AMB))[0]
                if len(ai):
                    hours = [pd.Timestamp(ts[i + 1 + k[i]]) for i in ai]
                    data.ensure_1m(con, cfg, symbol, hours)
                    for i, hr in zip(ai, hours):
                        m = con.execute(
                            "SELECT open, high, low FROM candles_1m WHERE symbol=? AND ts >= ? AND ts < ? ORDER BY ts",
                            [symbol, hr, hr + pd.Timedelta(hours=1)]).df()
                        if len(m) < 55:
                            continue
                        res, px = _resolve_1m(m, slp[i], tpp[i], d)
                        if res == "sl":
                            code[i] = T_SL
                            r_gross[i] = r_opt[i] = d * (px - entry[i]) / risk[i]
                        elif res == "tp":
                            code[i] = T_TP
                            r_gross[i] = r_opt[i] = d * (px - entry[i]) / risk[i]
                        if res in ("sl", "tp"):
                            resolved_by[i] = "1m"

            ok = valid
            df = pd.DataFrame({
                "symbol": symbol,
                "ts": ts[:N][ok],
                "direction": dname,
                "sl_atr": float(sl),
                "tp_atr": float(tp),
                "horizon_h": H,
                "outcome": OUT[code[ok]],
                "resolved_by": resolved_by[ok],
                "entry_px": entry[ok],
                "risk_frac": (risk / entry)[ok],
                "r_gross": r_gross[ok],
                "r_gross_opt": r_opt[ok],
                "funding_frac": fund_frac[ok],
                "exit_ts": ts[exit_idx][ok],
                "y_pess": (code[ok] == T_TP).astype(int),
                "y_opt": np.isin(code[ok], (T_TP, T_AMB)).astype(int),
            })
            out_frames.append(df)
            n_amb = int((df["outcome"] == "ambiguous").sum())
            print(f"  {symbol} {dname:5s} SL {sl} TP {tp}: {len(df)} filas, "
                  f"TP {int((df['outcome'] == 'tp').sum())}, SL {int((df['outcome'] == 'sl').sum())}, "
                  f"tiempo {int((df['outcome'] == 'time').sum())}, ambiguas {n_amb}")

    allf = pd.concat(out_frames, ignore_index=True)
    con.execute("DELETE FROM labels WHERE symbol=? AND horizon_h=?", [symbol, H])
    con.register("_lab", allf)
    con.execute("INSERT INTO labels SELECT symbol, ts, direction, sl_atr, tp_atr, horizon_h, outcome, resolved_by, "
                "entry_px, risk_frac, r_gross, r_gross_opt, funding_frac, exit_ts, y_pess, y_opt FROM _lab")
    con.unregister("_lab")
    print(f"{symbol}: {len(allf)} etiquetas guardadas")
