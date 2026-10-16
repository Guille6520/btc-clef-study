"""Cliente de Clef (Cloudflare Workers AI) con caché en DuckDB.

El formato REST exacto no está documentado en el changelog, así que este módulo asume el estándar de
Workers AI (POST /accounts/{id}/ai/run/@cf/cloudflare/<modelo>, cuerpo = input del modelo) y el comando
`probe` imprime la respuesta cruda para verificarlo ANTES de lanzar una pasada grande.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from . import data
from .features import build_state, compute_features, valid_rows

MODEL_IDS = {"clef": "@cf/cloudflare/clef", "clef-flash": "@cf/cloudflare/clef-flash"}


def qkey(direction, sl, tp):
    return f"{direction}_sl{str(sl).replace('.', 'p')}_tp{str(tp).replace('.', 'p')}"


def main_keys(cfg):
    lab = cfg["labeling"]
    return {d: qkey(d, lab["main_sl_atr"], lab["main_tp_atr"]) for d in ("long", "short")}


DIR_KEY = "up"


def question_set(cfg):
    """'tpsl' = TP antes que SL (por defecto) | 'dir' = ¿el precio a H horas es mayor que la entrada?"""
    return cfg["clef"].get("question_set", "tpsl")


def qset_version(cfg):
    lab = cfg["labeling"]
    if question_set(cfg) == "dir":
        return f"dir1-h{lab['horizon_hours']}"
    return (f"q1-{cfg['clef']['question_grid']}-h{lab['horizon_hours']}"
            f"-sl{lab['main_sl_atr']}-tp{lab['main_tp_atr']}")


def build_questions(cfg):
    lab = cfg["labeling"]
    H = lab["horizon_hours"]
    if question_set(cfg) == "dir":
        return {DIR_KEY: {
            "type": "noul",
            "instructions": (
                f"A position is opened at the open of the next hourly candle. Will the price at the open of the "
                f"candle {H} hours later be HIGHER than the entry price? Answer the probability that it is higher, "
                f"regardless of by how much."),
        }}
    if cfg["clef"]["question_grid"] == "full":
        combos = [(sl, tp) for sl in lab["grid_sl"] for tp in lab["grid_tp"]]
    else:
        combos = [(lab["main_sl_atr"], lab["main_tp_atr"])]
    qs = {}
    for d in ("long", "short"):
        for sl, tp in combos:
            side_word, sl_dir, tp_dir = ("LONG", "below", "above") if d == "long" else ("SHORT", "above", "below")
            qs[qkey(d, sl, tp)] = {
                "type": "noul",
                "instructions": (
                    f"A {side_word} position is opened at the open of the next hourly candle. Its stop-loss is "
                    f"{sl} ATR {sl_dir} the entry and its take-profit is {tp} ATR {tp_dir} the entry "
                    f"(ATR = 14-period hourly ATR, given as a percent of price in the state). "
                    f"Will price reach the take-profit BEFORE the stop-loss within the next {H} hours?"),
            }
    return qs


def _endpoint(model):
    acct = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not acct or not os.environ.get("CLOUDFLARE_API_TOKEN"):
        raise SystemExit("Faltan CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN (ver .env.example).")
    return f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/{MODEL_IDS[model]}"


def call_clef(cfg, state, questions, model=None):
    model = model or cfg["clef"]["model"]
    url = _endpoint(model)
    headers = {"Authorization": f"Bearer {os.environ['CLOUDFLARE_API_TOKEN']}"}
    body = {"model": model, "state": state, "questions": questions}
    last_err = None
    for attempt in range(5):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=cfg["clef"]["timeout_s"])
        except requests.RequestException as e:      # red caída o timeout: transitorio
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(2 ** attempt)
            continue
        if r.ok:
            return r.json()
        detail = _error_detail(r)
        if "daily free allocation" in detail or '"code":4006' in r.text.replace(" ", ""):
            raise QuotaExceeded(detail)             # no se arregla reintentando
        if r.status_code in (429, 500, 502, 503, 504):
            last_err = f"HTTP {r.status_code}: {detail}"
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"HTTP {r.status_code} (no reintentable): {detail}")
    raise RuntimeError(f"Clef falló tras reintentos: {last_err}")


class QuotaExceeded(RuntimeError):
    """Cuota diaria de neuronas agotada (código 4006). La pasada se detiene y es reanudable."""


def _error_detail(r):
    """Mensaje de error de Cloudflare (sin cabeceras ni credenciales)."""
    try:
        errs = r.json().get("errors") or []
        return "; ".join(f"[{e.get('code')}] {e.get('message')}" for e in errs) or r.text[:300]
    except ValueError:
        return r.text[:300]


def parse_probs(resp, questions):
    """Extrae {pregunta: P(yes)} de la respuesta. Tolera 'result' envolvente."""
    result = resp.get("result", resp)
    answers = result.get("answers", result)
    probs = {}
    for k in questions:
        a = answers.get(k)
        if a is None:
            raise ValueError(f"Respuesta sin la pregunta '{k}'. Claves recibidas: {list(answers)[:5]}")
        probs[k] = float(a["noul"])
    return probs


def probe(cfg, symbol, con):
    """Una llamada real sobre la última vela disponible; imprime la respuesta cruda."""
    c = data.load_candles(con, symbol)
    f = valid_rows(compute_features(c, data.load_funding(con, symbol), cfg["labeling"]["atr_period"]))
    row = f.iloc[-1]
    state, qs = build_state(row), build_questions(cfg)
    print("ESTADO:\n", state, "\n")
    print("PREGUNTAS:\n", json.dumps(qs, indent=1)[:1500], "\n")
    resp = call_clef(cfg, state, qs)
    print("RESPUESTA CRUDA:\n", json.dumps(resp, indent=1)[:3000], "\n")
    print("PROBABILIDADES PARSEADAS:", parse_probs(resp, qs))
    print(f"~tokens de entrada estimados: {(len(state) + len(json.dumps(qs))) // 4}")


def run(cfg, con, symbol, start=None, end=None, sample=None, dry=False, source="backtest", model=None):
    model = model or cfg["clef"]["model"]
    sv, qv = cfg["clef"]["state_version"], qset_version(cfg)
    c = data.load_candles(con, symbol)
    f = valid_rows(compute_features(c, data.load_funding(con, symbol), cfg["labeling"]["atr_period"]))
    if start:
        f = f[f.index >= pd.Timestamp(start)]
    if end:
        f = f[f.index < pd.Timestamp(end)]
    done = {pd.Timestamp(t) for (t,) in con.execute(
        "SELECT ts FROM clef_calls WHERE symbol=? AND model=? AND state_version=? AND qset_version=?",
        [symbol, model, sv, qv]).fetchall()}
    todo = f[~f.index.isin(done)]
    if sample and sample < len(todo):
        todo = todo.sample(sample, random_state=0).sort_index()
    qs = build_questions(cfg)
    est_tok = (len(build_state(todo.iloc[0])) + len(json.dumps(qs))) // 4 if len(todo) else 0
    print(f"{symbol} {model}: {len(todo)} llamadas pendientes ({len(done)} ya en caché), "
          f"~{est_tok} tokens de entrada cada una, ~{est_tok * len(todo) / 1e6:.1f}M tokens en total")
    if dry or not len(todo):
        return

    def work(ts_, row):
        state = build_state(row)
        resp = call_clef(cfg, state, qs, model)
        return ts_, state, parse_probs(resp, qs), resp

    buf, errors, quota_hit = [], 0, None
    with ThreadPoolExecutor(max_workers=cfg["clef"]["concurrency"]) as ex:
        futs = [ex.submit(work, ts_, row) for ts_, row in todo.iterrows()]
        for n, fu in enumerate(as_completed(futs), 1):
            try:
                ts_, state, probs, resp = fu.result()
                buf.append((symbol, ts_, model, sv, qv, source, state, json.dumps(probs), json.dumps(resp)))
            except QuotaExceeded as e:
                quota_hit = e
                for other in futs:                      # cancela lo que no ha empezado
                    other.cancel()
                break
            except Exception as e:                      # una vela fallida no tumba la pasada
                errors += 1
                if errors <= 5:
                    print(f"\n  error: {e}")
            if len(buf) >= 200 or n == len(futs):
                _flush(con, buf)
                buf = []
                print(f"  {n}/{len(futs)} (errores: {errors})", end="\r")
    _flush(con, buf)                                    # guarda lo ya conseguido antes de parar
    if quota_hit:
        print(f"\nCUOTA AGOTADA: {quota_hit}\nLo ya guardado se conserva; repite el comando cuando se libere la cuota.")
        return
    print(f"\nHecho. Errores: {errors}. Repite el comando para reintentar las que faltan.")


def _flush(con, buf):
    if not buf:
        return
    df = pd.DataFrame(buf, columns=["symbol", "ts", "model", "state_version", "qset_version",
                                    "source", "state", "probs", "raw"])
    con.register("_calls", df)
    con.execute("INSERT OR REPLACE INTO clef_calls (symbol, ts, model, state_version, qset_version, source, "
                "state, probs, raw) SELECT symbol, ts, model, state_version, qset_version, source, state, probs, "
                "raw FROM _calls")
    con.unregister("_calls")


def live(cfg, con, symbol):
    """Descarga lo último, y pregunta a Clef por la última vela cerrada (registro en vivo, sin operar)."""
    data.download_candles_1h(con, cfg, symbol)
    data.download_funding(con, cfg, symbol)
    last = con.execute("SELECT max(ts) FROM candles_1h WHERE symbol=?", [symbol]).fetchone()[0]
    run(cfg, con, symbol, start=last, source="live")
