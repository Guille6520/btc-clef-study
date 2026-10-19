"""Datos de derivados y contexto de mercado, todo de fuentes públicas gratuitas.

  * Binance (data.binance.vision): velas 1h del perpetuo con volumen de compradores agresores (desde 2020),
    índice de prima (base) y métricas de 5 min (open interest, ratios largo/corto, volumen agresor).
  * Deribit: DVOL (volatilidad implícita de BTC) desde 2021.
  * alternative.me: índice de miedo y avaricia.
  * FRED: VIX, Nasdaq, dólar, bono a 10 años y S&P 500 (diarios).

Regla anti-fuga: cada valor horario es el último conocido antes del CIERRE de la vela; los datos diarios se
desplazan un día completo.
"""
import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

BASE = "https://data.binance.vision/data/futures/um"
FRED = ["VIXCLS", "NASDAQCOM", "DTWEXBGS", "DGS10", "SP500"]
SYMBOL_BINANCE = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}


def _zip_csv(url, header=None, retries=3):
    for k in range(retries):
        try:
            r = requests.get(url, timeout=60)
        except requests.RequestException:
            time.sleep(1 + k)
            continue
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            time.sleep(1 + k)
            continue
        z = zipfile.ZipFile(io.BytesIO(r.content))
        return pd.read_csv(z.open(z.namelist()[0]), header=header)
    return None


def _klines(df):
    df = df[pd.to_numeric(df[0], errors="coerce").notna()].apply(pd.to_numeric)
    ms = df[0].where(df[0] < 1e14, df[0] / 1000)
    out = pd.DataFrame({"ts": pd.to_datetime(ms, unit="ms"), "open": df[1], "high": df[2], "low": df[3],
                        "close": df[4], "volume": df[5], "trades": df[8], "taker_buy_volume": df[9]})
    return out


def snapshot_candle_ts(create_time):
    """Apertura de la vela de 1 h a la que pertenece una instantánea: la de las 12:05 y la de las 13:00 exactas
    pertenecen a la vela de las 12:00 (que cierra a las 13:00). Nunca se usa una instantánea posterior al cierre."""
    return pd.to_datetime(create_time).dt.ceil("h") - pd.Timedelta(hours=1)


def _months(start, end):
    return pd.period_range(start, end, freq="M")


def _fetch_kline_kind(kind, sym, interval, start="2020-01"):
    """Descarga todos los meses (mensual si existe; si no, diario) de un tipo de velas del perpetuo."""
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    jobs = []
    for p in _months(start, today.to_period("M")):
        jobs.append(p)

    def month(p):
        m = _zip_csv(f"{BASE}/monthly/{kind}/{sym}/{interval}/{sym}-{interval}-{p}.zip")
        if m is not None:
            return _klines(m)
        frames = []
        for d in pd.date_range(p.start_time, min(p.end_time.normalize(), today - pd.Timedelta(days=1))):
            x = _zip_csv(f"{BASE}/daily/{kind}/{sym}/{interval}/{sym}-{interval}-{d:%Y-%m-%d}.zip")
            if x is not None:
                frames.append(_klines(x))
        return pd.concat(frames) if frames else None

    with ThreadPoolExecutor(max_workers=6) as ex:
        parts = [x for x in ex.map(month, jobs) if x is not None]
    return pd.concat(parts).drop_duplicates("ts").sort_values("ts") if parts else pd.DataFrame()


def download_perp(con, symbol):
    sym = SYMBOL_BINANCE[symbol]
    k = _fetch_kline_kind("klines", sym, "1h")
    p = _fetch_kline_kind("premiumIndexKlines", sym, "1h")
    if k.empty:
        print(f"{symbol}: sin velas de perpetuo")
        return
    out = pd.DataFrame({"symbol": symbol, "ts": k["ts"], "taker_buy_ratio": (k["taker_buy_volume"] / k["volume"]).values,
                        "trades": k["trades"].values})
    prem = p.set_index("ts")["close"] if not p.empty else pd.Series(dtype=float)
    out["premium"] = out["ts"].map(prem)
    con.register("_p", out)
    con.execute("INSERT OR REPLACE INTO perp_1h SELECT symbol, ts, taker_buy_ratio, trades, premium FROM _p")
    con.unregister("_p")
    print(f"{symbol}: {len(out)} velas del perpetuo (volumen agresor + prima) desde {out['ts'].min()}")


def download_oi(con, symbol, start="2020-01-01"):
    sym = SYMBOL_BINANCE[symbol]
    last = con.execute("SELECT max(ts) FROM oi_1h WHERE symbol=?", [symbol]).fetchone()[0]
    first = pd.Timestamp(last).normalize() - pd.Timedelta(days=1) if last is not None else pd.Timestamp(start)
    days = pd.date_range(first, pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=1))

    def day(d):
        return _zip_csv(f"{BASE}/daily/metrics/{sym}/{sym}-metrics-{d:%Y-%m-%d}.zip", header=0)

    with ThreadPoolExecutor(max_workers=8) as ex:
        frames = [x for x in ex.map(day, days) if x is not None]
    if not frames:
        print(f"{symbol}: sin métricas nuevas")
        return
    m = pd.concat(frames)
    m["create_time"] = pd.to_datetime(m["create_time"])
    # la instantánea de las 12:05 pertenece a la vela 12:00 (cierra a las 13:00); la de las 13:00 exactas, a la 12:00
    m["ts"] = snapshot_candle_ts(m["create_time"])
    h = m.sort_values("create_time").groupby("ts").last().reset_index()
    out = pd.DataFrame({"symbol": symbol, "ts": h["ts"], "oi": h["sum_open_interest"],
                        "oi_value": h["sum_open_interest_value"], "ls_top_acc": h["count_toptrader_long_short_ratio"],
                        "ls_top_pos": h["sum_toptrader_long_short_ratio"], "ls_all": h["count_long_short_ratio"],
                        "taker_ls_vol": h["sum_taker_long_short_vol_ratio"]})
    con.register("_o", out)
    con.execute("INSERT OR REPLACE INTO oi_1h SELECT symbol, ts, oi, oi_value, ls_top_acc, ls_top_pos, ls_all, "
                "taker_ls_vol FROM _o")
    con.unregister("_o")
    print(f"{symbol}: {len(out)} horas de open interest / ratios, de {out['ts'].min()} a {out['ts'].max()}")


def download_dvol(con):
    last = con.execute("SELECT max(ts) FROM dvol_1h").fetchone()[0]
    start = pd.Timestamp(last) if last is not None else pd.Timestamp("2021-03-01")
    end = pd.Timestamp.utcnow().tz_localize(None)
    rows, t = [], start
    while t < end:
        t2 = min(t + pd.Timedelta(days=30), end)
        r = requests.get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                         params={"currency": "BTC", "start_timestamp": int(t.timestamp() * 1000),
                                 "end_timestamp": int(t2.timestamp() * 1000), "resolution": 3600}, timeout=30)
        rows += r.json().get("result", {}).get("data", [])
        t = t2
        time.sleep(0.2)
    if not rows:
        print("DVOL: sin datos nuevos")
        return
    df = pd.DataFrame(rows, columns=["ms", "open", "high", "low", "close"])
    out = pd.DataFrame({"ts": pd.to_datetime(df["ms"], unit="ms"), "dvol": df["close"]}).drop_duplicates("ts")
    con.register("_d", out)
    con.execute("INSERT OR REPLACE INTO dvol_1h SELECT ts, dvol FROM _d")
    con.unregister("_d")
    print(f"DVOL: {len(out)} horas, de {out['ts'].min()} a {out['ts'].max()}")


def download_daily_ext(con):
    rows = []
    r = requests.get("https://api.alternative.me/fng/", params={"limit": 0, "format": "json"}, timeout=30)
    for x in r.json()["data"]:
        rows.append(("fng", pd.to_datetime(int(x["timestamp"]), unit="s").date(), float(x["value"])))
    for sid in FRED:
        csv = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": sid, "cosd": "2017-01-01"},
                           timeout=30).text
        df = pd.read_csv(io.StringIO(csv))
        df.columns = ["date", "value"]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        rows += [(sid, pd.Timestamp(d).date(), v) for d, v in zip(df["date"], df["value"]) if pd.notna(v)]
    out = pd.DataFrame(rows, columns=["name", "date", "value"]).drop_duplicates(["name", "date"])
    con.register("_x", out)
    con.execute("INSERT OR REPLACE INTO daily_ext SELECT name, date, value FROM _x")
    con.unregister("_x")
    print(f"diarios: {len(out)} registros ({', '.join(sorted(out['name'].unique()))})")


def download_all(con, cfg, symbol):
    download_perp(con, symbol)
    download_oi(con, symbol)
    if symbol == "BTC":
        download_dvol(con)
    download_daily_ext(con)
