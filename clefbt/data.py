"""Descarga de velas (Binance vía ccxt, público) y funding. Todo en UTC, timestamps ingenuos."""
import time

import ccxt
import pandas as pd

HOUR_MS = 3_600_000


def _ms(ts):
    return int(pd.Timestamp(ts).timestamp() * 1000)


def _to_df(rows, symbol):
    df = pd.DataFrame(rows, columns=["ms", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ms"], unit="ms")
    df["symbol"] = symbol
    return df[["symbol", "ts", "open", "high", "low", "close", "volume"]]


def _insert(con, table, df, cols):
    if df.empty:
        return 0
    con.register("_df", df)
    con.execute(f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _df")
    con.unregister("_df")
    return len(df)


def download_candles_1h(con, cfg, symbol):
    ex = ccxt.binance({"enableRateLimit": True})
    pair = cfg["data"]["ccxt"][symbol]["spot"]
    last = con.execute("SELECT max(ts) FROM candles_1h WHERE symbol=?", [symbol]).fetchone()[0]
    since = _ms(last) + HOUR_MS if last is not None else _ms(cfg["data"]["start"])
    now_ms = int(time.time() * 1000)
    total = 0
    while since < now_ms:
        rows = ex.fetch_ohlcv(pair, "1h", since=since, limit=1000)
        if not rows:
            break
        # descarta la vela aún abierta
        rows = [r for r in rows if r[0] + HOUR_MS <= now_ms]
        if not rows:
            break
        total += _insert(con, "candles_1h", _to_df(rows, symbol),
                         ["symbol", "ts", "open", "high", "low", "close", "volume"])
        since = rows[-1][0] + HOUR_MS
        print(f"  {symbol} 1h hasta {pd.to_datetime(rows[-1][0], unit='ms')}  (+{total})", end="\r")
    print(f"\n{symbol}: {total} velas 1h nuevas")


def download_funding(con, cfg, symbol):
    ex = ccxt.binanceusdm({"enableRateLimit": True})
    pair = cfg["data"]["ccxt"][symbol]["perp"]
    last = con.execute("SELECT max(ts) FROM funding WHERE symbol=?", [symbol]).fetchone()[0]
    since = _ms(last) + 1 if last is not None else _ms("2019-09-01")
    now_ms = int(time.time() * 1000)
    total = 0
    while since < now_ms:
        rows = ex.fetch_funding_rate_history(pair, since=since, limit=1000)
        if not rows:
            break
        df = pd.DataFrame({
            "ts": pd.to_datetime([r["timestamp"] for r in rows], unit="ms").floor("h"),
            "rate": [float(r["fundingRate"]) for r in rows],
        })
        df["symbol"] = symbol
        df = df.drop_duplicates(["symbol", "ts"], keep="last")
        total += _insert(con, "funding", df[["symbol", "ts", "rate"]], ["symbol", "ts", "rate"])
        new_since = rows[-1]["timestamp"] + 1
        if new_since <= since:
            break
        since = new_since
        print(f"  {symbol} funding hasta {df['ts'].iloc[-1]}  (+{total})", end="\r")
    print(f"\n{symbol}: {total} registros de funding nuevos")


def ensure_1m(con, cfg, symbol, hours):
    """Garantiza velas de 1m para las horas pedidas (solo las ambiguas). Devuelve nº de peticiones."""
    if not len(hours):
        return 0
    have = {
        pd.Timestamp(h) for (h,) in con.execute(
            "SELECT date_trunc('hour', ts) h FROM candles_1m WHERE symbol=? GROUP BY 1 HAVING count(*) >= 55",
            [symbol]).fetchall()
    }
    need = sorted({pd.Timestamp(h) for h in hours} - have)
    if not need:
        return 0
    ex = ccxt.binance({"enableRateLimit": True})
    pair = cfg["data"]["ccxt"][symbol]["spot"]
    reqs, i = 0, 0
    while i < len(need):
        start = need[i]
        rows = ex.fetch_ohlcv(pair, "1m", since=_ms(start), limit=1000)
        reqs += 1
        if rows:
            _insert(con, "candles_1m", _to_df(rows, symbol),
                    ["symbol", "ts", "open", "high", "low", "close", "volume"])
            covered_until = pd.to_datetime(rows[-1][0], unit="ms").floor("h")
        else:
            covered_until = start
        # salta las horas ya cubiertas por este bloque; la siguiente empieza fuera
        i += 1
        while i < len(need) and need[i] < covered_until:
            i += 1
        if reqs % 20 == 0:
            print(f"  1m: {reqs} peticiones, {len(need) - i} horas pendientes", end="\r")
    print(f"\n1m: {reqs} peticiones")
    return reqs


def load_candles(con, symbol):
    return con.execute(
        "SELECT ts, open, high, low, close, volume FROM candles_1h WHERE symbol=? ORDER BY ts", [symbol]
    ).df()


def load_funding(con, symbol):
    return con.execute("SELECT ts, rate FROM funding WHERE symbol=? ORDER BY ts", [symbol]).df()
