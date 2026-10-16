import argparse

from . import clef, config, data, evaluate, labeling
from .db import connect


def main():
    ap = argparse.ArgumentParser(prog="clefbt", description="Backtest de señales de Clef sobre BTC")
    ap.add_argument("cmd", choices=["init", "download", "label", "probe", "run", "live", "evaluate", "stats"])
    ap.add_argument("--symbol", help="por defecto, todos los de config.toml")
    ap.add_argument("--model", help="clef | clef-flash (por defecto, el de config.toml)")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--sample", type=int, help="run: solo N velas aleatorias (para pruebas baratas)")
    ap.add_argument("--dry", action="store_true", help="run: solo contar llamadas y estimar tokens")
    ap.add_argument("--source", help="evaluate: backtest | live")
    ap.add_argument("--qset", choices=["tpsl", "dir"],
                    help="conjunto de preguntas: tpsl (TP antes que SL, por defecto) | dir (¿sube a 24h?)")
    a = ap.parse_args()

    cfg = config.load()
    if a.qset:
        cfg["clef"]["question_set"] = a.qset
    con = connect()
    symbols = [a.symbol] if a.symbol else cfg["data"]["symbols"]

    if a.cmd == "init":
        print("Base de datos lista en", config.DATA_DIR)
    elif a.cmd == "download":
        for s in symbols:
            data.download_candles_1h(con, cfg, s)
            data.download_funding(con, cfg, s)
    elif a.cmd == "label":
        for s in symbols:
            labeling.run_labeling(con, cfg, s)
    elif a.cmd == "probe":
        clef.probe(cfg, symbols[0], con)
    elif a.cmd == "run":
        for s in symbols:
            clef.run(cfg, con, s, a.start, a.end, a.sample, a.dry, model=a.model)
    elif a.cmd == "live":
        for s in symbols:
            clef.live(cfg, con, s)
    elif a.cmd == "evaluate":
        for s in symbols:
            evaluate.report_to_file(con, cfg, s, model=a.model, source=a.source, start=a.start, end=a.end)
    elif a.cmd == "stats":
        for t in ("candles_1h", "candles_1m", "funding", "labels", "clef_calls"):
            print(f"{t:12s}", con.execute(f"SELECT count(*) FROM {t}").fetchone()[0])


if __name__ == "__main__":
    main()
