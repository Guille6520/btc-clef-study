import duckdb

from .config import DATA_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles_1h (
    symbol VARCHAR, ts TIMESTAMP, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE,
    PRIMARY KEY (symbol, ts)
);
CREATE TABLE IF NOT EXISTS candles_1m (
    symbol VARCHAR, ts TIMESTAMP, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE,
    PRIMARY KEY (symbol, ts)
);
-- ts = momento de liquidación del funding (redondeado a la hora)
CREATE TABLE IF NOT EXISTS funding (
    symbol VARCHAR, ts TIMESTAMP, rate DOUBLE,
    PRIMARY KEY (symbol, ts)
);
-- Resultado de cada operación hipotética. ts = apertura de la vela cuyo cierre es el momento de decisión.
-- Entrada: apertura de la vela siguiente. r_gross en múltiplos de R (riesgo = distancia al stop).
-- Si outcome='ambiguous': r_gross es el caso pesimista y r_gross_opt el optimista.
-- funding_frac: coste de funding como fracción del nominal (positivo = se paga).
CREATE TABLE IF NOT EXISTS labels (
    symbol VARCHAR, ts TIMESTAMP, direction VARCHAR, sl_atr DOUBLE, tp_atr DOUBLE, horizon_h INTEGER,
    outcome VARCHAR, resolved_by VARCHAR, entry_px DOUBLE, risk_frac DOUBLE,
    r_gross DOUBLE, r_gross_opt DOUBLE, funding_frac DOUBLE, exit_ts TIMESTAMP,
    y_pess INTEGER, y_opt INTEGER,
    PRIMARY KEY (symbol, ts, direction, sl_atr, tp_atr, horizon_h)
);
-- Cada respuesta de Clef, con la versión de estado y de preguntas para poder reutilizarla.
CREATE TABLE IF NOT EXISTS clef_calls (
    symbol VARCHAR, ts TIMESTAMP, model VARCHAR, state_version VARCHAR, qset_version VARCHAR,
    source VARCHAR, state VARCHAR, probs VARCHAR, raw VARCHAR, created_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (symbol, ts, model, state_version, qset_version)
);
"""


def connect(read_only=False):
    DATA_DIR.mkdir(exist_ok=True)
    con = duckdb.connect(str(DATA_DIR / "clefbt.duckdb"), read_only=read_only)
    if not read_only:
        con.execute(SCHEMA)
    return con
