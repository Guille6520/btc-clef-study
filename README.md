# Estudio del precio de BTC con Clef

¿Puede un modelo de decisión como **Clef** (Cloudflare Workers AI) filtrar buenas operaciones en BTC?
Este repositorio es un banco de pruebas para responderlo con datos y sin autoengaño: se le pregunta a Clef,
vela a vela, cómo acabaría una operación hipotética, se **guarda cada respuesta** y se compara con lo que pasó de verdad.

> **Estado:** investigación en curso. El primer resultado es **negativo** (ver más abajo). Publicar los resultados
> malos con la misma honestidad que los buenos es parte del objetivo.
>
> **Aviso:** esto no es consejo financiero ni un sistema de trading. Un backtest, aunque salga bien, no garantiza nada.

## La pregunta

Para cada vela de 1 h de BTC, al cierre, se le plantean a Clef preguntas sí/no con respuesta probabilística:

| Conjunto | Pregunta | Etiqueta real |
|---|---|---|
| `tpsl` | ¿Tocará el take profit antes que el stop en las próximas 24 h? (SL 1,5 × ATR, TP 3 × ATR) | primera barrera tocada |
| `dir` | ¿Estará el precio a 24 h por encima de la entrada? | precio de salida > entrada |

La configuración principal está fijada **antes** de ver resultados. La rejilla de SL/TP (1 / 1,5 / 2 × 1,5 / 2 / 3 ATR) es solo para sensibilidad.

## Metodología (lo que importa para que el resultado valga algo)

- **Sin fuga temporal.** Las features de la vela T usan datos hasta su cierre; la entrada es la **apertura de T+1**.
- **Empates TP/SL en la misma vela.** No se asigna a ciegas ni se descarta: se reproduce esa hora con velas de **1 minuto** y gana lo que
  toque primero. Lo que ni el 1 m decide queda `ambiguous` y se evalúa en el caso pesimista y en el optimista.
- **Realismo de ejecución.** Gaps (fill a la apertura si ya superó el stop), TP como orden límite que debe *atravesarse*, comisiones, slippage
  y **funding real** de la ida a la salida.
- **Estado anonimizado para el modelo.** Sin fechas ni nombre de moneda, precios en unidades de ATR. Reduce, **no elimina**, el riesgo de que el
  modelo recuerde el histórico (Clef se basa en Qwen, que ha visto años de BTC). La única validación sin fuga posible es el registro en vivo.
- **Evaluación honesta.** Calibración, AUC, comparación con la tasa base y con una **regresión logística walk-forward**, simulación con una
  posición a la vez, IC por bootstrap de bloques y test de permutación.
- **Todo guardado.** Cada llamada a Clef (estado enviado, probabilidades y respuesta cruda) queda en DuckDB con su versión; nada se paga dos veces.

## Resultados

### Prueba 1: `tpsl` con Clef-flash (muestra aleatoria de 1.988 velas, 2019-2026)

| | Largo | Corto |
|---|---|---|
| Tasa base (TP antes que SL) | 25,1 % | 24,8 % |
| **AUC de Clef** | **0,514** | **0,501** |
| AUC regresión logística (walk-forward) | 0,564 | 0,550 |
| Brier de Clef / de la tasa base | 0,212 / 0,188 | 0,226 / 0,187 |

- Clef-flash **no discrimina** (AUC ≈ 0,50, dentro del margen de error de ±0,03) y está **mal calibrado**: es optimista y peor que predecir siempre la tasa base.
- Las probabilidades altas no aciertan más (en largos, "60-70 %" acertó el 19 %).
- Simulando operar por umbral, **todas** las configuraciones pierden (-0,12 a -0,21 R por operación, con coste de 0,11 %) y el test de permutación no las distingue del azar.
- El modelo clásico saca un edge diminuto (AUC 0,55-0,56) que probablemente no sobrevive a los costes con un payoff 1:2.

Informe completo: [docs/resultados-tpsl.md](docs/resultados-tpsl.md).

### Prueba 2: `dir` (¿sube a 24 h?)

*Pendiente.* Sirve para separar "Clef no sabe de BTC" de "la pregunta TP/SL es demasiado difícil".

## Limitaciones conocidas

- Solo **Clef-flash**, **zero-shot**, con **un** formato de estado y **una** formulación de pregunta. No se puede concluir sobre Clef (27B) ni sobre otras formulaciones.
- Muestra de ~2.000 velas, no el histórico completo (~79.000). La pasada completa cuesta ~2 $ pero no se justifica sin señal en la muestra.
- Sin open interest histórico, opciones ni liquidaciones.
- Entrada simulada a mercado con coste conservador, sin modelar el fill de órdenes límite.
- Posible fuga por memorización del histórico (ver metodología).

## Reproducirlo (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env        # rellena CLOUDFLARE_ACCOUNT_ID y CLOUDFLARE_API_TOKEN
.\.venv\Scripts\python.exe -m clefbt download     # velas 1h + funding (Binance, público)
.\.venv\Scripts\python.exe -m clefbt label        # etiquetas TP/SL/tiempo, resuelve empates con 1m
.\.venv\Scripts\python.exe -m clefbt probe        # una llamada real a Clef: verifica el formato
.\.venv\Scripts\python.exe -m clefbt run --dry    # cuántas llamadas y tokens, antes de gastar
.\.venv\Scripts\python.exe -m clefbt run --sample 2000
.\.venv\Scripts\python.exe -m clefbt evaluate     # informe (se guarda en reports/)
# pregunta direccional
.\.venv\Scripts\python.exe -m clefbt run --qset dir --sample 2000
.\.venv\Scripts\python.exe -m clefbt evaluate --qset dir
```

Notas de coste (medidas): Clef-flash gasta ~2,55 neuronas por llamada de 2 preguntas (~740 tokens). El plan gratuito de Workers AI da 10.000
neuronas al día y la cuota es **por cuenta**; si se agota, `run` se detiene limpio y es reanudable.

## Estructura

```
config.toml          parámetros (SL/TP/plazo fijados, costes, umbrales, modelo)
clefbt/
  data.py            velas 1h/1m (Binance vía ccxt) y funding
  features.py        features y estado anonimizado para Clef
  labeling.py        etiquetado triple barrera sin sesgo de empate
  clef.py            cliente Workers AI con caché y control de cuota
  evaluate.py        calibración, baselines, simulación, permutación
docs/                resultados
```

## Próximos pasos

- Resultado de la prueba direccional.
- Clef 27B sobre la misma muestra; otras formulaciones del estado y de la pregunta.
- Clef como *feature* junto al modelo clásico (¿aporta algo que las features numéricas no tienen?).
- Registro en vivo cada hora (`clefbt live`), la única validación sin fuga.

## Licencia

MIT. Ver [LICENSE](LICENSE).
