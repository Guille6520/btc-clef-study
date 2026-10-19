# Estudio del precio de BTC con Clef

[![tests](https://github.com/Guille6520/btc-clef-study/actions/workflows/tests.yml/badge.svg)](https://github.com/Guille6520/btc-clef-study/actions/workflows/tests.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

¿Puede un modelo de decisión como **Clef** (Cloudflare Workers AI) filtrar buenas operaciones en BTC?
Este repositorio es un banco de pruebas para responderlo con datos y sin autoengaño: se le pregunta a Clef,
vela a vela, cómo acabaría una operación hipotética, se **guarda cada respuesta** y se compara con lo que pasó de verdad.

> **Estado:** investigación en curso. Los dos primeros resultados son **negativos** (ver más abajo). Publicar los resultados
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
- **Estado anonimizado para el modelo.** Sin fechas ni precios absolutos (todo va en unidades de ATR), **pero el estado sí indica que es BTC**. Reduce, **no elimina**, el riesgo de que el
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

### Prueba 2: `dir` con Clef-flash (¿sube a 24 h?; muestra aleatoria de 3.977 velas)

Sirve para separar "Clef no sabe de BTC" de "la pregunta TP/SL es demasiado difícil". Tampoco aquí hay señal:

| | Resultado |
|---|---|
| Tasa base (el precio sube a 24 h) | 52,1 % |
| P media de Clef | 43,8 % |
| **AUC de Clef** | **0,466** |
| Correlación de rangos con el retorno a 24 h | -0,015 (ruido: ±0,031) |
| AUC regresión logística (walk-forward) | 0,514 |
| Brier de Clef / de la tasa base | 0,341 / 0,250 |
| Operar el signo de Clef | 47,3 % de aciertos, **-0,08 ATR netos** por operación (IC90 [-0,18, +0,03]) |

- Clef-flash no predice la dirección y está peor calibrado que una moneda (LogLoss 0,94 frente a 0,69).
- Los extremos apuntan algo hacia el lado equivocado (cuando dice <30 % el precio subió el 55 %; cuando dice >70 %, el 47 %), pero la correlación
  con el retorno es nula y las ventanas de 24 h se solapan. **No es una señal utilizable**: invertir el signo no sería rentable (la media bruta
  pasaría de +0,10 a -0,10 ATR) y elegir el sentido a posteriori sería precisamente el sesgo que este estudio quiere evitar.
- Hasta con un modelo clásico, la dirección a 24 h es casi impredecible con estas features (AUC 0,514).

Informe completo: [docs/resultados-dir.md](docs/resultados-dir.md).

**Conclusión provisional:** con Clef-flash, zero-shot, con este estado y estas preguntas, no hay nada que explotar, ni para filtrar TP/SL ni
para predecir la dirección. Por eso no se ha lanzado la pasada completa de ~79.000 velas.

## Limitaciones conocidas

- Solo **Clef-flash**, **zero-shot**, con **un** formato de estado y **una** formulación de pregunta. No se puede concluir sobre Clef (27B) ni sobre otras formulaciones.
- Muestras aleatorias de ~2.000 (`tpsl`) y ~4.000 (`dir`) velas, no el histórico completo (~79.000). La pasada completa cuesta ~2 $ pero no se justifica sin señal en la muestra.
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

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -v
```

32 tests, **offline** (velas sintéticas, DuckDB en memoria y respuestas simuladas de Clef: sin red, claves ni datos descargados), y se ejecutan
en GitHub Actions en cada push y pull request. Se centran en lo que, si falla, da resultados falsos sin avisar:

- **Etiquetado:** TP/SL/tiempo, entrada en la apertura siguiente, gaps, empates en la misma vela (resueltos con 1m o dejados como `ambiguous`,
  nunca adivinados), TP que debe *atravesarse* y stop que cuenta al alcanzarse, y funding solo durante la operación.
- **Sin fuga temporal:** las features hasta la vela *t* no cambian si se alteran las velas posteriores; el estado no lleva fechas ni precios absolutos.
- **Evaluación:** AUC, Brier, LogLoss y la simulación con una sola posición a la vez.
- **Cliente de Clef:** versiones de caché estables, parseo de la respuesta real, la cuota agotada (4006) detiene sin reintentar y el token nunca aparece en un error.

Los tests se validaron con *mutaciones*: se introdujeron a propósito 14 errores en una copia del código (ignorar el gap, contar el TP al rozarlo,
mirar el futuro en una feature, filtrar el token…) y **la suite detectó los 14**.

## Estructura

```
config.toml          parámetros (SL/TP/plazo fijados, costes, umbrales, modelo)
tests/               suite offline (pytest)
.github/workflows/   CI: tests en cada push y pull request
clefbt/
  data.py            velas 1h/1m (Binance vía ccxt) y funding
  features.py        features y estado anonimizado para Clef
  labeling.py        etiquetado triple barrera sin sesgo de empate
  clef.py            cliente Workers AI con caché y control de cuota
  evaluate.py        calibración, baselines, simulación, permutación
docs/                resultados
```

## Próximos pasos

- Variante del estado sin mencionar el activo (`state_version` v2), para medir cuánto del resultado depende de saber que es BTC.
- Clef 27B sobre la misma muestra; otras formulaciones del estado y de la pregunta.
- Clef como *feature* junto al modelo clásico (¿aporta algo que las features numéricas no tienen?).
- Registro en vivo cada hora (`clefbt live`), la única validación sin fuga.

## Licencia

MIT. Ver [LICENSE](LICENSE).
