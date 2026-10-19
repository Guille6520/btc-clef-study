# Resultados: ML clásico, derivados, estrategias diarias y una pista que no resiste

Todo con BTC, costes del 0,11 % ida y vuelta más funding real, hiperparámetros fijos (sin ajustar con el test) y la **reserva ciega del último año
(desde 2025-10-14) sin tocar**. Los informes crudos se generan con `bench`, `bench --derivs`, `daily`, `contrarian` y `extremes`.

## 1. Protocolo

- **Walk-forward por años** con embargo de 25 h: se entrena con todo lo anterior a cada año de test menos 25 h, para que ninguna etiqueta de
  24 h del entrenamiento solape con el test (comprobado con tests, incluidas mutaciones).
- Modelos: heurísticas simples (tendencia, momentum, reversión), regresión logística y *gradient boosting* con hiperparámetros conservadores y fijos.
- Tres objetivos: dirección a 24 h, TP-antes-que-SL (largo y corto; SL 1,5 ATR, TP 3 ATR, 24 h) y volatilidad futura.
- Intervalos por *bootstrap* de bloques semanales; contraste de permutación en las simulaciones.

## 2. ML clásico frente a Clef (BTC, 2021 a oct-2025, ~41.700 velas fuera de muestra)

| Objetivo | Clef-flash | Mejor ML clásico | Heurística simple |
|---|---|---|---|
| TP antes que SL, largo (AUC) | 0,514 | **0,578** [0,564-0,591] | tendencia 0,525 |
| TP antes que SL, corto (AUC) | 0,501 | **0,573** [0,555-0,589] | tendencia 0,529 |
| Dirección a 24 h (AUC) | 0,466 | 0,522 [0,508-0,537] | **reversión (-RSI) 0,532** [0,516-0,548] |
| Volatilidad futura (Spearman, R² fuera de muestra) | no probado | **0,46** (0,53 en 2024-25), R² 0,22 | |

- El ML clásico **bate claramente a Clef** en TP/SL y es **consistente en todos los años** (AUC 0,55-0,60).
- Como el AUC sale parecido en largo y en corto, el modelo está captando "habrá movimiento", no la dirección: la **volatilidad** es lo que se puede predecir.
- La dirección a 24 h es casi impredecible (AUC 0,51-0,53), incluso con modelos más complejos.

## 3. Pero predecir no es ganar dinero

Simulación TP/SL (una posición a la vez, costes y funding incluidos):

- Siempre largo: **-0,117 R** por operación; siempre corto: **-0,123 R**.
- Los modelos **reducen la pérdida** (el contraste de permutación suele dar p < 0,05) pero la media sigue siendo negativa en casi todos los umbrales:
  entre -0,02 y -0,18 R. La excepción (logit, umbral 0,50: +0,02 R) tiene 158 operaciones y un IC de [-0,18, +0,20].
- Operar el signo de la dirección: bruta positiva solo en los umbrales altos y **neta -0,03 ATR** [-0,10, +0,04]. Los costes (≈ 0,18 ATR por operación)
  se comen la señal.

## 4. Derivados y contexto (2023 a oct-2025, 24.057 velas comunes)

Se añadieron volumen de compradores agresores y prima del perpetuo (Binance), open interest y ratios largo/corto (Binance), DVOL (Deribit),
miedo y avaricia, y VIX/Nasdaq/dólar/bono a 10 años (FRED), con retrasos para no mirar el futuro.

| Conjunto | AUC TP/SL largo | AUC TP/SL corto | R² volatilidad |
|---|---|---|---|
| v2 (price + vol + estructura + calendario) | 0,588 | 0,596 | 0,293 |
| v2 + perp | 0,594 | 0,607 | 0,292 |
| v2 + perp + OI | 0,560 | 0,582 | 0,250 |
| v2 + perp + OI + DVOL | 0,552 | 0,557 | 0,241 |
| v2 + macro | 0,566 | 0,602 | 0,271 |
| todo | 0,553 | 0,559 | 0,261 |

**Los derivados no mejoran nada y varios empeoran** (más variables, más sobreajuste con hiperparámetros fijos). En dirección, todos los conjuntos
quedan en AUC 0,51-0,53, por debajo de la heurística de reversión (0,537). En la simulación, la mejor celda (+0,08 R, 184 operaciones, IC [-0,11, +0,28])
sale de 35 combinaciones y no es significativa.

## 5. Estrategias diarias (2018-03 a oct-2025, 2.779 días; 8 variantes declaradas, parámetros típicos)

| Estrategia | CAGR | Sharpe | Peor caída |
|---|---|---|---|
| Comprar y mantener | +22,2 % | 0,64 | -79 % |
| Tendencia SMA200 (largo/plano) | +19,2 % | 0,62 | -68 % |
| Tendencia SMA200 (largo/corto) | -5,9 % | 0,22 | -88 % |
| Cruce SMA50/200 | +14,9 % | 0,53 | -68 % |
| Momentum 90 d | +24,4 % | 0,71 | -68 % |
| Momentum 180 d | +18,4 % | 0,59 | -68 % |
| Volatilidad objetivo 40 % (siempre largo) | +27,0 % | 0,75 | -66 % |
| **Tendencia SMA200 + vol. objetivo 40 %** | +21,5 % | 0,74 | **-48 %** |

Ninguna bate a comprar y mantener con significación (los IC del Sharpe se solapan por completo). Lo que sí aportan la tendencia y la volatilidad objetivo es
**menos caída con un retorno parecido**: gestión de riesgo, no *alpha*.

**Carry de funding** (largo spot + corto perpetuo, 1x): rendimiento anualizado del funding de +17 % (2020), +31 % (2021), +4 % (2022), +8 % (2023),
+12 % (2024) y +5 % (2025). En 2025 es comparable al bono a 10 años (≈ 5 %), y no incluye la base, el margen extra ni el riesgo de exchange.

## 6. Una pista que no resiste: continuación en los extremos de RSI

Al ver que la regla *contraria* por RSI perdía en los 7 años observados (-0,54 ATR bruto, cada vez peor cuanto más extremo el RSI), se probó la
*continuación* (RSI < 30 → corto, RSI > 70 → largo, 24 h, una posición a la vez) y se replicó donde no se había mirado:

| Segmento | n | Bruta ATR | Neta ATR (comisión 0,11 %) [IC90] | Años > 0 |
|---|---|---|---|---|
| BTC 2017-09..2019-09 (no visto) | 280 | +0,94 | +0,82 [+0,32, +1,33] | 3/3 |
| BTC 2019-09..2025-10 (visto) | 832 | +0,54 | +0,38 [+0,10, +0,65] | 7/7 |
| ETH 2018..2025 (no visto) | 1116 | +0,42 | +0,30 [+0,10, +0,51] | 7/8 |
| SOL 2020..2025 (no visto) | 721 | +0,21 | +0,14 [-0,10, +0,38] | 4/6 |

Parecía una pista sólida (consistente, con respuesta creciente a la extremidad y replicada). **Pero la robustez la desmonta como estrategia:**

- Es **cola gruesa**: acierta ~50 %; la mediana es +0,15 ATR en BTC y **negativa en ETH y SOL**; las 10 mejores operaciones aportan el 51-68 % del total en BTC
  y el 101 % en SOL.
- **Sin stop, la peor operación es -30 ATR (-16 % del precio)** y las caídas del capital a 1x van del -44 % (BTC) al -96 % (SOL).
- **No tolera una ejecución algo peor:** en BTC, +21 %/año con la comisión supuesta, +5,6 % con 0,1 % extra de deslizamiento y -7,9 % con 0,2 %. ETH y SOL son
  planos o negativos incluso con ejecución ideal.
- A 1x en BTC rinde **+21 %/año con Sharpe 0,65, frente a +22 % y 0,64 de comprar y mantener**: no añade nada, solo reduce la caída (-44 % frente a -79 %).

Es un hecho estilizado (tras extremos de pánico o euforia el movimiento tiende a continuar, dominado por pocos días) y no un edge explotable tal cual.

## 7. Conclusiones

1. **Clef zero-shot no tiene señal** (ni TP/SL ni dirección), y un ML clásico simple lo supera con claridad.
2. **La volatilidad es predecible; la dirección, casi no.** Con TP/SL simétrico eso mejora el ranking pero no el resultado económico tras costes.
3. **Los datos de derivados no aportan** sobre las variables de precio con este protocolo.
4. **A baja frecuencia los costes casi desaparecen**, pero ninguna regla estándar bate a comprar y mantener; solo reduce la caída.
5. **Lo realista para ganar dinero es gestión de riesgo** (tendencia + volatilidad objetivo, dimensionado por la volatilidad predicha) más, quizá, *carry* si compensa frente a
   la renta fija; no una predicción direccional rentable.

## 8. Limitaciones

- Una sola moneda para casi todo (BTC) y un solo mercado; ETH y SOL solo para la pista del RSI.
- Hiperparámetros fijos: un ajuste fino podría mejorar el AUC, pero también el riesgo de sobreajuste.
- No se modela el fill de órdenes límite ni la selección adversa; los escenarios de comisión baja son optimistas.
- Se probaron muchas variantes (la lista se declara arriba); ninguna celda concreta debe tomarse como hallazgo.
- Los datos de 2017-2021 son un mercado alcista excepcional.
- La reserva ciega (desde 2025-10-14) sigue sin usar. Debe evaluarse **una sola vez**, y solo con una estrategia ya decidida.
