# Resultados: prueba `tpsl` con Clef-flash

Muestra aleatoria de 1.988 velas de 1 h de BTC (2019-09 a 2026-10). SL 1,5 × ATR, TP 3 × ATR, plazo 24 h,
coste ida y vuelta 0,11 % del nominal más funding real. Velas ambiguas sin resolver: 0.

## Largo

- Tasa base (TP antes que SL): 0,251. P media de Clef: 0,273.
- Brier Clef 0,2123 (tasa base constante: 0,1880). LogLoss 0,6241. **AUC 0,5144.**
- Baseline logístico walk-forward (1.384 filas): AUC 0,5641, Brier 0,1919. Clef en esas filas: AUC 0,5125, Brier 0,2182.

| P de Clef | n | P media | Acierto real |
|---|---|---|---|
| 0,0-0,2 | 902 | 0,130 | 0,235 |
| 0,2-0,3 | 296 | 0,248 | 0,280 |
| 0,3-0,4 | 286 | 0,350 | 0,280 |
| 0,4-0,5 | 274 | 0,447 | 0,252 |
| 0,5-0,6 | 183 | 0,543 | 0,251 |
| 0,6-0,7 | 47 | 0,642 | 0,191 |

## Corto

- Tasa base: 0,248. P media de Clef: 0,348.
- Brier Clef 0,2260 (constante: 0,1867). LogLoss 0,6461. **AUC 0,5011.**
- Baseline logístico walk-forward (1.384 filas): AUC 0,5501, Brier 0,1887. Clef en esas filas: AUC 0,5151, Brier 0,2231.

| P de Clef | n | P media | Acierto real |
|---|---|---|---|
| 0,0-0,2 | 453 | 0,157 | 0,236 |
| 0,2-0,3 | 538 | 0,245 | 0,264 |
| 0,3-0,4 | 306 | 0,348 | 0,248 |
| 0,4-0,5 | 215 | 0,448 | 0,251 |
| 0,5-0,6 | 245 | 0,548 | 0,241 |
| 0,6-0,7 | 194 | 0,646 | 0,227 |
| 0,7-0,8 | 37 | 0,727 | 0,324 |

## Simulación (una posición a la vez, costes y funding incluidos)

| Umbral | n | Media R | Suma R | Acierto | PF | Máx. DD (R) | IC90 media R | p permutación |
|---|---|---|---|---|---|---|---|---|
| 0,25 | 1445 | -0,122 | -175,9 | 0,363 | 0,816 | 175,9 | [-0,18, -0,06] | 0,410 |
| 0,30 | 1326 | -0,129 | -171,7 | 0,359 | 0,805 | 171,7 | [-0,19, -0,07] | 0,570 |
| 0,35 | 1149 | -0,127 | -146,1 | 0,358 | 0,808 | 146,2 | [-0,19, -0,06] | 0,575 |
| 0,40 | 970 | -0,150 | -146,0 | 0,346 | 0,777 | 150,3 | [-0,22, -0,08] | 0,785 |
| 0,45 | 772 | -0,161 | -124,3 | 0,339 | 0,763 | 129,9 | [-0,24, -0,09] | 0,815 |
| 0,50 | 601 | -0,192 | -115,5 | 0,326 | 0,723 | 120,5 | [-0,28, -0,11] | 0,955 |
| 0,60 | 242 | -0,212 | -51,3 | 0,318 | 0,696 | 53,8 | [-0,34, -0,08] | 0,875 |

Referencias sin filtro: siempre largo -0,114 R de media; siempre corto -0,132 R.

## Lectura

- AUC ≈ 0,50 con un margen de ±0,03: **Clef-flash no distingue** buenas de malas operaciones en esta tarea.
- Está **mal calibrado** (optimista) y su Brier es peor que predecir siempre la tasa base.
- Todas las configuraciones pierden dinero tras costes, y empeoran al subir el umbral. El test de permutación no las separa del azar.
- La simulación sobre una muestra dispersa no equivale a una pasada real: vale el **signo**, no las cifras absolutas de R ni de operaciones.
- Es un resultado para Clef-flash, zero-shot, con un estado y una pregunta concretos. No concluye sobre otros modelos ni formulaciones.
