# Resultados: prueba `dir` con Clef-flash

Pregunta: ¿el precio a 24 h (apertura de la vela 24 h después de la de entrada) será mayor que el de entrada?
Muestra aleatoria de 3.977 velas de 1 h de BTC (2019-09 a 2026-10). Coste ida y vuelta supuesto: 0,11 % del nominal (≈ 0,177 ATR de media).

## Métricas globales

- Tasa base (sube): 0,521. P media de Clef: 0,438.
- Brier Clef 0,3405 (tasa base constante: 0,2495). LogLoss 0,9393. **AUC 0,4659.**
- Correlación de rangos entre P(sube) y el retorno a 24 h (en ATR): **-0,0153** (±0,031 al 95 % si fuera azar).
- Baseline logístico walk-forward (3.269 filas): AUC 0,5144, Brier 0,2512. Clef en esas filas: AUC 0,4673, Brier 0,3386.

## Calibración

| P de Clef | n | P media | Sube en la realidad |
|---|---|---|---|
| 0,0-0,3 | 1622 | 0,156 | 0,551 |
| 0,3-0,4 | 240 | 0,347 | 0,542 |
| 0,4-0,45 | 124 | 0,427 | 0,508 |
| 0,45-0,5 | 113 | 0,477 | 0,522 |
| 0,5-0,55 | 178 | 0,526 | 0,517 |
| 0,55-0,6 | 197 | 0,575 | 0,508 |
| 0,6-0,7 | 532 | 0,651 | 0,524 |
| 0,7-1,0 | 971 | 0,769 | 0,472 |

## Operar el signo (largo si P > 0,5, corto si P < 0,5), en unidades de ATR

| \|P-0,5\| ≥ | n | Aciertos | Media bruta | Media neta | IC90 neta |
|---|---|---|---|---|---|
| 0,00 | 3977 | 0,473 | 0,100 | -0,077 | [-0,184, 0,031] |
| 0,03 | 3802 | 0,473 | 0,096 | -0,081 | [-0,192, 0,029] |
| 0,05 | 3686 | 0,471 | 0,097 | -0,080 | [-0,192, 0,032] |
| 0,10 | 3365 | 0,468 | 0,107 | -0,068 | [-0,186, 0,050] |
| 0,15 | 3005 | 0,464 | 0,133 | -0,040 | [-0,162, 0,081] |
| 0,20 | 2592 | 0,458 | 0,101 | -0,070 | [-0,200, 0,060] |

## Lectura

- Clef-flash **no predice la dirección** a 24 h: el AUC es ≤ 0,5 y la correlación con el retorno real es indistinguible de cero.
- Está **peor calibrado que una moneda**: el LogLoss (0,94) supera al de predecir 50/50 (0,69) y su Brier es mucho peor que el de la tasa base.
- Los extremos apuntan ligeramente al lado contrario (muy bajista → sube el 55 %; muy alcista → sube el 47 %). No se interpreta como señal:
  la correlación es nula, las ventanas de 24 h de la muestra se solapan (el margen de error real es mayor que el nominal) y la inversión del
  signo no sería rentable (la media bruta pasaría de +0,10 a -0,10 ATR).
- Operar el signo, tras costes, da una media de -0,04 a -0,08 ATR por operación con intervalos que incluyen el cero: no hay evidencia de edge.
- Un modelo clásico (regresión logística walk-forward con las mismas features) llega apenas a AUC 0,514: la dirección a 24 h en BTC es
  casi impredecible con estas variables.
- Es un resultado para Clef-flash, zero-shot, con un estado y una pregunta concretos (para el modelo 27B, ver la sección siguiente). No concluye sobre otras formulaciones ni sobre fine-tuning.

## Clef 27B (596 velas aleatorias)

Mismo estado y misma pregunta, con el modelo grande (`@cf/cloudflare/clef`):

- Tasa base (sube): 0,503. P media de Clef: 0,427.
- Brier 0,3594 (constante: 0,2500), LogLoss 1,0164, **AUC 0,4578**.
- Correlación de rangos con el retorno a 24 h: **+0,014** (±0,080 si fuera azar).
- Sobreconfianza extrema: cuando dice 70-100 % (153 casos, P media 0,81) el precio subió el 45 %; cuando dice <30 % (251 casos), el 55 %.
- Operar su signo: media neta de +0,07 ATR con un IC90 de [-0,21, +0,36]: no distinguible de cero con esta muestra.

El modelo grande da el mismo resultado que el pequeño: no hay señal direccional utilizable.
