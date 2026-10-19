"""Gráficos de resultados del estudio.

Las cifras se han copiado a mano de los informes que genera `python -m clefbt ...` (carpeta reports/).
Requiere matplotlib (no está en requirements.txt). Uso:  python docs/make_charts.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
os.makedirs(OUT, exist_ok=True)

# --- paleta validada (slots 1-3, modo claro) y tintas ---
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

names = {f.name for f in font_manager.fontManager.ttflist}
FONT = "Segoe UI" if "Segoe UI" in names else "DejaVu Sans"
plt.rcParams.update({"font.family": FONT, "axes.unicode_minus": True, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "text.color": INK, "axes.edgecolor": BASE})


def pct(x):
    return f"{x:.0f}".replace(".", ",")


def es(x, nd=3, sign=False):
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",").replace("-", "−")


# =========================================================================================
# FIGURA 1: ¿distingue lo que va a pasar? y ¿gana dinero?
# =========================================================================================
fig = plt.figure(figsize=(5.4, 6.75), dpi=200)
fig.text(0.05, 0.955, "Un modelo de decisión, probado donde no debía", fontsize=13.5, fontweight="bold", va="top")
fig.text(0.05, 0.918, "Mismos datos, mismas comisiones y validación sin mirar el futuro", fontsize=8, color=INK2, va="top")

# leyenda (identidad nunca solo por color: cada fila lleva además su nombre)
for lx, color, label in ((0.05, BLUE, "Clef"), (0.20, ORANGE, "ML clásico"), (0.42, AQUA, "Regla simple (RSI)")):
    fig.add_artist(Line2D([lx], [0.885], marker="o", markersize=5.5, color=color, linestyle="none", transform=fig.transFigure))
    fig.text(lx + 0.02, 0.885, label, fontsize=7.6, color=INK2, va="center")

# ---------------- Panel A: AUC ----------------
axA = fig.add_axes([0.30, 0.40, 0.66, 0.40])
rows = [  # (grupo/None, etiqueta, valor, color)
    ("Dirección a 24 h: ¿sube o baja?", None, None, None),
    (None, "Clef 27B", 0.458, BLUE),
    (None, "Clef-flash", 0.466, BLUE),
    (None, "ML clásico", 0.522, ORANGE),
    (None, "RSI (reversión)", 0.532, AQUA),
    ("TP antes que stop · en largo", None, None, None),
    (None, "Clef-flash", 0.514, BLUE),
    (None, "ML clásico", 0.578, ORANGE),
    ("TP antes que stop · en corto", None, None, None),
    (None, "Clef-flash", 0.501, BLUE),
    (None, "ML clásico", 0.573, ORANGE),
]
n = len(rows)
axA.set_xlim(0.42, 0.62)
axA.set_ylim(n - 0.4, -0.9)
axA.axvspan(0.42, 0.5, color="#f1f0ec", zorder=0)
axA.axvline(0.5, color=INK2, lw=1.2, zorder=1)
for x in (0.44, 0.46, 0.48, 0.52, 0.54, 0.56, 0.58, 0.60):
    axA.axvline(x, color=GRID, lw=0.6, zorder=0)
yt, yl = [], []
for i, (g, lab, v, c) in enumerate(rows):
    if g:
        axA.text(-0.379, i, g, fontsize=7.6, fontweight="bold", va="center", color=INK, clip_on=False,
                 transform=axA.get_yaxis_transform())      # x en fracción del eje, y en datos: alineado a la izquierda del todo
        continue
    axA.plot([0.5, v], [i, i], color=c, lw=2.4, solid_capstyle="round", zorder=2)
    axA.scatter([v], [i], s=62, color=c, edgecolor=SURFACE, linewidth=1.8, zorder=3)
    side = 1 if v >= 0.5 else -1
    axA.text(v + side * 0.0055, i, es(v), fontsize=8, ha="left" if side > 0 else "right", va="center", color=INK, fontweight="bold")
    yt.append(i)
    yl.append(lab)
axA.set_yticks(yt)
axA.set_yticklabels(yl, fontsize=7.8, color=INK)
axA.tick_params(axis="y", length=0, pad=4)
axA.set_xticks([0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60])
axA.set_xticklabels([es(x, 2) for x in (0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60)], fontsize=6.8, color=MUTED)
axA.tick_params(axis="x", length=0, pad=3)
for s in ("top", "right", "left", "bottom"):
    axA.spines[s].set_visible(False)
axA.text(0.5, -0.75, "azar", fontsize=7, color=INK2, ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.18", fc=SURFACE, ec="none"))
fig.text(0.05, 0.856, "¿Distingue lo que va a pasar?", fontsize=9.6, fontweight="bold", va="center")
fig.text(0.05, 0.834, "AUC: 0,50 = tirar una moneda · 1,00 = acierta siempre", fontsize=7.2, color=MUTED, va="center")

# ---------------- Panel B: resultado económico ----------------
axB = fig.add_axes([0.30, 0.095, 0.66, 0.205])
data = [  # etiqueta, mínimo, máximo, media, color
    ("Sin modelo\n(siempre largo)", None, None, -0.117, INK2),
    ("Clef-flash", -0.212, -0.122, -0.156, BLUE),
    ("ML clásico\n(20 variantes)", -0.182, 0.020, -0.081, ORANGE),
    ("ML + derivados\n(35 variantes)", -0.239, 0.082, -0.079, ORANGE),
]
axB.set_xlim(-0.30, 0.12)
axB.set_ylim(len(data) - 0.45, -0.75)
axB.axvspan(0, 0.12, color="#eaf5ee", zorder=0)
for x in (-0.25, -0.20, -0.15, -0.10, -0.05, 0.05, 0.10):
    axB.axvline(x, color=GRID, lw=0.6, zorder=0)
axB.axvline(0, color=INK2, lw=1.2, zorder=1)
for i, (lab, lo, hi, mean, c) in enumerate(data):
    if lo is not None:
        axB.plot([lo, hi], [i, i], color=c, lw=5, solid_capstyle="round", alpha=0.35, zorder=2)
    axB.scatter([mean], [i], s=62, color=c, edgecolor=SURFACE, linewidth=1.8, zorder=3)
    axB.text(mean, i - 0.36, es(mean, 2, True), fontsize=7.6, ha="center", va="center", color=INK, fontweight="bold")
axB.set_yticks(range(len(data)))
axB.set_yticklabels([d[0] for d in data], fontsize=7.4, color=INK, linespacing=1.15)
axB.tick_params(axis="y", length=0, pad=4)
axB.set_xticks([-0.25, -0.15, -0.05, 0.05, 0.10])
axB.set_xticklabels([es(x, 2, True) for x in (-0.25, -0.15, -0.05, 0.05, 0.10)], fontsize=6.8, color=MUTED)
axB.tick_params(axis="x", length=0, pad=3)
for s in ("top", "right", "left", "bottom"):
    axB.spines[s].set_visible(False)
axB.text(0.004, -0.42, "gana →", fontsize=6.8, color="#006300", ha="left", va="center")
axB.text(-0.004, -0.42, "← pierde", fontsize=6.8, color=INK2, ha="right", va="center")
fig.text(0.05, 0.352, "¿Gana dinero?", fontsize=9.6, fontweight="bold", va="center")
fig.text(0.05, 0.329, "Resultado medio por operación tras comisiones, en R (1 R = lo que arriesgas)", fontsize=7.2, color=MUTED, va="center")
fig.text(0.05, 0.036, "Barra: de la peor a la mejor combinación de modelo y umbral · punto: media. BTC, 24 h, comisión 0,11 % + funding.\n"
                      "Margen de error del AUC ≈ ±0,02. Muestras de Clef y de ML distintas.   github.com/Guille6520/btc-clef-study",
         fontsize=5.9, color=MUTED, va="center", linespacing=1.5)
fig.savefig(os.path.join(OUT, "resultados-1.png"), dpi=200, facecolor=SURFACE)
plt.close(fig)

# =========================================================================================
# FIGURA 2: a baja frecuencia solo baja la caída
# =========================================================================================
fig = plt.figure(figsize=(5.4, 5.4), dpi=200)
fig.text(0.05, 0.945, "Ninguna regla diaria supera con claridad a comprar y mantener", fontsize=11.2, fontweight="bold", va="top")
fig.text(0.05, 0.895, "Lo que sí consiguen la tendencia y la volatilidad objetivo es bajar la caída máxima", fontsize=7.8, color=INK2, va="top")
ax = fig.add_axes([0.36, 0.20, 0.60, 0.60])
strat = [  # nombre, caída máx (%), CAGR (%), énfasis
    ("Comprar y mantener", 79, 22.2, False),
    ("Tendencia SMA200", 68, 19.2, False),
    ("Volatilidad objetivo 40 %", 66, 27.0, False),
    ("Tendencia + vol. objetivo", 48, 21.5, True),
]
ax.set_xlim(0, 135)
ax.set_ylim(len(strat) - 0.4, -0.6)
for x in (20, 40, 60, 80):
    ax.axvline(x, color=GRID, lw=0.6, zorder=0)
for i, (nm, dd, cagr, hl) in enumerate(strat):
    ax.barh(i, dd, height=0.52, color=BLUE if not hl else "#1c5cab", zorder=2, edgecolor=SURFACE, linewidth=0)
    ax.text(dd + 1.5, i - 0.02, f"−{dd} %", fontsize=9, fontweight="bold", va="center", color=INK)
    ax.text(135, i - 0.02, f"+{cagr:.1f} %/año".replace(".", ","), fontsize=8, va="center", ha="right", color=INK2)
ax.set_yticks(range(len(strat)))
ax.set_yticklabels([s[0] for s in strat], fontsize=8, color=INK)
ax.tick_params(axis="y", length=0, pad=4)
ax.set_xticks([0, 20, 40, 60, 80])
ax.set_xticklabels(["0", "20 %", "40 %", "60 %", "80 %"], fontsize=6.8, color=MUTED)
ax.tick_params(axis="x", length=0, pad=3)
for s in ("top", "right", "left", "bottom"):
    ax.spines[s].set_visible(False)
ax.text(135, -0.55, "retorno anual", fontsize=6.8, color=MUTED, ha="right", va="center")
ax.text(0, -0.55, "caída máxima del capital", fontsize=6.8, color=MUTED, ha="left", va="center")
fig.text(0.05, 0.095, "BTC, marzo 2018 a octubre 2025 (el último año se ha dejado sin tocar). Decisión al cierre del día, comisión y funding incluidos.\n"
                      "El Sharpe es parecido en las cuatro (0,6 a 0,75) y sus intervalos se solapan: menos caída, no más ganancia.\n"
                      "github.com/Guille6520/btc-clef-study",
         fontsize=5.9, color=MUTED, va="center", linespacing=1.5)
fig.savefig(os.path.join(OUT, "resultados-2.png"), dpi=200, facecolor=SURFACE)
plt.close(fig)
print("ok", OUT)
