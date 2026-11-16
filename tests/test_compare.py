"""Comparación Clef vs ML sobre las mismas velas: estadística pareada y remuestreo por semanas."""
import numpy as np
import pandas as pd

from clefbt import compare


def test_week_ids_group_by_week():
    idx = pd.date_range("2024-01-01", periods=24 * 14, freq="h")      # 2024-01-01 es lunes
    w = compare.week_ids(idx)
    assert len(set(w)) in (2, 3)
    assert (np.diff(w) >= 0).all()
    assert w[0] == w[1] and w[-1] > w[0]


def test_cluster_ci_constant_statistic_has_zero_width():
    groups = np.repeat(np.arange(20), 5)
    lo, hi = compare.cluster_ci(lambda i: 3.0, groups, 50, np.random.default_rng(0))
    assert lo == hi == 3.0


def test_cluster_ci_brackets_the_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(1.0, 1.0, 2000)
    groups = np.repeat(np.arange(200), 10)
    lo, hi = compare.cluster_ci(lambda i: x[i].mean(), groups, 300, rng)
    assert lo < x.mean() < hi
    assert hi - lo < 0.3


def test_paired_auc_identical_models_have_zero_difference():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 1000)
    p = rng.random(1000)
    groups = np.repeat(np.arange(100), 10)
    a, b, d, lo, hi = compare.paired_auc(y, p, p.copy(), groups, 100, rng)
    assert a == b and d == 0 and lo == hi == 0


def test_paired_auc_detects_a_better_model():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 3000)
    informative = y + rng.normal(0, 0.8, 3000)
    noise = rng.random(3000)
    groups = np.repeat(np.arange(300), 10)
    a_ref, a_oth, d, lo, hi = compare.paired_auc(y, noise, informative, groups, 200, rng)
    assert abs(a_ref - 0.5) < 0.05 and a_oth > 0.7
    assert d > 0 and lo > 0                      # el IC de la diferencia excluye 0
