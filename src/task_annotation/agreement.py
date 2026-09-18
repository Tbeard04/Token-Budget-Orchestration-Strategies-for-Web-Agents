"""
agreement.py - chance-corrected agreement statistics, implemented directly.

Cohen (1960) for kappa; Cohen (1968) for weighted kappa; Landis & Koch (1977) for the interpretation bands
"""
from __future__ import annotations


def confusion(a: list, b: list, labels: list) -> list[list[int]]:
    #Rows = rater a (human), cols = rater b (model)
    idx = {v: i for i, v in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for x, y in zip(a, b):
        m[idx[x]][idx[y]] += 1
    return m


def kappa(a: list, b: list, labels: list, weights: str = "none") -> float | None:
    #Cohen's kappa. weights: 'none', 'linear' or 'quadratic'
    n = len(a)
    if n == 0:
        return None
    k = len(labels)
    m = confusion(a, b, labels)

    rows = [sum(r) for r in m]
    cols = [sum(m[i][j] for i in range(k)) for j in range(k)]

    def w(i: int, j: int) -> float:
        if weights == "none":
            return 0.0 if i == j else 1.0
        d = abs(i - j) / (k - 1) if k > 1 else 0.0
        return d if weights == "linear" else d * d

    obs = sum(w(i, j) * m[i][j] for i in range(k) for j in range(k)) / n
    exp = sum(w(i, j) * rows[i] * cols[j] / n for i in range(k) for j in range(k)) / n

    if exp == 0:
        # constant and identical --> perfect; constant and different --> undefined
        return 1.0 if obs == 0 else None
    return 1.0 - obs / exp


def landis_koch(k: float | None) -> str:
    # interpretation bands
    if k is None:
        return "undefined"
    if k < 0.00:
        return "poor (worse than chance)"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def pearson(xs: list, ys: list) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sxy / (sxx * syy) ** 0.5


def _ranks(vs: list) -> list[float]:
    #Average ranks, ties shared
    order = sorted(range(len(vs)), key=lambda i: vs[i])
    ranks = [0.0] * len(vs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def spearman(xs: list, ys: list) -> float | None:
    return pearson(_ranks(xs), _ranks(ys)) if len(xs) >= 3 else None