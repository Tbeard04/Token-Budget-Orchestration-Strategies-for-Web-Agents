"""
agreement.py - chance-corrected agreement statistics, implemented directly.

Cohen (1960) for kappa; Cohen (1968) for weighted kappa; Landis & Koch (1977) for the interpretation bands
"""
from __future__ import annotations

#function to create the confusion matrix
def confusion(a: list, b: list, labels: list) -> list[list[int]]:
    #Rows = rater a (human), cols = rater b (model)
    idx = {v: i for i, v in enumerate(labels)}

    m = [[0] * len(labels) for _ in labels]
    for x, y in zip(a, b):
        m[idx[x]][idx[y]] += 1
    return m

#function to calculate the Cohen's kappa
def kappa(a: list, b: list, labels: list, weights: str = "none") -> float | None:
    #Cohen's kappa. weights: 'none', 'linear' or 'quadratic'
    n = len(a)
    if n == 0:
        return None
    k = len(labels)
    m = confusion(a, b, labels)
    rows = [sum(r) for r in m]
    cols = [sum(m[i][j] for i in range(k)) for j in range(k)]

    #function to calculate the weight
    def w(i: int, j: int) -> float:
        #if the weights are none
        if weights == "none":
            return 0.0 if i == j else 1.0
        #calculate the difference
        d = abs(i - j) / (k - 1) if k > 1 else 0.0
        return d if weights == "linear" else d * d

    #calculate the observed agreement
    obs = sum(w(i, j) * m[i][j] for i in range(k) for j in range(k)) / n
    #calculate the expected agreement
    exp = sum(w(i, j) * rows[i] * cols[j] / n for i in range(k) for j in range(k)) / n

    #if the expected agreement is 0
    if exp == 0:
        #if the observed agreement is 0
        # constant and identical --> perfect; constant and different --> undefined
        return 1.0 if obs == 0 else None
    return 1.0 - obs / exp

#function to calculate the Landis & Koch interpretation bands
def landis_koch(k: float | None) -> str:
    #if the kappa is none
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

#function to calculate the Pearson correlation coefficient
def pearson(xs: list, ys: list) -> float | None:
    #calculate the number of items
    n = len(xs)
    #if the number of items is less than 3
    if n < 3:
        return None
    #calculate the mean of the xs and ys
    mx, my = sum(xs) / n, sum(ys) / n
    #calculate the sum of the products of the differences
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    #calculate the sum of the squares of the differences
    sxx = sum((x - mx) ** 2 for x in xs)
    #calculate the sum of the squares of the differences
    syy = sum((y - my) ** 2 for y in ys)
    #if the sum of the squares of the differences is 0
    if sxx == 0 or syy == 0:
        return None
    #calculate the Pearson correlation coefficient
    return sxy / (sxx * syy) ** 0.5

#function to calculate the average ranks
def _ranks(vs: list) -> list[float]:
    #calculate the order of the vs
    order = sorted(range(len(vs)), key=lambda i: vs[i])
    #create a list to store the ranks
    ranks = [0.0] * len(vs)
    #initialize the index
    i = 0
    #while the index is less than the length of the order
    while i < len(order):
        #initialize the index j
        j = i
        #while the index j + 1 is less than the length of the order and the value at the index j + 1 is equal to the value at the index i
        while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
            j += 1
        #calculate the average rank
        avg = (i + j) / 2 + 1
        #for each index in the range i to j + 1
        for t in range(i, j + 1):
            #set the rank at the index t to the average rank
            ranks[order[t]] = avg
        #increment the index i
        i = j + 1
    return ranks

#function to calculate the Spearman correlation coefficient
def spearman(xs: list, ys: list) -> float | None:
    #if the number of items is less than 3
    if len(xs) < 3:
        return None
    #calculate the Pearson correlation coefficient
    return pearson(_ranks(xs), _ranks(ys)) if len(xs) >= 3 else None