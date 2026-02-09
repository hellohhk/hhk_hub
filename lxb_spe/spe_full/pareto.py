from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return bool(np.all(a >= b) and np.any(a > b))


def pareto_front(points: Sequence[np.ndarray]) -> List[int]:
    idxs: List[int] = []
    for i, p in enumerate(points):
        dominated_flag = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if dominates(q, p):
                dominated_flag = True
                break
        if not dominated_flag:
            idxs.append(i)
    return idxs


def nondominated_sort(points: Sequence[np.ndarray]) -> List[List[int]]:
    n = len(points)
    S: List[List[int]] = [[] for _ in range(n)]
    n_dom = [0] * n
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(points[p], points[q]):
                S[p].append(q)
            elif dominates(points[q], points[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)

    return fronts


def crowding_distance(points: Sequence[np.ndarray], front: Sequence[int]) -> Dict[int, float]:
    if not front:
        return {}
    m = len(points[front[0]])
    dist: Dict[int, float] = {i: 0.0 for i in front}

    for k in range(m):
        sorted_idx = sorted(front, key=lambda i: points[i][k])
        dist[sorted_idx[0]] = float("inf")
        dist[sorted_idx[-1]] = float("inf")
        min_v = float(points[sorted_idx[0]][k])
        max_v = float(points[sorted_idx[-1]][k])
        if max_v == min_v:
            continue
        for j in range(1, len(sorted_idx) - 1):
            prev_v = float(points[sorted_idx[j - 1]][k])
            next_v = float(points[sorted_idx[j + 1]][k])
            dist[sorted_idx[j]] += (next_v - prev_v) / (max_v - min_v)

    return dist

