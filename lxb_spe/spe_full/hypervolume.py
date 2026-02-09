from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from .pareto import pareto_front


def hypervolume_2d(points: Sequence[np.ndarray], ref: np.ndarray) -> float:
    ref = np.asarray(ref, dtype=float)
    if ref.shape[0] != 2:
        raise ValueError("hypervolume_2d expects ref to have 2 dimensions")

    filtered: List[np.ndarray] = []
    for p in points:
        p = np.asarray(p, dtype=float)
        if p.shape[0] != 2:
            raise ValueError("hypervolume_2d expects points to have 2 dimensions")
        if p[0] > ref[0] and p[1] > ref[1]:
            filtered.append(p)
    if not filtered:
        return 0.0

    nd_idx = pareto_front(filtered)
    nd = [filtered[i] for i in nd_idx]
    nd.sort(key=lambda v: float(v[0]))

    hv = 0.0
    current_max_y = float(ref[1])
    for i in range(len(nd) - 1, -1, -1):
        x_i = float(nd[i][0])
        y_i = float(nd[i][1])
        if y_i > current_max_y:
            current_max_y = y_i
        x_prev = float(nd[i - 1][0]) if i > 0 else float(ref[0])
        hv += (x_i - x_prev) * (current_max_y - float(ref[1]))
    return float(max(hv, 0.0))


def hypervolume_contribution_2d(point: np.ndarray, front: Sequence[np.ndarray], ref: np.ndarray) -> float:
    point = np.asarray(point, dtype=float)
    ref = np.asarray(ref, dtype=float)
    hv_before = hypervolume_2d(front, ref)
    hv_after = hypervolume_2d(list(front) + [point], ref)
    return float(max(hv_after - hv_before, 0.0))

