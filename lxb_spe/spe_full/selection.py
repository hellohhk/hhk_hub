from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .genome import StructuredGenome
from .pareto import crowding_distance, nondominated_sort


@dataclass(frozen=True)
class NSGA2SelectionResult:
    selected: List[StructuredGenome]
    ranks: Dict[str, int]
    crowding: Dict[str, float]


def nsga2_select(pop: Sequence[StructuredGenome], mu: int) -> NSGA2SelectionResult:
    if mu <= 0:
        return NSGA2SelectionResult(selected=[], ranks={}, crowding={})

    points = [g.mu() for g in pop]
    fronts = nondominated_sort(points)

    selected: List[StructuredGenome] = []
    ranks: Dict[str, int] = {}
    crowding: Dict[str, float] = {}

    for rank, front in enumerate(fronts):
        if not front:
            continue
        dist = crowding_distance(points, front)
        for idx in front:
            ranks[pop[idx].uid] = rank
            crowding[pop[idx].uid] = dist.get(idx, 0.0)

        if len(selected) + len(front) <= mu:
            selected.extend([pop[i] for i in front])
            continue

        remaining = mu - len(selected)
        sorted_front = sorted(front, key=lambda i: dist.get(i, 0.0), reverse=True)
        selected.extend([pop[i] for i in sorted_front[:remaining]])
        break

    return NSGA2SelectionResult(selected=selected, ranks=ranks, crowding=crowding)


def tournament_select(
    pop: Sequence[StructuredGenome],
    *,
    ranks: Dict[str, int],
    crowding: Dict[str, float],
    rng: np.random.Generator,
    k: int = 2,
) -> StructuredGenome:
    if len(pop) == 1:
        return pop[0]
    candidates = [pop[int(rng.integers(0, len(pop)))] for _ in range(max(2, k))]

    def key(g: StructuredGenome) -> Tuple[int, float]:
        return (ranks.get(g.uid, 10**9), -crowding.get(g.uid, 0.0))

    return sorted(candidates, key=key)[0]

