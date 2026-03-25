from dataclasses import dataclass
from typing import Any, List, Sequence
import numpy as np
from .pareto import _dominates

@dataclass
class NSGA2SelectionResult:
    selected: List[Any]
    ranks: List[int]
    crowding: List[float]

def nsga2_select(population: Sequence[Any], mu: int) -> NSGA2SelectionResult:
    if not population:
        return NSGA2SelectionResult([], [], [])
    pop = list(population)
    n = len(pop)
    pts = [p.mu() for p in pop]

    domination_counts = [0] * n
    dominated_lists = [[] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(pts[i], pts[j]):
                dominated_lists[i].append(j)
            elif _dominates(pts[j], pts[i]):
                domination_counts[i] += 1

    fronts = []
    current_front = [i for i in range(n) if domination_counts[i] == 0]
    while current_front:
        fronts.append(current_front)
        next_front = []
        for i in current_front:
            for j in dominated_lists[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        current_front = next_front

    ranks = [0] * n
    for rank, f in enumerate(fronts):
        for i in f:
            ranks[i] = rank

    crowding = [0.0] * n
    dim = len(pts[0])
    for f in fronts:
        if len(f) <= 2:
            for i in f:
                crowding[i] = float("inf")
            continue
        for d in range(dim):
            sorted_f = sorted(f, key=lambda idx: pts[idx][d])
            crowding[sorted_f[0]] = float("inf")
            crowding[sorted_f[-1]] = float("inf")
            min_val = pts[sorted_f[0]][d]
            max_val = pts[sorted_f[-1]][d]
            if max_val == min_val:
                continue
            for i in range(1, len(sorted_f) - 1):
                idx = sorted_f[i]
                prev_idx = sorted_f[i - 1]
                next_idx = sorted_f[i + 1]
                val_diff = pts[next_idx][d] - pts[prev_idx][d]
                crowding[idx] += val_diff / (max_val - min_val)

    selected_indices = []
    for f in fronts:
        if len(selected_indices) + len(f) <= mu:
            selected_indices.extend(f)
        else:
            sorted_f = sorted(f, key=lambda idx: crowding[idx], reverse=True)
            needed = mu - len(selected_indices)
            selected_indices.extend(sorted_f[:needed])
            break

    selected = [pop[i] for i in selected_indices]
    out_ranks = [ranks[i] for i in selected_indices]
    out_crowding = [crowding[i] for i in selected_indices]
    return NSGA2SelectionResult(selected=selected, ranks=out_ranks, crowding=out_crowding)

def tournament_select(population: Sequence[Any], ranks: Sequence[int], crowding: Sequence[float], rng: np.random.Generator, k: int = 2) -> Any:
    indices = rng.choice(len(population), size=k, replace=False)
    best_idx = indices[0]
    for idx in indices[1:]:
        if ranks[idx] < ranks[best_idx]:
            best_idx = idx
        elif ranks[idx] == ranks[best_idx] and crowding[idx] > crowding[best_idx]:
            best_idx = idx
    return population[best_idx]