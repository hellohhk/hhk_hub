from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
from .hypervolume import hypervolume_2d


@dataclass
class HVCUCBScheduleConfig:
    beta: float = 0.4
    ref_point: np.ndarray = np.array([0.0, 0.0])


def _compute_hvc(pts: Sequence[np.ndarray], ref_point: np.ndarray, idx: int) -> float:
    base_hv = hypervolume_2d(pts, ref_point)
    pts_without = [p for i, p in enumerate(pts) if i != idx]
    hv_without = hypervolume_2d(pts_without, ref_point)
    return max(0.0, base_hv - hv_without)


def pick_hvc_ucb(population: Sequence[Any], cfg: HVCUCBScheduleConfig, total_samples: int) -> Any:
    pop = list(population)
    pts = [p.mu() for p in pop]
    hvc_scores = []
    for i in range(len(pop)):
        hvc = _compute_hvc(pts, cfg.ref_point, i)
        hvc_scores.append(hvc)

    ucb_scores = []
    for i, p in enumerate(pop):
        n = p.n
        if n == 0:
            return p
        exploit = hvc_scores[i]
        explore = cfg.beta * np.sqrt(np.log(max(2, total_samples)) / n)
        ucb_scores.append(exploit + explore)

    best_idx = int(np.argmax(ucb_scores))
    return pop[best_idx]