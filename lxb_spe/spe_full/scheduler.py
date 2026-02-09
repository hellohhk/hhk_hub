from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from .genome import StructuredGenome
from .hypervolume import hypervolume_contribution_2d


@dataclass(frozen=True)
class HVCUCBScheduleConfig:
    beta: float = 0.4
    ref_point: np.ndarray = field(default_factory=lambda: np.asarray([0.0, 0.0], dtype=float))


def pick_hvc_ucb(
    candidates: Sequence[StructuredGenome],
    *,
    cfg: HVCUCBScheduleConfig,
    total_samples: int,
) -> StructuredGenome:
    ref = np.asarray(cfg.ref_point, dtype=float)
    mus = [c.mu() for c in candidates]
    nonempty_mus = [m for c, m in zip(candidates, mus) if c.n > 0]

    best_idx = 0
    best_score = -1e18
    for i, c in enumerate(candidates):
        mu_i = mus[i]
        reward = hypervolume_contribution_2d(mu_i, nonempty_mus, ref) if c.n > 0 else 0.0
        confidence = cfg.beta * float(np.sqrt(2.0 * np.log(max(1, total_samples + 1)) / (c.n + 1e-9)))
        score = reward + confidence
        if score > best_score:
            best_score = score
            best_idx = i
    return candidates[best_idx]
