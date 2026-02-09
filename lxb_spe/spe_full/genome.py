from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


LocusKey = str


@dataclass
class StructuredGenome:
    loci: Dict[LocusKey, str]
    uid: str
    parents: Tuple[str, ...] = field(default_factory=tuple)
    operator: str = "init"
    radius: Dict[str, float] = field(default_factory=dict)
    n: int = 0
    mean: Optional[np.ndarray] = None
    m2: Optional[np.ndarray] = None
    history: List[np.ndarray] = field(default_factory=list)

    output_emb_n: int = 0
    output_emb_mean: Optional[np.ndarray] = None
    output_emb_m2: Optional[np.ndarray] = None

    output_len_n: int = 0
    output_len_mean: float = 0.0
    output_len_m2: float = 0.0

    def clone_with(self, *, loci: Dict[LocusKey, str], uid: str, parents: Sequence[str], operator: str) -> "StructuredGenome":
        return StructuredGenome(loci=dict(loci), uid=uid, parents=tuple(parents), operator=operator)

    def prompt_text(self) -> str:
        role = self.loci.get("L_role", "")
        instruct = self.loci.get("L_instruct", "")
        const = self.loci.get("L_const", "")
        style = self.loci.get("L_style", "")
        return f"Role: {role}\nTask: {instruct}\nConstraint: {const}\nStyle: {style}"

    def update(self, y: np.ndarray, keep_history: bool = True) -> None:
        y = np.asarray(y, dtype=float)
        if self.mean is None:
            self.mean = np.zeros_like(y)
            self.m2 = np.zeros_like(y)

        self.n += 1
        delta = y - self.mean
        self.mean = self.mean + delta / self.n
        delta2 = y - self.mean
        self.m2 = self.m2 + delta * delta2
        if keep_history:
            self.history.append(y)

    def update_output_embedding(self, emb: np.ndarray) -> None:
        emb = np.asarray(emb, dtype=float)
        if self.output_emb_mean is None:
            self.output_emb_mean = np.zeros_like(emb)
            self.output_emb_m2 = np.zeros_like(emb)

        self.output_emb_n += 1
        delta = emb - self.output_emb_mean
        self.output_emb_mean = self.output_emb_mean + delta / self.output_emb_n
        delta2 = emb - self.output_emb_mean
        self.output_emb_m2 = self.output_emb_m2 + delta * delta2

    def output_emb_trace_var(self) -> float:
        if self.output_emb_mean is None or self.output_emb_m2 is None or self.output_emb_n < 2:
            return float("inf")
        var = self.output_emb_m2 / (self.output_emb_n - 1)
        return float(np.sum(var))

    def update_output_length(self, length: int) -> None:
        x = float(max(0, int(length)))
        self.output_len_n += 1
        delta = x - self.output_len_mean
        self.output_len_mean = self.output_len_mean + delta / self.output_len_n
        delta2 = x - self.output_len_mean
        self.output_len_m2 = self.output_len_m2 + delta * delta2

    def output_len_var(self) -> float:
        if self.output_len_n < 2:
            return float("inf")
        return float(self.output_len_m2 / (self.output_len_n - 1))

    def mu(self) -> np.ndarray:
        if self.mean is None:
            return np.zeros(2, dtype=float)
        return self.mean

    def var(self) -> np.ndarray:
        if self.mean is None or self.m2 is None or self.n < 2:
            return np.full_like(self.mu(), np.inf, dtype=float)
        return self.m2 / (self.n - 1)
