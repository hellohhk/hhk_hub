from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class HashingNgramEmbedder:
    dim: int = 1024
    ngram_range: Tuple[int, int] = (3, 5)
    lowercase: bool = True
    l2_normalize: bool = True

    def embed(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(self.dim, dtype=np.float32)

        if self.lowercase:
            text = text.lower()

        v = np.zeros(self.dim, dtype=np.float32)
        min_n, max_n = self.ngram_range
        text_len = len(text)

        for n in range(min_n, max_n + 1):
            if text_len < n:
                continue
            for i in range(0, text_len - n + 1):
                s = text[i : i + n].encode("utf-8", errors="ignore")
                h = hashlib.blake2b(s, digest_size=8).digest()
                hv = int.from_bytes(h, byteorder="little", signed=False)
                idx = hv % self.dim
                sign = 1.0 if (hv & 1) == 0 else -1.0
                v[idx] += sign

        if self.l2_normalize:
            norm = float(np.linalg.norm(v))
            if norm > 0:
                v = v / norm
        return v


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    cos = float(np.dot(a, b) / (na * nb))
    return float(1.0 - max(min(cos, 1.0), -1.0))

