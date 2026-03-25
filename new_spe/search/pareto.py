from typing import List, Sequence
import numpy as np

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """如果 a 支配 b，则返回 True（多目标，数值越大越好）"""
    return bool(np.all(a >= b)) and bool(np.any(a > b))

def pareto_front(points: Sequence[np.ndarray]) -> List[int]:
    """返回位于第一帕累托前沿的点的索引列表"""
    if not points:
        return []
    pts = [np.asarray(p, dtype=float) for p in points]
    n = len(pts)
    front: List[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i != j and _dominates(pts[j], pts[i]):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front