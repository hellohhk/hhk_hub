from typing import Sequence
import numpy as np

def hypervolume_2d(pts: Sequence[np.ndarray], ref_point: np.ndarray) -> float:
    """计算 2D 空间的超体积（越大越好假设）"""
    if not pts:
        return 0.0
    valid = []
    for p in pts:
        if p[0] >= ref_point[0] and p[1] >= ref_point[1]:
            valid.append(p)
    if not valid:
        return 0.0
    # 按第一维度升序排序
    valid.sort(key=lambda x: float(x[0]))
    hv = 0.0
    for i in range(len(valid)):
        w = valid[i][0] - ref_point[0]
        # 对于 2D 来说，如果当前点的第一个维度大，为了形成非支配，它的第二个维度会变小
        # 取有效高度
        h = max(0.0, valid[i][1] - ref_point[1])
        if i == 0:
            hv += w * h
        else:
            w_diff = valid[i][0] - valid[i - 1][0]
            hv += w_diff * h
    return float(hv)