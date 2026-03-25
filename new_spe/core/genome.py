from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

LocusKey = str


@dataclass
class StructuredGenome:
    """
    【结构化基因组】
    用于 SPE 多目标帕累托进化的核心数据结构。
    包含提示词的模块化表示，以及用于追踪搜索半径和评估方差的在线统计量。
    """
    loci: Dict[LocusKey, str]
    uid: str
    parents: Tuple[str, ...] = field(default_factory=tuple)
    operator: str = "init"
    # 记录该个体与其父代之间的各类距离（例如 embedding L2 displacement）
    radius: Dict[str, float] = field(default_factory=dict)

    # --- 目标得分 (y) 的在线均值/方差追踪 (Welford 算法) ---
    n: int = 0
    mean: Optional[np.ndarray] = None
    m2: Optional[np.ndarray] = None
    history: List[np.ndarray] = field(default_factory=list)

    # --- 输出 Embedding 的在线均值/方差追踪 ---
    output_emb_n: int = 0
    output_emb_mean: Optional[np.ndarray] = None
    output_emb_m2: Optional[np.ndarray] = None

    # --- 输出长度的在线均值/方差追踪 ---
    output_len_n: int = 0
    output_len_mean: float = 0.0
    output_len_m2: float = 0.0

    def clone_with(self, *, loci: Dict[LocusKey, str], uid: str, parents: Sequence[str],
                   operator: str) -> "StructuredGenome":
        """创建子代副本"""
        return StructuredGenome(loci=dict(loci), uid=uid, parents=tuple(parents), operator=operator)

    def prompt_text(self) -> str:
        """渲染为最终喂给 LLM 的字符串"""
        role = self.loci.get("L_role", "")
        instruct = self.loci.get("L_instruct", "")
        const = self.loci.get("L_const", "")
        style = self.loci.get("L_style", "")
        return f"Role: {role}\nTask: {instruct}\nConstraint: {const}\nStyle: {style}"

    def update(self, y: np.ndarray, keep_history: bool = True) -> None:
        """更新目标得分 (y) 的统计量"""
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
        """更新输出 Embedding 的统计量"""
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
        """计算输出 Embedding 的方差迹 (Trace Variance)"""
        if self.output_emb_mean is None or self.output_emb_m2 is None or self.output_emb_n < 2:
            return float("inf")
        var = self.output_emb_m2 / (self.output_emb_n - 1)
        return float(np.sum(var))

    def update_output_length(self, length: int) -> None:
        """更新输出长度的统计量"""
        x = float(max(0, int(length)))
        self.output_len_n += 1
        delta = x - self.output_len_mean
        self.output_len_mean = self.output_len_mean + delta / self.output_len_n
        delta2 = x - self.output_len_mean
        self.output_len_m2 = self.output_len_m2 + delta * delta2

    def output_len_var(self) -> float:
        """计算输出长度的方差"""
        if self.output_len_n < 2:
            return float("inf")
        return float(self.output_len_m2 / (self.output_len_n - 1))

    def mu(self) -> np.ndarray:
        """获取当前个体的平均目标得分向量"""
        if self.mean is None:
            # 默认返回 2D 向量 (对应双目标优化，如 [Accuracy, -Length])
            return np.zeros(2, dtype=float)
        return self.mean

    def var(self) -> np.ndarray:
        """获取目标得分的方差"""
        if self.mean is None or self.m2 is None or self.n < 2:
            return np.full_like(self.mu(), np.inf, dtype=float)
        return self.m2 / (self.n - 1)