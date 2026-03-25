from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Sequence

import numpy as np

# 导入我们重构后的核心组件
from new_spe.core.genome import StructuredGenome
from new_spe.models.deepseek_kernel import DeepSeekKernel


class KernelOperator(Protocol):
    name: str
    arity: int

    def apply(
            self,
            parents: Sequence[StructuredGenome],
            *,
            kernel: DeepSeekKernel,
            rng: np.random.Generator,
            invariant_loci: Sequence[str],
    ) -> Dict[str, str]:
        ...


def _project_invariants(child_loci: Dict[str, str], parent: StructuredGenome, invariant_loci: Sequence[str]) -> Dict[
    str, str]:
    """
    【核投影不变量机制】
    无论子代怎么变异，强制将 invariant_loci (通常是 Constraint 约束) 投影回父代的原貌。
    这就是 SPE 保证格式不出错、实现“带镣铐跳舞”的物理屏障。
    """
    projected = dict(child_loci)
    for k in invariant_loci:
        projected[k] = parent.loci.get(k, "")
    return projected


@dataclass(frozen=True)
class IntraLocusRewrite:
    """算子：大步重写 (K_rew)"""
    name: str = "K_rew"
    arity: int = 1

    def apply(
            self,
            parents: Sequence[StructuredGenome],
            *,
            kernel: DeepSeekKernel,
            rng: np.random.Generator,
            invariant_loci: Sequence[str],
    ) -> Dict[str, str]:
        parent = parents[0]
        system_msg = (
            "You are a meta-prompt optimizer.\n"
            "Mutate ONLY the instruction text to improve reasoning quality and robustness under noise.\n"
            "You MUST return JSON: {\"new_instruct\": \"...\"}\n"
            "Do not include any other keys.\n"
        )
        user_msg = (
            f"Current instruct:\n{parent.loci.get('L_instruct', '')}\n\n"
            f"Fixed constraints:\n{parent.loci.get('L_const', '')}\n\n"
            "Return a new instruction that is compatible with the constraints."
        )
        result = kernel.chat(system_msg, user_msg, expect_json=True, stream=False,
                             temperature=kernel.config.temperature)
        new_instruct = parent.loci.get("L_instruct", "")
        try:
            parsed = json.loads(result.content)
            if isinstance(parsed, dict) and isinstance(parsed.get("new_instruct"), str) and parsed[
                "new_instruct"].strip():
                new_instruct = parsed["new_instruct"].strip()
        except Exception:
            pass

        child = dict(parent.loci)
        child["L_instruct"] = new_instruct
        return _project_invariants(child, parent, invariant_loci)


@dataclass(frozen=True)
class IntraLocusRefine:
    """算子：小步润色 (K_ref)"""
    name: str = "K_ref"
    arity: int = 1

    def apply(
            self,
            parents: Sequence[StructuredGenome],
            *,
            kernel: DeepSeekKernel,
            rng: np.random.Generator,
            invariant_loci: Sequence[str],
    ) -> Dict[str, str]:
        parent = parents[0]
        system_msg = (
            "You are a meta-prompt optimizer.\n"
            "Refine ONLY the instruction text with minimal edits to increase clarity and reduce failure cases.\n"
            "You MUST return JSON: {\"new_instruct\": \"...\"}\n"
            "Do not include any other keys.\n"
        )
        user_msg = (
            f"Current instruct:\n{parent.loci.get('L_instruct', '')}\n\n"
            f"Fixed constraints:\n{parent.loci.get('L_const', '')}\n\n"
            "Perform minimal edits. Keep style consistent."
        )
        # 润色算子使用更低的 temperature (更趋于保守和稳定)
        result = kernel.chat(system_msg, user_msg, expect_json=True, stream=False,
                             temperature=max(0.1, kernel.config.temperature * 0.5))
        new_instruct = parent.loci.get("L_instruct", "")
        try:
            parsed = json.loads(result.content)
            if isinstance(parsed, dict) and isinstance(parsed.get("new_instruct"), str) and parsed[
                "new_instruct"].strip():
                new_instruct = parsed["new_instruct"].strip()
        except Exception:
            pass

        child = dict(parent.loci)
        child["L_instruct"] = new_instruct
        return _project_invariants(child, parent, invariant_loci)


@dataclass(frozen=True)
class LocusCrossover:
    """算子：基因座交叉/交换 (K_swp)"""
    name: str = "K_swp"
    arity: int = 2

    def apply(
            self,
            parents: Sequence[StructuredGenome],
            *,
            kernel: DeepSeekKernel,
            rng: np.random.Generator,
            invariant_loci: Sequence[str],
    ) -> Dict[str, str]:
        a, b = parents[0], parents[1]
        # 找出允许被交叉的基因座 (排除被保护的不变量 invariant_loci)
        loci_keys = [k for k in a.loci.keys() if k in b.loci.keys() and k not in set(invariant_loci)]
        if not loci_keys:
            return dict(a.loci)

        # 随机挑选一个非受保护的基因座，用父代 B 的替换父代 A 的
        chosen = loci_keys[int(rng.integers(0, len(loci_keys)))]
        child = dict(a.loci)
        child[chosen] = b.loci.get(chosen, child.get(chosen, ""))
        return _project_invariants(child, a, invariant_loci)


@dataclass(frozen=True)
class SemanticInterpolation:
    """算子：语义融合/插值 (K_mix)"""
    name: str = "K_mix"
    arity: int = 2

    def apply(
            self,
            parents: Sequence[StructuredGenome],
            *,
            kernel: DeepSeekKernel,
            rng: np.random.Generator,
            invariant_loci: Sequence[str],
    ) -> Dict[str, str]:
        a, b = parents[0], parents[1]
        system_msg = (
            "You are a meta-prompt optimizer.\n"
            "Fuse two instructions into one that inherits strengths of both while staying concise.\n"
            "You MUST return JSON: {\"new_instruct\": \"...\"}\n"
            "Do not include any other keys.\n"
        )
        user_msg = (
            f"Instruction A:\n{a.loci.get('L_instruct', '')}\n\n"
            f"Instruction B:\n{b.loci.get('L_instruct', '')}\n\n"
            f"Fixed constraints:\n{a.loci.get('L_const', '')}\n\n"
            "Return a single fused instruction."
        )
        result = kernel.chat(system_msg, user_msg, expect_json=True, stream=False,
                             temperature=kernel.config.temperature)
        new_instruct = a.loci.get("L_instruct", "")
        try:
            parsed = json.loads(result.content)
            if isinstance(parsed, dict) and isinstance(parsed.get("new_instruct"), str) and parsed[
                "new_instruct"].strip():
                new_instruct = parsed["new_instruct"].strip()
        except Exception:
            pass

        child = dict(a.loci)
        child["L_instruct"] = new_instruct
        return _project_invariants(child, a, invariant_loci)