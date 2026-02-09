from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Sequence

import numpy as np

from .genome import StructuredGenome
from .kernel import DeepSeekKernel


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


def _project_invariants(child_loci: Dict[str, str], parent: StructuredGenome, invariant_loci: Sequence[str]) -> Dict[str, str]:
    projected = dict(child_loci)
    for k in invariant_loci:
        projected[k] = parent.loci.get(k, "")
    return projected


@dataclass(frozen=True)
class IntraLocusRewrite:
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
            f"Current instruct:\n{parent.loci.get('L_instruct','')}\n\n"
            f"Fixed constraints:\n{parent.loci.get('L_const','')}\n\n"
            "Return a new instruction that is compatible with the constraints."
        )
        result = kernel.chat(system_msg, user_msg, expect_json=True, stream=False, temperature=kernel.config.temperature)
        new_instruct = parent.loci.get("L_instruct", "")
        try:
            parsed = json.loads(result.content)
            if isinstance(parsed, dict) and isinstance(parsed.get("new_instruct"), str) and parsed["new_instruct"].strip():
                new_instruct = parsed["new_instruct"].strip()
        except Exception:
            pass

        child = dict(parent.loci)
        child["L_instruct"] = new_instruct
        return _project_invariants(child, parent, invariant_loci)


@dataclass(frozen=True)
class IntraLocusRefine:
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
            f"Current instruct:\n{parent.loci.get('L_instruct','')}\n\n"
            f"Fixed constraints:\n{parent.loci.get('L_const','')}\n\n"
            "Perform minimal edits. Keep style consistent."
        )
        result = kernel.chat(system_msg, user_msg, expect_json=True, stream=False, temperature=max(0.1, kernel.config.temperature * 0.5))
        new_instruct = parent.loci.get("L_instruct", "")
        try:
            parsed = json.loads(result.content)
            if isinstance(parsed, dict) and isinstance(parsed.get("new_instruct"), str) and parsed["new_instruct"].strip():
                new_instruct = parsed["new_instruct"].strip()
        except Exception:
            pass

        child = dict(parent.loci)
        child["L_instruct"] = new_instruct
        return _project_invariants(child, parent, invariant_loci)


@dataclass(frozen=True)
class LocusCrossover:
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
        loci_keys = [k for k in a.loci.keys() if k in b.loci.keys() and k not in set(invariant_loci)]
        if not loci_keys:
            return dict(a.loci)
        chosen = loci_keys[int(rng.integers(0, len(loci_keys)))]
        child = dict(a.loci)
        child[chosen] = b.loci.get(chosen, child.get(chosen, ""))
        return _project_invariants(child, a, invariant_loci)


@dataclass(frozen=True)
class SemanticInterpolation:
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
            f"Instruction A:\n{a.loci.get('L_instruct','')}\n\n"
            f"Instruction B:\n{b.loci.get('L_instruct','')}\n\n"
            f"Fixed constraints:\n{a.loci.get('L_const','')}\n\n"
            "Return a single fused instruction."
        )
        result = kernel.chat(system_msg, user_msg, expect_json=True, stream=False, temperature=kernel.config.temperature)
        new_instruct = a.loci.get("L_instruct", "")
        try:
            parsed = json.loads(result.content)
            if isinstance(parsed, dict) and isinstance(parsed.get("new_instruct"), str) and parsed["new_instruct"].strip():
                new_instruct = parsed["new_instruct"].strip()
        except Exception:
            pass

        child = dict(a.loci)
        child["L_instruct"] = new_instruct
        return _project_invariants(child, a, invariant_loci)

