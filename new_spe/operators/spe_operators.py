from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Sequence

import numpy as np

# 导入核心组件
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
    """核投影不变量机制：强制保护不能被修改的基因座"""
    projected = dict(child_loci)
    for k in invariant_loci:
        projected[k] = parent.loci.get(k, "")
    return projected


@dataclass(frozen=True)
class IntraLocusRewrite:
    name: str = "K_rew"
    arity: int = 1

    def apply(self, parents: Sequence[StructuredGenome], *, kernel: DeepSeekKernel, rng: np.random.Generator,
              invariant_loci: Sequence[str]) -> Dict[str, str]:
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
    name: str = "K_ref"
    arity: int = 1

    def apply(self, parents: Sequence[StructuredGenome], *, kernel: DeepSeekKernel, rng: np.random.Generator,
              invariant_loci: Sequence[str]) -> Dict[str, str]:
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
    name: str = "K_swp"
    arity: int = 2

    def apply(self, parents: Sequence[StructuredGenome], *, kernel: DeepSeekKernel, rng: np.random.Generator,
              invariant_loci: Sequence[str]) -> Dict[str, str]:
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

    def apply(self, parents: Sequence[StructuredGenome], *, kernel: DeepSeekKernel, rng: np.random.Generator,
              invariant_loci: Sequence[str]) -> Dict[str, str]:
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


@dataclass(frozen=True)
class ErrorDrivenRefine:
    """
    【全新升级版算子】：全局智能诊断与动态路由 (Intelligent Diagnostic Router)
    Teacher 审视错题后，自主决定需要修改 Role, Instruct, Constraint 还是 Style，并输出更新。
    """
    name: str = "K_err_diag"
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
        new_loci = dict(parent.loci)

        # 1. 检查是否有错题
        failures = getattr(parent, 'failure_cases', [])
        if not failures:
            return new_loci

        # 2. 动态计算当前可修改的基因座（排除被锁死的 invariant_loci）
        all_possible_loci = ["L_role", "L_instruct", "L_const", "L_style"]
        available_loci = [k for k in all_possible_loci if k not in invariant_loci]

        if not available_loci:
            # 如果全被锁死了，大模型也没辙
            return new_loci

        # 3. 抽样一道错题
        sample_fail = random.choice(failures)
        failed_q = sample_fail.get('question', sample_fail.get('task', 'Unknown Question'))
        wrong_a = sample_fail.get('wrong_ans', sample_fail.get('wrong_output', 'Unknown Output'))
        correct_a = sample_fail.get('correct_ans', sample_fail.get('ground_truth', 'Unknown Truth'))

        # 4. 构建强制输出 JSON 格式的 Prompt
        system_msg = (
            "You are an expert Prompt Engineer specializing in General Artificial Intelligence.\n"
            "Your task is to diagnose why a student model failed a task and dynamically decide WHICH prompt module needs to be updated.\n"
            "You MUST output ONLY a valid JSON object."
        )

        user_msg = f"""The student AI uses the following structured prompt to solve a wide mix of reasoning tasks.
[Current Prompt Modules]:
- L_role: {parent.loci.get('L_role', '')}
- L_instruct: {parent.loci.get('L_instruct', '')}
- L_const: {parent.loci.get('L_const', '')}
- L_style: {parent.loci.get('L_style', '')}

It failed on this specific sample:
[Question]: {failed_q}
[Student's Wrong Output]: {wrong_a}
[Ground Truth]: {correct_a}

TASK:
1. Diagnose the root cause of the failure.
2. Decide WHICH SINGLE MODULE is best suited to fix this issue. 
   - Choose `L_const` to add universal logic guardrails or strict formatting rules.
   - Choose `L_instruct` if the core task explanation is flawed.
   - Choose `L_role` to shift the persona.
   - Choose `L_style` for purely output tone adjustments.
   (You MUST choose from these available modules: {available_loci})
3. Provide the COMPLETE updated text for the chosen module. If choosing L_const, you should generally append your new rule to the existing constraints. Keep rules generalized (don't overfit to this specific math/logic question).

OUTPUT EXACTLY THIS JSON FORMAT:
{{
  "diagnosis": "Brief explanation of the failure...",
  "target_module": "L_const", 
  "updated_text": "The complete new text for this module..."
}}"""

        try:
            # 【关键修改】：启用 expect_json=True 确保大模型返回规范数据
            result = kernel.chat(system_msg, user_msg, expect_json=True, stream=False,
                                 temperature=kernel.config.temperature)

            # 5. 解析 JSON 数据
            parsed = json.loads(result.content)
            diagnosis = parsed.get("diagnosis", "No diagnosis provided.")
            target_module = parsed.get("target_module", "")
            updated_text = parsed.get("updated_text", "")

            # 6. 安全校验与应用路由
            if target_module in available_loci and updated_text:
                # 优雅的终端打印展示大模型的思考过程
                print(f"      🧠 诊断反思: {diagnosis}")
                print(f"      🎯 路由决策: 决定修改 [{target_module}]")
                print(f"      👉 更新内容: \"{updated_text}\"")

                # 动态替换目标基因座的内容
                new_loci[target_module] = updated_text
            else:
                print(f"      ⚠️ [ErrorDrivenRefine] 大模型返回了非法的模块名称: {target_module}")

        except Exception as e:
            print(f"      [ErrorDrivenRefine] API 调用失败或 JSON 解析错误: {e}")

        # 依然经过不变量投影保护（双重保险）
        return _project_invariants(new_loci, parent, invariant_loci)