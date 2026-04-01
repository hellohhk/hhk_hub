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
        # 🌟 升级版：强制引入高级推理范式与认知重构
        system_msg = (
            "You are an elite AI Prompt Engineer.\n"
            "Your task is to fundamentally REWRITE the instruction text to maximize the LLM's logical reasoning capability.\n"
            "You must introduce advanced prompt paradigms (e.g., explicit step-by-step deduction triggers, cognitive reframing, or edge-case handling guidelines) to make the instruction extremely robust.\n"
            "You MUST return JSON: {\"new_instruct\": \"...\"}\n"
            "Do not include any other keys.\n"
        )
        user_msg = (
            f"Current instruct:\n{parent.loci.get('L_instruct', '')}\n\n"
            f"Fixed constraints:\n{parent.loci.get('L_const', '')}\n\n"
            "Return a completely upgraded instruction that drastically improves logical rigor while remaining compatible with the fixed constraints."
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
        # 🌟 升级版：提升修补精度与防御性，堵住逻辑漏洞
        system_msg = (
            "You are an expert Prompt Auditor.\n"
            "Refine the instruction text with high-precision edits. Your goal is to eliminate ambiguity, close logical loopholes, and add defensive phrasing to prevent hallucination or common failure modes.\n"
            "You MUST return JSON: {\"new_instruct\": \"...\"}\n"
            "Do not include any other keys.\n"
        )
        user_msg = (
            f"Current instruct:\n{parent.loci.get('L_instruct', '')}\n\n"
            f"Fixed constraints:\n{parent.loci.get('L_const', '')}\n\n"
            "Perform surgical edits to optimize clarity and defensiveness. Keep the core intent identical."
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
        # 🌟 升级版：协同策略融合，提取底层认知框架而非简单拼接
        system_msg = (
            "You are a Master Prompt Synthesizer.\n"
            "Do not just concatenate these two instructions. You must extract the underlying reasoning STRATEGIES and cognitive frameworks from both, and fuse them into a single, synergistic, and highly potent instruction.\n"
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
    【升级版算子】：全局智能诊断、动态路由与约束凝练 (Intelligent Diagnostic Router & Condensation)
    Teacher 审视错题后，自主决定需要修改哪个模块。
    当决定修改 L_const 时，会自动将新规则与旧规则进行去重和融合凝练。
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

        # 2. 动态计算当前可修改的基因座
        all_possible_loci = ["L_role", "L_instruct", "L_const", "L_style"]
        available_loci = [k for k in all_possible_loci if k not in invariant_loci]

        if not available_loci:
            return new_loci

        # 3. 抽样一道错题
        sample_fail = random.choice(failures)
        failed_q = sample_fail.get('question', sample_fail.get('task', 'Unknown Question'))
        wrong_a = sample_fail.get('wrong_ans', sample_fail.get('wrong_output', 'Unknown Output'))
        correct_a = sample_fail.get('correct_ans', sample_fail.get('ground_truth', 'Unknown Truth'))

        # 4. 构建强制输出 JSON 格式且带有严格凝练要求的 Prompt
        system_msg = (
            "You are an expert Prompt Engineer specializing in General Artificial Intelligence.\n"
            "Your task is to diagnose why a student model failed a task and dynamically decide WHICH prompt module needs to be updated.\n"
            "You MUST output ONLY a valid JSON object."
        )

        # 🌟 升级版：强化 CoT 与约束生成的任务指令
        user_msg = f"""The student AI uses the following structured prompt to solve a WIDE MIX of reasoning tasks (including Math, Logic, Reading Comprehension, etc.).
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
1. Diagnose the exact cognitive or logical root cause of the failure.
2. Decide WHICH SINGLE MODULE must be updated. (Choose from: {available_loci})
3. Provide the COMPLETE updated text for the chosen module.

CRITICAL DIRECTIVES:
- If choosing `L_instruct`: Rewrite it to explicitly guide the AI away from this specific logical trap.
- If choosing `L_const`: You MUST formulate an explicit Chain-of-Thought (CoT) rule, a strict step-by-step verification protocol, or an absolute mathematical/formatting guardrail. DO NOT just output a vague suggestion.
- For `L_const`, integrate the new rule with existing ones. Remove redundancies. Output a highly CONDENSED list of absolute laws (max 5-7 rules).

OUTPUT EXACTLY THIS JSON FORMAT:
{{
  "diagnosis": "Brief explanation of the failure...",
  "target_module": "L_const", 
  "updated_text": "The complete new text for this module..."
}}"""

        try:
            result = kernel.chat(system_msg, user_msg, expect_json=True, stream=False,
                                 temperature=kernel.config.temperature)

            # 5. 解析 JSON 数据
            parsed = json.loads(result.content)
            diagnosis = parsed.get("diagnosis", "No diagnosis provided.")
            target_module = parsed.get("target_module", "")
            updated_text = parsed.get("updated_text", "")

            # 6. 安全校验与应用路由
            if target_module in available_loci and updated_text:
                print(f"      🧠 诊断反思: {diagnosis}")
                print(f"      🎯 路由决策: 决定修改 [{target_module}]")
                if target_module == "L_const":
                    print(f"      ✨ 执行约束凝练: 规则已去重并压缩！")
                print(f"      👉 更新内容:\n{updated_text}")

                new_loci[target_module] = updated_text
            else:
                pass

        except Exception as e:
            pass

        return _project_invariants(new_loci, parent, invariant_loci)


@dataclass(frozen=True)
class CompactnessPruner:
    """
    【新增算子】：冗余修剪算子 (Compactness Pruner / K_prune)
    专为多目标优化（紧凑度）设计。在不改变核心逻辑和约束的前提下，对提示词进行强力“瘦身去重”。
    """
    name: str = "K_prune"
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

        # 1. 找到可以修剪且内容不为空的基因座
        available_loci = [k for k in ["L_instruct", "L_const", "L_role", "L_style"]
                          if k not in invariant_loci and new_loci.get(k, "").strip()]

        if not available_loci:
            return new_loci

        # 2. 加权随机挑选：倾向于修剪往往最容易发生冗余的 L_instruct 和 L_const
        weights = []
        for k in available_loci:
            if k in ["L_instruct", "L_const"]:
                weights.append(0.4)
            else:
                weights.append(0.1)

        probs = [w / sum(weights) for w in weights]
        target_module = str(rng.choice(available_loci, p=probs))
        original_text = new_loci[target_module]

        # 3. 构建修剪任务 Prompt
        system_msg = (
            "You are an expert Prompt Optimizer specializing in Token Economy and Text Compression.\n"
            "Your task is to aggressively compress and prune the given prompt module WITHOUT losing ANY of its core logical rules, instructions, or intent.\n"
            "You MUST return JSON: {\"compressed_text\": \"...\"}\n"
        )

        user_msg = f"""Here is the text for the [{target_module}] module:
{original_text}

TASK:
1. Remove all conversational filler, redundant phrasing, and repetitive instructions.
2. Condense the remaining rules and logic into the most compact, token-efficient form possible (aiming for at least 20% length reduction).
3. CRITICAL: DO NOT change the core meaning or drop any vital constraints.

Return ONLY the compressed text in this exact JSON format:
{{
  "compressed_text": "The condensed version of the text..."
}}"""

        try:
            # 【细节】：使用极低的温度 (0.1)，保证大模型在做“减法”时极度理智保守，绝不自我发散修改逻辑
            result = kernel.chat(system_msg, user_msg, expect_json=True, stream=False, temperature=0.1)

            parsed = json.loads(result.content)
            compressed_text = parsed.get("compressed_text", "").strip()

            # 4. 只有当大模型返回了有效且【确实变得更短】的文本时，才接受修改
            if compressed_text and len(compressed_text) < len(original_text):
                saved_chars = len(original_text) - len(compressed_text)
                print(f"      ✂️ [修剪瘦身触发] 成功修剪 [{target_module}]! (减重 {saved_chars} 字符)")
                new_loci[target_module] = compressed_text

        except Exception as e:
            pass

        return _project_invariants(new_loci, parent, invariant_loci)