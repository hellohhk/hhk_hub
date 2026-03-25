from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests

from new_spe.models.deepseek_kernel import DeepSeekKernel


@dataclass(frozen=True)
class BBHExample:
    input: str
    target: str
    task_source: str = "mixed"


@dataclass
class BBHTaskStub:
    """用于兼容合并数据集的任务存根"""
    name: str
    description: str
    examples: List[BBHExample]


# --- 文本处理工具函数 ---
def _normalize_text(s: str) -> str:
    s = s.strip().replace("\u00a0", " ")
    return re.sub(r"\s+", " ", s)


def _normalize_answer(s: str) -> str:
    s = _normalize_text(s).strip(".").strip().lower()
    if s in {"yes", "no"}: return s
    m = re.search(r"\b([A-Z])\b", s.upper())
    return m.group(1) if m else s


def answers_equivalent(pred: str, gold: str) -> bool:
    pred_n = _normalize_answer(pred)
    golds = [g.strip() for g in gold.split("|||") if g.strip()] if "|||" in gold else [gold]
    return any(pred_n == _normalize_answer(g) for g in golds)


# --- 核心评测类 ---
class BBHEvaluator:
    def __init__(
            self,
            path: str,
            *,
            tasks: Optional[Sequence[str]] = None,
            seed: int = 42,
            n_shot: int = 3,
            include_description: bool = True,
    ):
        self.path = Path(path)
        self.rng = np.random.default_rng(seed)
        self.n_shot = int(max(0, n_shot))
        self.include_description = bool(include_description)
        self.flat_dataset: List[Tuple[BBHTaskStub, int]] = []

        # 情况 A: 路径是一个合并后的单个 JSON 文件
        if self.path.is_file() and self.path.suffix == ".json":
            self._load_from_merged_file()
        # 情况 B: 路径是一个包含多个任务文件的文件夹
        else:
            self._load_from_dir(tasks)

    def _load_from_merged_file(self):
        """加载合并后的全任务大文件"""
        print(f"📦 正在从合并文件加载数据: {self.path.name}")
        with open(self.path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # 将原始字典列表转换为 BBHExample 对象列表
        examples = [
            BBHExample(
                input=ex['input'],
                target=ex['target'],
                task_source=ex.get('task_source', 'mixed')
            ) for ex in raw_data
        ]

        # 创建一个全局任务存根，让 few-shot 采样能从全集里抽题
        global_stub = BBHTaskStub(name="Mixed-BBH", description="Mixed tasks from BIG-Bench Hard", examples=examples)

        for i in range(len(examples)):
            self.flat_dataset.append((global_stub, i))

    def _load_from_dir(self, tasks_filter: Optional[Sequence[str]]):
        """原有的文件夹加载逻辑"""
        wanted = {t.strip() for t in tasks_filter} if tasks_filter else None
        for p in sorted(self.path.glob("*.json")):
            if wanted and p.stem not in wanted:
                continue

            obj = json.loads(p.read_text(encoding="utf-8"))
            examples = [BBHExample(input=ex['input'], target=ex['target'], task_source=p.stem)
                        for ex in obj.get("examples", [])]

            task_obj = BBHTaskStub(
                name=obj.get("name", p.stem),
                description=obj.get("description", ""),
                examples=examples
            )
            for i in range(len(examples)):
                self.flat_dataset.append((task_obj, i))

    def _format_prompt(self, task: BBHTaskStub, query_idx: int) -> Tuple[str, str]:
        """格式化 Few-shot Prompt"""
        exs = task.examples
        query = exs[query_idx]

        # 从当前任务(或合并池)中抽取 n 个示例，排除掉当前题目本身
        idxs = [i for i in range(len(exs)) if i != query_idx]
        self.rng.shuffle(idxs)
        shot_idxs = idxs[: min(self.n_shot, len(idxs))]

        parts = [f"Task: {task.name}"]
        if self.include_description and task.description:
            parts.append(task.description.strip())

        for si in shot_idxs:
            parts.append(f"Input: {exs[si].input}\nTarget: {exs[si].target}")

        user_msg = "\n\n".join(parts + [f"Input: {query.input}\nTarget:"])
        return user_msg, query.target

    def _do_evaluate(self, kernel: DeepSeekKernel, prompt: str, task: BBHTaskStub, q_idx: int, embedder=None) -> Dict[
        str, object]:
        user_msg, gold = self._format_prompt(task, q_idx)
        result = kernel.chat(prompt, user_msg, expect_json=False, stream=True)
        response_text = result.content or ""

        # 提取答案行 (通常是最后一行)
        pred = response_text.strip().splitlines()[-1].strip() if response_text.strip() else ""
        correct = answers_equivalent(pred, gold)

        # 紧凑度得分 (越短越高)
        compactness = float(1.0 / (1.0 + np.log1p(len(response_text)))) if response_text else 0.0

        out = {
            "y": np.asarray([1.0 if correct else 0.0, compactness], dtype=float),
            "gold": gold,
            "pred": pred,
            "task": task.examples[q_idx].task_source,  # 记录这道题原始所属的任务
            "response_len": len(response_text),
        }
        if embedder:
            out["response_emb"] = embedder.embed(response_text)
        return out

    def evaluate_once(self, kernel: DeepSeekKernel, prompt: str, embedder=None) -> Dict[str, object]:
        """进化期间随机抽题评估"""
        idx = int(self.rng.integers(0, len(self.flat_dataset)))
        task_stub, q_idx = self.flat_dataset[idx]
        return self._do_evaluate(kernel, prompt, task_stub, q_idx, embedder)

    def evaluate_by_index(self, kernel: DeepSeekKernel, prompt: str, index: int, embedder=None) -> Dict[str, object]:
        """最终大考按索引全量评估"""
        task_stub, q_idx = self.flat_dataset[index]
        return self._do_evaluate(kernel, prompt, task_stub, q_idx, embedder)