from __future__ import annotations

import io
import json
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests

from .kernel import DeepSeekKernel


@dataclass(frozen=True)
class BBHExample:
    input: str
    target: str


@dataclass(frozen=True)
class BBHTask:
    name: str
    description: str
    examples: List[BBHExample]


def _normalize_text(s: str) -> str:
    s = s.strip()
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s


_CHOICE_RE = re.compile(r"\b([A-Z])\b")


def _normalize_answer(s: str) -> str:
    s = _normalize_text(s)
    s = s.strip().strip(".").strip()
    s_low = s.lower()
    if s_low in {"yes", "no"}:
        return s_low
    m = _CHOICE_RE.search(s.upper())
    if m:
        return m.group(1)
    return s_low


def answers_equivalent(pred: str, gold: str) -> bool:
    pred_n = _normalize_answer(pred)
    gold = _normalize_text(gold)
    if "|||" in gold:
        golds = [g.strip() for g in gold.split("|||") if g.strip()]
    else:
        golds = [gold]
    for g in golds:
        if pred_n == _normalize_answer(g):
            return True
    return False


def ensure_bbh_downloaded(cache_dir: str, *, repo_zip_url: str = "https://github.com/suzgunmirac/BIG-Bench-Hard/archive/refs/heads/main.zip") -> Path:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    marker = cache / ".bbh_ready"
    tasks_dir = cache / "bbh"
    if marker.exists() and tasks_dir.exists():
        return tasks_dir

    r = requests.get(repo_zip_url, timeout=120)
    r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))

    extracted = 0
    for info in zf.infolist():
        name = info.filename.replace("\\", "/")
        if not name.endswith(".json"):
            continue
        if "/bbh/" not in name:
            continue
        rel = name.split("/bbh/", 1)[1]
        out_path = tasks_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(out_path, "wb") as dst:
            dst.write(src.read())
        extracted += 1

    if extracted == 0:
        raise RuntimeError("未在下载的 ZIP 中找到 BBH 任务文件")

    marker.write_text("ok", encoding="utf-8")
    return tasks_dir


def _load_task_json(path: Path) -> BBHTask:
    obj = json.loads(path.read_text(encoding="utf-8"))
    name = str(obj.get("name") or path.stem)
    desc = str(obj.get("description") or "")
    examples_raw = obj.get("examples") or []
    examples: List[BBHExample] = []
    for ex in examples_raw:
        if not isinstance(ex, dict):
            continue
        inp = str(ex.get("input", ""))
        tgt = str(ex.get("target", ""))
        if inp and tgt:
            examples.append(BBHExample(input=inp, target=tgt))
    if not examples:
        raise RuntimeError(f"Task has no examples: {path}")
    return BBHTask(name=name, description=desc, examples=examples)


class BBHEvaluator:
    def __init__(
        self,
        tasks_dir: str,
        *,
        tasks: Optional[Sequence[str]] = None,
        seed: int = 0,
        n_shot: int = 3,
        include_description: bool = True,
    ):
        self.tasks_dir = Path(tasks_dir)
        self.rng = np.random.default_rng(seed)
        self.n_shot = int(max(0, n_shot))
        self.include_description = bool(include_description)
        self._tasks = self._load_tasks(tasks)

    @classmethod
    def from_cache(
        cls,
        *,
        cache_dir: str,
        tasks: Optional[Sequence[str]] = None,
        seed: int = 0,
        n_shot: int = 3,
        include_description: bool = True,
    ) -> "BBHEvaluator":
        tasks_dir = ensure_bbh_downloaded(cache_dir)
        return cls(str(tasks_dir), tasks=tasks, seed=seed, n_shot=n_shot, include_description=include_description)

    def _load_tasks(self, tasks: Optional[Sequence[str]]) -> List[BBHTask]:
        wanted = None
        if tasks:
            wanted = {t.strip() for t in tasks if t.strip()}
        items: List[BBHTask] = []
        for p in sorted(self.tasks_dir.glob("*.json")):
            task = _load_task_json(p)
            if wanted is not None and task.name not in wanted and p.stem not in wanted:
                continue
            items.append(task)
        if not items:
            available = [p.stem for p in sorted(self.tasks_dir.glob("*.json"))][:20]
            raise RuntimeError(f"未加载到任何 BBH 任务。示例可用任务：{available}")
        return items

    def sample(self) -> Tuple[BBHTask, int]:
        t_idx = int(self.rng.integers(0, len(self._tasks)))
        task = self._tasks[t_idx]
        e_idx = int(self.rng.integers(0, len(task.examples)))
        return task, e_idx

    def _format_prompt(self, task: BBHTask, query_idx: int) -> Tuple[str, str]:
        exs = task.examples
        query = exs[query_idx]

        idxs = [i for i in range(len(exs)) if i != query_idx]
        self.rng.shuffle(idxs)
        shot_idxs = idxs[: min(self.n_shot, len(idxs))]

        parts: List[str] = []
        parts.append(f"Task: {task.name}")
        if self.include_description and task.description:
            parts.append(task.description.strip())
        for si in shot_idxs:
            parts.append(f"Input: {exs[si].input}\nTarget: {exs[si].target}")
        user_msg = "\n\n".join(parts + [f"Input: {query.input}\nTarget:"])
        return user_msg, query.target

    def evaluate_once(
        self,
        *,
        kernel: DeepSeekKernel,
        prompt: str,
        embedder=None,
    ) -> Dict[str, object]:
        task, q_idx = self.sample()
        user_msg, gold = self._format_prompt(task, q_idx)
        result = kernel.chat(prompt, user_msg, expect_json=False, stream=True, temperature=kernel.config.temperature)
        response_text = result.content or ""
        pred = response_text.strip().splitlines()[-1].strip() if response_text.strip() else ""
        correct = answers_equivalent(pred, gold)
        compactness = float(1.0 / (1.0 + np.log1p(max(0, len(response_text))))) if response_text else 0.0
        y = np.asarray([1.0 if correct else 0.0, compactness], dtype=float)

        out: Dict[str, object] = {
            "y": y,
            "task": task.name,
            "example_idx": q_idx,
            "gold": gold,
            "pred": pred,
            "response_len": len(response_text),
        }
        if embedder is not None:
            try:
                out["response_emb"] = embedder.embed(response_text)
            except Exception:
                pass
        return out

