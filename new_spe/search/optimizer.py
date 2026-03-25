from __future__ import annotations
import time
import threading
import concurrent.futures
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

from new_spe.core.genome import StructuredGenome
from new_spe.models.deepseek_kernel import DeepSeekKernel
from new_spe.search.embedding import HashingNgramEmbedder, l2_distance
from new_spe.operators.spe_operators import (
    IntraLocusRefine, IntraLocusRewrite, KernelOperator, LocusCrossover, SemanticInterpolation,
)
from new_spe.search.pareto import pareto_front
from new_spe.search.selection import nsga2_select, tournament_select


@dataclass(frozen=True)
class SPEOptimizerConfig:
    budget: int
    mu: int = 4
    lambd: int = 4
    gens: int = 10
    batch_size: int = 20
    invariant_loci: Tuple[str, ...] = ("L_const",)
    seed: int = 0
    keep_history: bool = False


EvalFn = Callable[[str], Dict[str, object]]
LogFn = Callable[[Dict[str, object]], None]


def _uid(prefix: str, rng: np.random.Generator) -> str:
    return f"{prefix}_{int(rng.integers(10 ** 12, 10 ** 13 - 1))}"


class SPEOptimizer:
    def __init__(
            self, *, kernel: DeepSeekKernel, cfg: SPEOptimizerConfig,
            operators: Optional[Sequence[KernelOperator]] = None,
            operator_probs: Optional[Sequence[float]] = None,
            embedder: Optional[HashingNgramEmbedder] = None,
    ):
        self.kernel = kernel
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.used_budget = 0
        self.total_samples = 0
        self.embedder = embedder or HashingNgramEmbedder()
        self.elite_pool: List[StructuredGenome] = []

        self._lock = threading.Lock()

        ops = list(operators) if operators is not None else [IntraLocusRewrite(), IntraLocusRefine(), LocusCrossover(),
                                                             SemanticInterpolation()]
        self.operators = ops
        self.operator_probs = np.asarray(operator_probs if operator_probs else [0.35, 0.25, 0.2, 0.2], dtype=float)
        self.operator_probs = self.operator_probs / np.sum(self.operator_probs)

    def _update_elite_pool(self, candidates: Sequence[StructuredGenome]):
        combined = self.elite_pool + list(candidates)
        unique_dict = {}
        for g in combined:
            txt = g.prompt_text()
            if txt not in unique_dict or g.n > unique_dict[txt].n:
                unique_dict[txt] = g
        unique_list = list(unique_dict.values())
        if unique_list:
            front_idxs = pareto_front([g.mu() for g in unique_list])
            self.elite_pool = [unique_list[i] for i in front_idxs]

    def _op_cost(self, op: KernelOperator) -> int:
        return 1 if op.name in {"K_rew", "K_ref", "K_mix", "K_err_diag"} else 0

    def _evaluate_once(self, genome: StructuredGenome, eval_fn: EvalFn, log_fn: Optional[LogFn],
                       meta: Dict[str, object]) -> bool:
        if self.used_budget >= self.cfg.budget:
            return False

        out = eval_fn(genome.prompt_text())
        y = out.get("y")
        if y is None:
            return False

        with self._lock:
            if self.used_budget >= self.cfg.budget:
                return False

            if "response_emb" in out and out["response_emb"] is not None:
                try:
                    genome.update_output_embedding(np.asarray(out["response_emb"], dtype=float))
                except Exception:
                    pass
            if "response_len" in out and out["response_len"] is not None:
                try:
                    genome.update_output_length(int(out["response_len"]))
                except Exception:
                    pass

            if "failures" in out and out["failures"]:
                genome.failure_cases.extend(out["failures"])

            genome.update(np.asarray(y, dtype=float), keep_history=self.cfg.keep_history)
            self.used_budget += 1
            self.total_samples += 1

            if log_fn is not None:
                rec = dict(meta)
                rec.update({
                    "uid": genome.uid, "operator": genome.operator, "parents": list(genome.parents),
                    "n": genome.n, "mu": genome.mu().tolist(), "budget_used": self.used_budget,
                })
                log_fn(rec)
        return True

    def _evaluate_n(self, genome: StructuredGenome, eval_fn: EvalFn, log_fn: Optional[LogFn], n: int,
                    meta: Dict[str, object]) -> None:
        remaining = self.cfg.budget - self.used_budget
        actual_n = min(n, remaining)
        if actual_n <= 0: return

        max_workers = min(actual_n, 20)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._evaluate_once, genome, eval_fn, log_fn, meta) for _ in range(actual_n)]
            concurrent.futures.wait(futures)

    def _choose_operator(self) -> KernelOperator:
        idx = int(self.rng.choice(len(self.operators), p=self.operator_probs))
        return self.operators[idx]

    def _make_offspring(self, parents: Sequence[StructuredGenome], op: KernelOperator) -> StructuredGenome:
        base_parent = parents[0]
        new_loci = op.apply(parents, kernel=self.kernel, rng=self.rng, invariant_loci=self.cfg.invariant_loci)
        uid = _uid("g", self.rng)
        child = base_parent.clone_with(loci=new_loci, uid=uid, parents=[p.uid for p in parents], operator=op.name)
        try:
            child_emb = self.embedder.embed(child.prompt_text())
            p1_emb = self.embedder.embed(parents[0].prompt_text())
            child.radius["prompt_disp_l2_p1"] = l2_distance(child_emb, p1_emb)
        except Exception:
            pass
        return child

    def evolve(self, *, init_population: Sequence[StructuredGenome], eval_fn: EvalFn, log_fn: Optional[LogFn] = None) -> \
            List[StructuredGenome]:
        pop: List[StructuredGenome] = list(init_population)

        print("\n🌱 [初始化阶段] 开始评估初始种子 (多线程火力全开)...")
        for i, g in enumerate(pop):
            # ==========================================
            # 【新增】：在这里打印种子 Prompt 的详细结构
            # ==========================================
            print(f"\n   -------------------------------------------------")
            print(f"   🔍 预览 种子 {i + 1} ({g.uid}) 的 Prompt 结构:")
            print(f"   [Role]:       {g.loci.get('L_role', '')}")
            print(f"   [Instruct]:   {g.loci.get('L_instruct', '')}")
            print(f"   [Constraint]: {g.loci.get('L_const', '')}")
            print(f"   [Style]:      {g.loci.get('L_style', '')}")
            print(f"   -------------------------------------------------")

            self._evaluate_n(g, eval_fn, log_fn, self.cfg.batch_size, meta={"phase": "init"})

            # 【重构为清爽输出】：不再打印几百行，只有一行结果
            failures_count = len(getattr(g, 'failure_cases', []))
            print(
                f"   👉 种子 {i + 1} ({g.uid}) 评估完成 | 测验 {g.n} 题 | 平均准确率: {g.mu()[0]:.2%} | 收集到错题: {failures_count} 道")

        self._update_elite_pool(pop)

        print("\n🚀 [进化阶段] 开始跨代繁殖与优胜劣汰...")
        for gen in range(self.cfg.gens):
            if self.used_budget >= self.cfg.budget:
                print(f"\n   ⚠️ 预算已达上限 ({self.used_budget}/{self.cfg.budget})，提前终止进化。")
                break

            # 【重构为清爽输出】：添加明显的代数分割线
            print(f"\n--- [ 第 {gen + 1} 代进化 | 当前消耗预算: {self.used_budget} / {self.cfg.budget} ] ---")

            sel = nsga2_select(pop, mu=len(pop))
            offspring: List[StructuredGenome] = []

            for _ in range(self.cfg.lambd):
                if self.used_budget >= self.cfg.budget: break
                op = self._choose_operator()
                cost = self._op_cost(op)
                if self.used_budget + cost > self.cfg.budget: break
                self.used_budget += cost

                if op.arity == 1:
                    parents = [tournament_select(pop, ranks=sel.ranks, crowding=sel.crowding, rng=self.rng)]
                else:
                    parents = [tournament_select(pop, ranks=sel.ranks, crowding=sel.crowding, rng=self.rng) for _ in
                               range(2)]

                child = self._make_offspring(parents, op)
                self._evaluate_n(child, eval_fn, log_fn, self.cfg.batch_size, meta={"phase": "offspring", "gen": gen})
                offspring.append(child)

                # 【重构为清爽输出】：打印本次触发的算子和繁育结果
                failures_count = len(getattr(child, 'failure_cases', []))
                print(f"   ⚡ [触发算子] {op.name}")
                print(
                    f"      📊 子代评估完成 | 准确率: {child.mu()[0]:.2%} | 当前长度: {len(child.prompt_text())} 字符 | 收集错题: {failures_count} 道")

            pool = pop + offspring + self.elite_pool
            pop = nsga2_select(pool, mu=self.cfg.mu).selected
            self._update_elite_pool(pop)
            best_score = max(g.mu()[0] for g in self.elite_pool)
            print(
                f"\n🏆 第 {gen + 1} 代结束 | 全局精英池最高得分: {best_score:.2%} | 当前预算: {self.used_budget}/{self.cfg.budget}")

            if log_fn is not None:
                log_fn({"phase": "gen_end", "gen": gen, "budget_used": self.used_budget,
                        "population": [g.uid for g in pop], "elite_pool": [g.uid for g in self.elite_pool],
                        "population_mu": [g.mu().tolist() for g in pop]})

        print(f"\n🌟 进化完成，全局精英池共保底收录了 {len(self.elite_pool)} 个帕累托最优 Prompt。")
        return self.elite_pool

    def extract_pareto(self, pop: Sequence[StructuredGenome]) -> List[StructuredGenome]:
        idxs = pareto_front([g.mu() for g in pop])
        return [pop[i] for i in idxs]