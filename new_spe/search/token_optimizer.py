from __future__ import annotations
import threading
import time
import concurrent.futures
import traceback
from typing import Dict, List, Optional, Sequence
import numpy as np

from new_spe.search.optimizer import SPEOptimizer, SPEOptimizerConfig, EvalFn, LogFn
from new_spe.core.genome import StructuredGenome
from new_spe.search.selection import nsga2_select, tournament_select


class TokenBoundedSPEOptimizer(SPEOptimizer):
    """
    继承自 SPEOptimizer，由 Token 消耗驱动进化，并加入了防卡死超时机制。
    """

    def _evaluate_once(self, genome: StructuredGenome, eval_fn: EvalFn, log_fn: Optional[LogFn],
                       meta: Dict[str, object]) -> bool:
        try:
            if getattr(self.kernel, 'is_budget_exhausted', lambda: False)():
                return False

            out = eval_fn(genome.prompt_text())
            y = out.get("y")
            if y is None:
                return False

            with self._lock:
                if getattr(self.kernel, 'is_budget_exhausted', lambda: False)():
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
                self.total_samples += 1

                if log_fn is not None:
                    rec = dict(meta)
                    rec.update({
                        "uid": genome.uid, "operator": genome.operator, "parents": list(genome.parents),
                        "n": genome.n, "mu": genome.mu().tolist(),
                        "tokens_used": getattr(self.kernel, 'total_tokens_consumed', 0)
                    })
                    log_fn(rec)
            return True
        except Exception as e:
            print(f"\n❌ [线程内报错] 测验异常: {e}")
            return False

    # 🌟 核心修复区：重写多线程评估，加入超时强杀！
    def _evaluate_n(self, genome: StructuredGenome, eval_fn: EvalFn, log_fn: Optional[LogFn], n: int,
                    meta: Dict[str, object]) -> None:
        actual_n = n
        if actual_n <= 0: return

        # 🌟 降低并发数，保护 API，防封锁防假死 (由 20 降至 10)
        max_workers = min(actual_n, 10)
        print(f"      🧪 并发测验开始 -> 共 {actual_n} 题 (并发数: {max_workers})")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._evaluate_once, genome, eval_fn, log_fn, meta) for _ in range(actual_n)]

            # 🌟 设置整体容忍的超时时间 (例如 40题，假设最多卡 3 分钟)
            timeout_s = 180
            done, not_done = concurrent.futures.wait(futures, timeout=timeout_s)

            if not_done:
                print(f"      ⚠️ [超时强杀] API 假死！已强行抛弃 {len(not_done)} 个未完成的超时测验。")
                for f in not_done:
                    f.cancel()

    def evolve(self, *, init_population: Sequence[StructuredGenome], eval_fn: EvalFn, log_fn: Optional[LogFn] = None) -> \
    List[StructuredGenome]:
        pop: List[StructuredGenome] = list(init_population)

        print("\n🌱 [Token 实验初始化] 开始评估初始种子...")
        for i, g in enumerate(pop):
            self._evaluate_n(g, eval_fn, log_fn, self.cfg.batch_size, meta={"phase": "init"})
            print(f"   👉 种子 {g.uid} 评估结束 | 成功测验: {g.n} 题 | 准确率: {g.mu()[0]:.2%}")
            if getattr(self.kernel, 'is_budget_exhausted', lambda: False)():
                break

        self._update_elite_pool(pop)

        if len(pop) == 1 and self.cfg.mu > 1:
            print(f"\n   🌱 [单点起源扩增] 将唯一种子克隆 {self.cfg.mu - 1} 份以填满初始种群...")
            base_seed = pop[0]
            for i in range(1, self.cfg.mu):
                c = base_seed.clone_with(loci=base_seed.loci, uid=f"{base_seed.uid}_c{i}", parents=[base_seed.uid],
                                         operator="init_clone")
                c.n = base_seed.n
                if base_seed.mean is not None: c.mean = np.copy(base_seed.mean)
                if base_seed.m2 is not None: c.m2 = np.copy(base_seed.m2)
                c.failure_cases = list(base_seed.failure_cases)
                pop.append(c)

        best_acc = max([g.mu()[0] for g in pop] + [0.0]) if pop else 0.0
        if hasattr(self.kernel, 'log_performance'):
            self.kernel.log_performance(best_acc)

        gen = 0
        print("\n🚀 [Token 驱动进化] 开始跨代繁殖...")
        while not getattr(self.kernel, 'is_budget_exhausted', lambda: False)():
            gen += 1
            tokens_used = getattr(self.kernel, 'total_tokens_consumed', 0)
            budget = getattr(self.kernel, 'token_budget', 0)

            print(f"\n--- [ 进化代数 {gen} | 当前 Token 消耗: {tokens_used} / {budget} ] ---")

            sel = nsga2_select(pop, mu=len(pop))
            offspring: List[StructuredGenome] = []

            for _ in range(self.cfg.lambd):
                if getattr(self.kernel, 'is_budget_exhausted', lambda: False)(): break

                op = self._choose_operator()

                if op.arity == 1:
                    parents = [tournament_select(pop, ranks=sel.ranks, crowding=sel.crowding, rng=self.rng)]
                else:
                    parents = [tournament_select(pop, ranks=sel.ranks, crowding=sel.crowding, rng=self.rng) for _ in
                               range(2)]

                # 🌟 增加探针打印：让你明确知道是不是在请求大模型改写时卡住了
                print(f"\n   🧬 [触发算子] {op.name} -> 正在请求大模型改写 Prompt...")
                t0 = time.time()
                child = self._make_offspring(parents, op)
                print(f"      ✅ 改写完成! 耗时: {time.time() - t0:.1f}s")

                if getattr(self.kernel, 'is_budget_exhausted', lambda: False)(): break

                # 开始并发评估子代
                self._evaluate_n(child, eval_fn, log_fn, self.cfg.batch_size, meta={"phase": "offspring", "gen": gen})

                # 打印这个子代最终扛住了多少题，拿了多少分
                print(f"      📊 子代 {child.uid} 最终得分: {child.mu()[0]:.2%} (有效题数: {child.n})\n")

                offspring.append(child)

            if not offspring: break

            pool = pop + offspring + self.elite_pool
            pop = nsga2_select(pool, mu=self.cfg.mu).selected
            self._update_elite_pool(pop)

            current_best = max([g.mu()[0] for g in self.elite_pool] + [0.0])
            if hasattr(self.kernel, 'log_performance'):
                self.kernel.log_performance(current_best)

            print(f"🏆 第 {gen} 代结束 | 全局最高准确率: {current_best:.2%} | 精英池大小: {len(self.elite_pool)}")

        print("\n🔴 实验结束：Token 预算已触达上限！")
        return self.elite_pool