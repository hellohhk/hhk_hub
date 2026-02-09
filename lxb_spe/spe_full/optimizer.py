from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .genome import StructuredGenome
from .kernel import DeepSeekKernel
from .embedding import HashingNgramEmbedder, l2_distance
from .operators import (
    IntraLocusRefine,
    IntraLocusRewrite,
    KernelOperator,
    LocusCrossover,
    SemanticInterpolation,
)
from .pareto import pareto_front
from .scheduler import HVCUCBScheduleConfig, pick_hvc_ucb
from .selection import nsga2_select, tournament_select


@dataclass(frozen=True)
class SPEOptimizerConfig:
    budget: int
    mu: int = 4
    lambd: int = 4
    gens: int = 10
    n_init: int = 2
    n_init_offspring: int = 2
    schedule_multiplier: int = 2
    invariant_loci: Tuple[str, ...] = ("L_const",)
    seed: int = 0
    keep_history: bool = False
    schedule_cfg: HVCUCBScheduleConfig = field(default_factory=HVCUCBScheduleConfig)


EvalFn = Callable[[str], Dict[str, object]]
LogFn = Callable[[Dict[str, object]], None]


def _uid(prefix: str, rng: np.random.Generator) -> str:
    return f"{prefix}_{int(rng.integers(10**12, 10**13 - 1))}"


class SPEOptimizer:
    def __init__(
        self,
        *,
        kernel: DeepSeekKernel,
        cfg: SPEOptimizerConfig,
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

        ops = list(operators) if operators is not None else [IntraLocusRewrite(), IntraLocusRefine(), LocusCrossover(), SemanticInterpolation()]
        self.operators = ops
        if operator_probs is None:
            self.operator_probs = np.asarray([0.35, 0.25, 0.2, 0.2], dtype=float)
        else:
            self.operator_probs = np.asarray(operator_probs, dtype=float)
        self.operator_probs = self.operator_probs / np.sum(self.operator_probs)

    def _op_cost(self, op: KernelOperator) -> int:
        if op.name in {"K_rew", "K_ref", "K_mix"}:
            return 1
        return 0

    def _evaluate_once(self, genome: StructuredGenome, eval_fn: EvalFn, log_fn: Optional[LogFn], meta: Dict[str, object]) -> bool:
        if self.used_budget >= self.cfg.budget:
            return False
        out = eval_fn(genome.prompt_text())
        y = out.get("y")
        if y is None:
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
        genome.update(np.asarray(y, dtype=float), keep_history=self.cfg.keep_history)
        self.used_budget += 1
        self.total_samples += 1
        if log_fn is not None:
            rec = dict(meta)
            rec.update(
                {
                    "uid": genome.uid,
                    "operator": genome.operator,
                    "parents": list(genome.parents),
                    "n": genome.n,
                    "mu": genome.mu().tolist(),
                    "y": np.asarray(y, dtype=float).tolist(),
                    "budget_used": self.used_budget,
                    "total_samples": self.total_samples,
                    "prompt_radius": dict(genome.radius),
                    "output_emb_trace_var": genome.output_emb_trace_var(),
                    "output_len_var": genome.output_len_var(),
                    "y_var": genome.var().tolist(),
                }
            )
            for k in ["problem_id", "subject", "extracted", "gold", "response_len"]:
                if k in out:
                    rec[k] = out[k]
            log_fn(rec)
        return True

    def _evaluate_n(self, genome: StructuredGenome, eval_fn: EvalFn, log_fn: Optional[LogFn], n: int, meta: Dict[str, object]) -> None:
        for _ in range(n):
            ok = self._evaluate_once(genome, eval_fn, log_fn, meta)
            if not ok:
                break

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
            if len(parents) > 1:
                p2_emb = self.embedder.embed(parents[1].prompt_text())
                child.radius["prompt_disp_l2_p2"] = l2_distance(child_emb, p2_emb)
                child.radius["prompt_disp_l2_mean"] = float(
                    0.5 * (child.radius["prompt_disp_l2_p1"] + child.radius["prompt_disp_l2_p2"])
                )
        except Exception:
            pass
        return child

    def evolve(
        self,
        *,
        init_population: Sequence[StructuredGenome],
        eval_fn: EvalFn,
        log_fn: Optional[LogFn] = None,
    ) -> List[StructuredGenome]:
        pop: List[StructuredGenome] = list(init_population)
        if not pop:
            raise ValueError("init_population must not be empty")

        for g in pop:
            self._evaluate_n(g, eval_fn, log_fn, self.cfg.n_init, meta={"phase": "init"})

        for gen in range(self.cfg.gens):
            if self.used_budget >= self.cfg.budget:
                break

            sel = nsga2_select(pop, mu=len(pop))
            offspring: List[StructuredGenome] = []

            for _ in range(self.cfg.lambd):
                if self.used_budget >= self.cfg.budget:
                    break
                op = self._choose_operator()
                if self.used_budget + self._op_cost(op) > self.cfg.budget:
                    break

                if op.arity == 1:
                    p1 = tournament_select(pop, ranks=sel.ranks, crowding=sel.crowding, rng=self.rng)
                    parents = [p1]
                else:
                    p1 = tournament_select(pop, ranks=sel.ranks, crowding=sel.crowding, rng=self.rng)
                    p2 = tournament_select(pop, ranks=sel.ranks, crowding=sel.crowding, rng=self.rng)
                    parents = [p1, p2]

                if self._op_cost(op) > 0:
                    self.used_budget += self._op_cost(op)
                child = self._make_offspring(parents, op)
                offspring.append(child)
                self._evaluate_n(child, eval_fn, log_fn, self.cfg.n_init_offspring, meta={"phase": "offspring_init", "gen": gen})

            pool = pop + offspring
            schedule_steps = int(self.cfg.schedule_multiplier * max(1, len(pool)))
            for step in range(schedule_steps):
                if self.used_budget >= self.cfg.budget:
                    break
                target = pick_hvc_ucb(pool, cfg=self.cfg.schedule_cfg, total_samples=self.total_samples)
                self._evaluate_once(target, eval_fn, log_fn, meta={"phase": "schedule", "gen": gen, "step": step})

            pop = nsga2_select(pool, mu=self.cfg.mu).selected

            if log_fn is not None:
                log_fn(
                    {
                        "phase": "gen_end",
                        "gen": gen,
                        "budget_used": self.used_budget,
                        "population": [g.uid for g in pop],
                        "population_mu": [g.mu().tolist() for g in pop],
                    }
                )

        return pop

    def extract_pareto(self, pop: Sequence[StructuredGenome]) -> List[StructuredGenome]:
        idxs = pareto_front([g.mu() for g in pop])
        return [pop[i] for i in idxs]
