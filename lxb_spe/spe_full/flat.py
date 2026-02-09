from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .embedding import HashingNgramEmbedder, l2_distance
from .kernel import DeepSeekKernel
from .scheduler import HVCUCBScheduleConfig, pick_hvc_ucb
from .selection import nsga2_select, tournament_select


@dataclass
class FlatGenome:
    text: str
    uid: str
    parents: Tuple[str, ...] = field(default_factory=tuple)
    operator: str = "init"
    radius: Dict[str, float] = field(default_factory=dict)

    n: int = 0
    mean: Optional[np.ndarray] = None
    m2: Optional[np.ndarray] = None
    history: List[np.ndarray] = field(default_factory=list)

    output_emb_n: int = 0
    output_emb_mean: Optional[np.ndarray] = None
    output_emb_m2: Optional[np.ndarray] = None

    output_len_n: int = 0
    output_len_mean: float = 0.0
    output_len_m2: float = 0.0

    def prompt_text(self) -> str:
        return self.text

    def update(self, y: np.ndarray, keep_history: bool = True) -> None:
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

    def mu(self) -> np.ndarray:
        if self.mean is None:
            return np.zeros(2, dtype=float)
        return self.mean

    def var(self) -> np.ndarray:
        if self.mean is None or self.m2 is None or self.n < 2:
            return np.full_like(self.mu(), np.inf, dtype=float)
        return self.m2 / (self.n - 1)

    def update_output_embedding(self, emb: np.ndarray) -> None:
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
        if self.output_emb_mean is None or self.output_emb_m2 is None or self.output_emb_n < 2:
            return float("inf")
        var = self.output_emb_m2 / (self.output_emb_n - 1)
        return float(np.sum(var))

    def update_output_length(self, length: int) -> None:
        x = float(max(0, int(length)))
        self.output_len_n += 1
        delta = x - self.output_len_mean
        self.output_len_mean = self.output_len_mean + delta / self.output_len_n
        delta2 = x - self.output_len_mean
        self.output_len_m2 = self.output_len_m2 + delta * delta2

    def output_len_var(self) -> float:
        if self.output_len_n < 2:
            return float("inf")
        return float(self.output_len_m2 / (self.output_len_n - 1))


@dataclass(frozen=True)
class FlatOperator:
    name: str
    arity: int

    def apply(self, parents: Sequence[FlatGenome], *, kernel: DeepSeekKernel, rng: np.random.Generator) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class FlatRewrite(FlatOperator):
    name: str = "FlatRewrite"
    arity: int = 1

    def apply(self, parents: Sequence[FlatGenome], *, kernel: DeepSeekKernel, rng: np.random.Generator) -> str:
        p = parents[0]
        system_msg = (
            "You are a meta-prompt optimizer.\n"
            "Rewrite the entire prompt to improve reasoning quality.\n"
            "Return ONLY the rewritten prompt text."
        )
        user_msg = f"Current prompt:\n{p.text}"
        r = kernel.chat(system_msg, user_msg, expect_json=False, stream=True, temperature=kernel.config.temperature)
        return (r.content or p.text).strip() or p.text


@dataclass(frozen=True)
class FlatRefine(FlatOperator):
    name: str = "FlatRefine"
    arity: int = 1

    def apply(self, parents: Sequence[FlatGenome], *, kernel: DeepSeekKernel, rng: np.random.Generator) -> str:
        p = parents[0]
        system_msg = (
            "You are a meta-prompt optimizer.\n"
            "Make minimal edits to the entire prompt to improve clarity and reduce failure cases.\n"
            "Return ONLY the refined prompt text."
        )
        user_msg = f"Current prompt:\n{p.text}"
        r = kernel.chat(system_msg, user_msg, expect_json=False, stream=True, temperature=max(0.1, kernel.config.temperature * 0.5))
        return (r.content or p.text).strip() or p.text


@dataclass(frozen=True)
class FlatMix(FlatOperator):
    name: str = "FlatMix"
    arity: int = 2

    def apply(self, parents: Sequence[FlatGenome], *, kernel: DeepSeekKernel, rng: np.random.Generator) -> str:
        a, b = parents[0], parents[1]
        system_msg = (
            "You are a meta-prompt optimizer.\n"
            "Fuse two prompts into one, keeping the final prompt concise.\n"
            "Return ONLY the fused prompt text."
        )
        user_msg = f"Prompt A:\n{a.text}\n\nPrompt B:\n{b.text}\n"
        r = kernel.chat(system_msg, user_msg, expect_json=False, stream=True, temperature=kernel.config.temperature)
        return (r.content or a.text).strip() or a.text


@dataclass(frozen=True)
class FlatSwap(FlatOperator):
    name: str = "FlatSwap"
    arity: int = 2

    def apply(self, parents: Sequence[FlatGenome], *, kernel: DeepSeekKernel, rng: np.random.Generator) -> str:
        a, b = parents[0], parents[1]
        return b.text if bool(rng.integers(0, 2)) else a.text


@dataclass(frozen=True)
class FlatOptimizerConfig:
    budget: int
    mu: int = 4
    lambd: int = 4
    gens: int = 10
    n_init: int = 2
    n_init_offspring: int = 2
    schedule_multiplier: int = 2
    seed: int = 0
    keep_history: bool = False
    schedule_cfg: HVCUCBScheduleConfig = field(default_factory=HVCUCBScheduleConfig)


EvalFn = Callable[[str], Dict[str, object]]
LogFn = Callable[[Dict[str, object]], None]


def _uid(prefix: str, rng: np.random.Generator) -> str:
    return f"{prefix}_{int(rng.integers(10**12, 10**13 - 1))}"


class FlatOptimizer:
    def __init__(
        self,
        *,
        kernel: DeepSeekKernel,
        cfg: FlatOptimizerConfig,
        operators: Optional[Sequence[FlatOperator]] = None,
        operator_probs: Optional[Sequence[float]] = None,
        embedder: Optional[HashingNgramEmbedder] = None,
    ):
        self.kernel = kernel
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.used_budget = 0
        self.total_samples = 0
        self.embedder = embedder or HashingNgramEmbedder()

        self.operators = list(operators) if operators is not None else [FlatRewrite(), FlatRefine(), FlatMix(), FlatSwap()]
        if operator_probs is None:
            self.operator_probs = np.asarray([0.4, 0.25, 0.25, 0.1], dtype=float)
        else:
            self.operator_probs = np.asarray(operator_probs, dtype=float)
        self.operator_probs = self.operator_probs / np.sum(self.operator_probs)

    def _op_cost(self, op: FlatOperator) -> int:
        if op.name in {"FlatRewrite", "FlatRefine", "FlatMix"}:
            return 1
        return 0

    def _evaluate_once(self, genome: FlatGenome, eval_fn: EvalFn, log_fn: Optional[LogFn], meta: Dict[str, object]) -> bool:
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

    def _evaluate_n(self, genome: FlatGenome, eval_fn: EvalFn, log_fn: Optional[LogFn], n: int, meta: Dict[str, object]) -> None:
        for _ in range(n):
            ok = self._evaluate_once(genome, eval_fn, log_fn, meta)
            if not ok:
                break

    def _choose_operator(self) -> FlatOperator:
        idx = int(self.rng.choice(len(self.operators), p=self.operator_probs))
        return self.operators[idx]

    def _make_offspring(self, parents: Sequence[FlatGenome], op: FlatOperator) -> FlatGenome:
        base = parents[0]
        new_text = op.apply(parents, kernel=self.kernel, rng=self.rng)
        uid = _uid("flat", self.rng)
        child = FlatGenome(text=new_text, uid=uid, parents=tuple(p.uid for p in parents), operator=op.name)
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

    def evolve(self, *, init_population: Sequence[FlatGenome], eval_fn: EvalFn, log_fn: Optional[LogFn] = None) -> List[FlatGenome]:
        pop: List[FlatGenome] = list(init_population)
        if not pop:
            raise ValueError("init_population must not be empty")

        for g in pop:
            self._evaluate_n(g, eval_fn, log_fn, self.cfg.n_init, meta={"phase": "init"})

        for gen in range(self.cfg.gens):
            if self.used_budget >= self.cfg.budget:
                break

            sel = nsga2_select(pop, mu=len(pop))
            offspring: List[FlatGenome] = []

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

            pool: List[FlatGenome] = pop + offspring
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

