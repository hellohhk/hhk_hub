import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from spe_full import (
    DeepSeekKernel,
    BBHEvaluator,
    SPEOptimizer,
    SPEOptimizerConfig,
    StructuredGenome,
    load_deepseek_config,
)
from spe_full.scheduler import HVCUCBScheduleConfig
from spe_full.embedding import HashingNgramEmbedder


def _jsonl_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log_fn(obj):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return log_fn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bbh_cache_dir", type=str, default="data/bbh")
    p.add_argument("--bbh_tasks", type=str, default="")
    p.add_argument("--bbh_n_shot", type=int, default=3)
    p.add_argument("--budget", type=int, default=200)
    p.add_argument("--gens", type=int, default=10)
    p.add_argument("--mu", type=int, default=4)
    p.add_argument("--lambda_", type=int, default=6)
    p.add_argument("--n_init", type=int, default=2)
    p.add_argument("--n_init_offspring", type=int, default=2)
    p.add_argument("--schedule_multiplier", type=int, default=2)
    p.add_argument("--beta", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_dir", type=str, default="logs_spe_full")
    p.add_argument("--role", type=str, default="Math Expert")
    p.add_argument("--instruct", type=str, default="Solve the problem step by step.")
    p.add_argument("--const", type=str, default="Put your final answer within \\boxed{}.")
    p.add_argument("--style", type=str, default="Academic and rigorous")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = load_deepseek_config(os.path.join(os.path.dirname(__file__), "apikey.txt"))
    kernel = DeepSeekKernel(cfg, verbose=False)

    tasks = [t.strip() for t in args.bbh_tasks.split(",") if t.strip()] or None
    evaluator = BBHEvaluator.from_cache(
        cache_dir=args.bbh_cache_dir,
        tasks=tasks,
        seed=args.seed,
        n_shot=args.bbh_n_shot,
        include_description=True,
    )
    embedder = HashingNgramEmbedder()

    def eval_fn(prompt: str):
        return evaluator.evaluate_once(kernel=kernel, prompt=prompt, embedder=embedder)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log_dir) / f"run_{run_id}.jsonl"
    log_fn = _jsonl_logger(log_path)

    optimizer_cfg = SPEOptimizerConfig(
        budget=args.budget,
        mu=args.mu,
        lambd=args.lambda_,
        gens=args.gens,
        n_init=args.n_init,
        n_init_offspring=args.n_init_offspring,
        schedule_multiplier=args.schedule_multiplier,
        seed=args.seed,
        schedule_cfg=HVCUCBScheduleConfig(beta=args.beta, ref_point=np.asarray([0.0, 0.0], dtype=float)),
    )

    optimizer = SPEOptimizer(kernel=kernel, cfg=optimizer_cfg, embedder=embedder)

    init_pop = []
    for i in range(args.mu):
        loci = {"L_role": args.role, "L_instruct": args.instruct, "L_const": args.const, "L_style": args.style}
        init_pop.append(StructuredGenome(loci=loci, uid=f"init_{i}", operator="init"))

    log_fn(
        {
            "phase": "run_start",
            "bbh_cache_dir": args.bbh_cache_dir,
            "bbh_tasks": tasks,
            "bbh_n_shot": args.bbh_n_shot,
            "budget": args.budget,
            "gens": args.gens,
            "mu": args.mu,
            "lambda": args.lambda_,
            "n_init": args.n_init,
            "n_init_offspring": args.n_init_offspring,
            "schedule_multiplier": args.schedule_multiplier,
            "beta": args.beta,
            "seed": args.seed,
        }
    )

    final_pop = optimizer.evolve(init_population=init_pop, eval_fn=eval_fn, log_fn=log_fn)
    pareto = optimizer.extract_pareto(final_pop)

    best = max(final_pop, key=lambda g: float(g.mu()[0]))

    summary = {
        "phase": "run_end",
        "budget_used": optimizer.used_budget,
        "final_population": [{"uid": g.uid, "mu": g.mu().tolist(), "n": g.n, "operator": g.operator} for g in final_pop],
        "pareto": [{"uid": g.uid, "mu": g.mu().tolist(), "n": g.n, "operator": g.operator} for g in pareto],
        "best_by_accuracy": {"uid": best.uid, "mu": best.mu().tolist(), "prompt": best.prompt_text()},
        "log_path": str(log_path),
    }
    log_fn(summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
