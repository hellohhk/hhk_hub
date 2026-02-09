import json

import numpy as np

from spe_full.genome import StructuredGenome
from spe_full.optimizer import SPEOptimizer, SPEOptimizerConfig
from spe_full.operators import IntraLocusRewrite, LocusCrossover
from spe_full.scheduler import HVCUCBScheduleConfig
from spe_full.embedding import HashingNgramEmbedder


class _FakeCfg:
    model = "fake"
    temperature = 0.6
    timeout_s = 1
    api_key = "x"
    base_url = "http://localhost"


class FakeKernel:
    def __init__(self):
        self.config = _FakeCfg()
        self.calls = 0

    def chat(self, system_msg: str, user_msg: str, *, expect_json: bool, stream: bool, temperature: float, extra=None):
        self.calls += 1
        if expect_json:
            return type("R", (), {"content": json.dumps({"new_instruct": f"mut_{self.calls}"})})
        return type("R", (), {"content": "resp"})


def test_optimizer_runs_with_budget():
    kernel = FakeKernel()
    embedder = HashingNgramEmbedder(dim=64)
    cfg = SPEOptimizerConfig(
        budget=50,
        mu=3,
        lambd=4,
        gens=3,
        n_init=1,
        n_init_offspring=1,
        schedule_multiplier=1,
        seed=0,
        schedule_cfg=HVCUCBScheduleConfig(beta=0.2, ref_point=np.asarray([0.0, 0.0])),
    )

    def eval_fn(prompt: str):
        acc = 1.0 if "mut" in prompt else 0.0
        compact = 0.5
        resp = f"resp::{prompt}"
        return {
            "y": np.asarray([acc, compact], dtype=float),
            "problem_id": "p0",
            "response_len": len(resp),
            "response_emb": embedder.embed(resp),
        }

    init_pop = [
        StructuredGenome(
            loci={"L_role": "r", "L_instruct": "base", "L_const": "c", "L_style": "s"},
            uid=f"init_{i}",
            operator="init",
        )
        for i in range(cfg.mu)
    ]

    opt = SPEOptimizer(
        kernel=kernel,
        cfg=cfg,
        operators=[IntraLocusRewrite(), LocusCrossover()],
        operator_probs=[0.7, 0.3],
        embedder=embedder,
    )
    final_pop = opt.evolve(init_population=init_pop, eval_fn=eval_fn, log_fn=None)

    assert len(final_pop) == cfg.mu
    assert opt.used_budget <= cfg.budget
    assert max(float(g.mu()[0]) for g in final_pop) >= 0.0
