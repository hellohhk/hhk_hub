import numpy as np

from spe_full.embedding import HashingNgramEmbedder
from spe_full.flat import FlatGenome, FlatOptimizer, FlatOptimizerConfig, FlatRewrite
from spe_full.scheduler import HVCUCBScheduleConfig


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
        return type("R", (), {"content": f"flat_mut_{self.calls}"})


def test_flat_optimizer_runs():
    kernel = FakeKernel()
    embedder = HashingNgramEmbedder(dim=64)
    cfg = FlatOptimizerConfig(
        budget=60,
        mu=3,
        lambd=3,
        gens=2,
        n_init=1,
        n_init_offspring=1,
        schedule_multiplier=1,
        seed=0,
        schedule_cfg=HVCUCBScheduleConfig(beta=0.2, ref_point=np.asarray([0.0, 0.0])),
    )

    def eval_fn(prompt: str):
        acc = 1.0 if "flat_mut" in prompt else 0.0
        compact = 0.5
        resp = f"resp::{prompt}"
        return {
            "y": np.asarray([acc, compact], dtype=float),
            "problem_id": "p0",
            "response_len": len(resp),
            "response_emb": embedder.embed(resp),
        }

    init_pop = [FlatGenome(text="base prompt", uid=f"init_{i}") for i in range(cfg.mu)]
    opt = FlatOptimizer(kernel=kernel, cfg=cfg, operators=[FlatRewrite()], operator_probs=[1.0], embedder=embedder)
    final_pop = opt.evolve(init_population=init_pop, eval_fn=eval_fn, log_fn=None)

    assert len(final_pop) == cfg.mu
    assert opt.used_budget <= cfg.budget
    assert any(g.output_len_n > 0 for g in final_pop)
