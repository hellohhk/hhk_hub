import json

import numpy as np

from spe_full.bbh import BBHEvaluator, answers_equivalent
from spe_full.embedding import HashingNgramEmbedder


class _FakeCfg:
    model = "fake"
    temperature = 0.6
    timeout_s = 1
    api_key = "x"
    base_url = "http://localhost"


class FakeKernel:
    def __init__(self, reply: str):
        self.config = _FakeCfg()
        self.reply = reply

    def chat(self, system_msg: str, user_msg: str, *, expect_json: bool, stream: bool, temperature: float, extra=None):
        return type("R", (), {"content": self.reply})


def test_answers_equivalent_multi():
    assert answers_equivalent("A", "A ||| B")
    assert answers_equivalent("yes", "yes")


def test_bbh_evaluator_local_dir(tmp_path):
    task = {
        "name": "toy_task",
        "description": "Return YES if input contains 1 else NO.",
        "examples": [{"input": "1", "target": "YES"}, {"input": "0", "target": "NO"}],
    }
    (tmp_path / "toy_task.json").write_text(json.dumps(task), encoding="utf-8")

    ev = BBHEvaluator(str(tmp_path), tasks=["toy_task"], seed=0, n_shot=0)
    embedder = HashingNgramEmbedder(dim=64)
    kernel = FakeKernel(reply="YES")

    out = ev.evaluate_once(kernel=kernel, prompt="sys", embedder=embedder)
    assert "y" in out
    y = out["y"]
    assert isinstance(y, np.ndarray)
    assert y.shape[0] == 2
    assert out["task"] == "toy_task"
    assert out["response_len"] >= 0

