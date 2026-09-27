"""Continuous batching must not change anyone's output: every request run
through the engine, alongside others, gives the same greedy tokens as that
prompt generated alone (float32, where batch-shape rounding can't flip a token
in practice)."""

import os
import pytest
import torch
from transformers import AutoTokenizer

from engine.engine import Engine
from engine.generate import greedy_generate

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.expanduser("~/models/Qwen3-0.6B"))
N = 24   # new tokens per request

TEXTS = [
    "The history of computing is a story of trading one scarce resource for another.",
    "def fibonacci(n):",
    "Translate to French: the weather is lovely today, and",
    "List three prime numbers:",
]


@pytest.fixture(scope="module", params=[False, True], ids=["eager", "cuda_graphs"])
def engine(request):
    # 2 slots, so the other requests must wait for one to free up
    return Engine(MODEL_DIR, max_slots=2, max_len=256, dtype=torch.float32, cuda_graphs=request.param)


@pytest.fixture(scope="module")
def prompts():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    return [tok(t, add_special_tokens=False)["input_ids"] for t in TEXTS]


@pytest.fixture(scope="module")
def alone(engine, prompts):
    return [greedy_generate(engine.model, p, N) for p in prompts]


def run(engine, arrivals):
    """arrivals: {step_number: [(request_id, prompt_ids), ...]}. Steps until idle."""
    outputs, step = {}, 0
    while engine.has_work() or any(s >= step for s in arrivals):
        for rid, ids in arrivals.get(step, []):
            engine.add_request(rid, ids, N, temperature=0.0, ignore_eos=True)
        for rid, token, _ in engine.step():
            outputs.setdefault(rid, []).append(token)
        step += 1
    return outputs


def test_batched_equals_alone(engine, prompts, alone):
    out = run(engine, {0: [(f"r{i}", p) for i, p in enumerate(prompts)]})
    for i in range(len(prompts)):
        assert out[f"r{i}"] == alone[i], f"request {i} differs when batched"
    assert not engine.running, "every sequence left the batch"


def test_staggered_arrivals(engine, prompts, alone):
    """Requests join mid-flight (and wait for a free slot); outputs still match."""
    out = run(engine, {0: [("a", prompts[0])], 5: [("b", prompts[1])], 7: [("c", prompts[2])], 30: [("d", prompts[3])]})
    for rid, i in zip("abcd", range(4)):
        assert out[rid] == alone[i], f"request {rid} differs"
    assert not engine.has_work() and not engine.running


def test_rejects_too_long(engine):
    with pytest.raises(ValueError):
        engine.add_request("x", [1] * 250, 16)
