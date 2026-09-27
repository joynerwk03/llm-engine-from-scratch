"""Correctness against Hugging Face transformers, on the same weights.

Both models run in float32, so the only differences left are the order of
floating-point operations: logits should agree to ~1e-4, and the most likely
next token must be identical at every position.

    .venv/bin/python -m pytest tests -q -s            # -s prints the measured differences
"""

import os
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.expanduser("~/models/Qwen3-0.6B"))
DEVICE = "cuda"
MAX_ABS_DIFF = 1e-2   # fp32 vs fp32; a correct forward pass lands near 1e-4

PROMPTS = [
    "The history of computing is a story of trading one scarce resource for another.",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_DIR)


@pytest.fixture(scope="module")
def prompts(tokenizer):
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Tim buys fireworks worth $400 and another pack worth twice that. How much did he spend?"}],
        add_generation_prompt=True, tokenize=False)
    return [tokenizer(p, add_special_tokens=False)["input_ids"] for p in PROMPTS + [chat]]


@pytest.fixture(scope="module")
def reference():
    return AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).to(DEVICE).eval()


@pytest.fixture(scope="module")
def ours():
    from engine.model import Qwen3
    return Qwen3(MODEL_DIR, device=DEVICE, dtype=torch.float32)


@torch.inference_mode()
def test_logits_match_reference(ours, reference, prompts):
    for ids in prompts:
        t = torch.tensor(ids, device=DEVICE)
        got = ours.forward(t)
        want = reference(input_ids=t[None]).logits[0].float()
        assert got.shape == want.shape, f"shape {tuple(got.shape)}, expected {tuple(want.shape)}"
        diff = (got - want).abs().max().item()
        same_top = (got.argmax(-1) == want.argmax(-1)).float().mean().item()
        print(f"\n  {len(ids):3d} tokens: max |logit diff| {diff:.2e}, same top token at {same_top:.0%} of positions")
        assert diff < MAX_ABS_DIFF, f"logits differ by up to {diff:.3e}"
        assert same_top == 1.0


@torch.inference_mode()
def test_greedy_generation_matches_reference(ours, reference, prompts):
    """Checkpoint 1b: 64 greedy tokens through the KV cache, identical to Hugging Face."""
    try:
        from engine.generate import greedy_generate
    except ImportError:
        pytest.skip("engine/generate.py not written yet (checkpoint 1b)")
    for ids in prompts:
        got = greedy_generate(ours, ids, max_new_tokens=64)
        out = reference.generate(torch.tensor([ids], device=DEVICE), max_new_tokens=64, do_sample=False)
        want = out[0, len(ids):].tolist()   # may stop early at an end-of-text token
        first_diff = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
        print(f"\n  {len(ids):3d}-token prompt: {len(want)} reference tokens, "
              f"first difference at {'none' if first_diff is None else first_diff}")
        assert got[:len(want)] == want
