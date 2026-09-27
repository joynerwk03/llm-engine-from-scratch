"""One test per section of engine/model.py, each against the matching Hugging
Face component with the same (real) weights. Run one at a time as you go:

    .venv/bin/python -m pytest tests/test_parts.py -q -s -k rms_norm
    .venv/bin/python -m pytest tests/test_parts.py -q -s -k rope
    .venv/bin/python -m pytest tests/test_parts.py -q -s -k attention
    .venv/bin/python -m pytest tests/test_parts.py -q -s -k mlp

Hugging Face shapes are [batch, heads, T, dim]; ours are [T, heads, dim] with
no batch. The adapters below translate, so your functions keep the simple shapes.
"""

import os
import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from engine import model as ours

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.expanduser("~/models/Qwen3-0.6B"))
DEVICE = "cuda"
T = 7   # tokens in the random test inputs


@pytest.fixture(scope="module")
def hf():
    return AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).to(DEVICE).eval()


@pytest.fixture(scope="module")
def cfg():
    return ours.load_config(MODEL_DIR)


@pytest.fixture(scope="module")
def layer0(hf):
    """Layer 0's weights, in the dict layout engine.model.Qwen3 uses."""
    L = hf.model.layers[0]
    a, m = L.self_attn, L.mlp
    return {"input_layernorm": L.input_layernorm.weight, "post_attention_layernorm": L.post_attention_layernorm.weight,
            "q_proj": a.q_proj.weight, "k_proj": a.k_proj.weight, "v_proj": a.v_proj.weight, "o_proj": a.o_proj.weight,
            "q_norm": a.q_norm.weight, "k_norm": a.k_norm.weight,
            "gate_proj": m.gate_proj.weight, "up_proj": m.up_proj.weight, "down_proj": m.down_proj.weight}


def randn(*shape, dtype=torch.float32):
    g = torch.Generator(device=DEVICE).manual_seed(0)
    return torch.randn(*shape, generator=g, device=DEVICE, dtype=dtype)


def report(name, got, want):
    diff = (got.float() - want.float()).abs().max().item()
    print(f"\n  {name}: max |difference| {diff:.2e}")


@torch.inference_mode()
def test_rms_norm(hf, cfg):
    ref = hf.model.layers[0].input_layernorm            # Hugging Face's Qwen3RMSNorm, real weight
    x = randn(T, cfg.hidden_size) * 3
    got = ours.rms_norm(x, ref.weight, cfg.rms_norm_eps)
    report("float32", got, ref(x))
    torch.testing.assert_close(got, ref(x))
    # the precision detail: in bf16, the mean must still be computed in float32
    xb, wb = x.bfloat16(), ref.weight.bfloat16()
    want = wb * (xb.float() * torch.rsqrt(xb.float().pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)).to(torch.bfloat16)
    got = ours.rms_norm(xb, wb, cfg.rms_norm_eps)
    report("bfloat16", got, want)
    assert got.dtype == torch.bfloat16, "return x's dtype"
    torch.testing.assert_close(got, want)


@torch.inference_mode()
def test_rope(hf, cfg):
    positions = torch.tensor([0, 1, 2, 5, 100, 1000, 8000], device=DEVICE)
    cos, sin = ours.rope_cos_sin(positions, cfg.head_dim, cfg.rope_theta)
    assert cos.shape == (len(positions), cfg.head_dim) and cos.dtype == torch.float32
    x = randn(len(positions), cfg.num_attention_heads, cfg.head_dim)
    want_cos, want_sin = hf.model.rotary_emb(x, positions[None])     # [1, T, head_dim]
    report("cos", cos, want_cos[0]); report("sin", sin, want_sin[0])
    torch.testing.assert_close(cos, want_cos[0])
    torch.testing.assert_close(sin, want_sin[0])
    got = ours.apply_rope(x, cos, sin)
    q_hf = x.transpose(0, 1)[None]                                    # [1, heads, T, dim]
    want, _ = apply_rotary_pos_emb(q_hf, q_hf, want_cos, want_sin)
    want = want[0].transpose(0, 1)                                    # back to [T, heads, dim]
    report("rotated", got, want)
    torch.testing.assert_close(got, want)


@torch.inference_mode()
def test_mlp(hf, cfg, layer0):
    x = randn(T, cfg.hidden_size)
    got = ours.mlp(x, layer0)
    want = hf.model.layers[0].mlp(x[None])[0]
    report("mlp", got, want)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


@torch.inference_mode()
def test_attention(hf, cfg, layer0):
    x = randn(T, cfg.hidden_size)
    positions = torch.arange(T, device=DEVICE)
    cos, sin = ours.rope_cos_sin(positions, cfg.head_dim, cfg.rope_theta)
    got = ours.attention(x, layer0, cos, sin, cfg)
    assert got.shape == (T, cfg.hidden_size), f"shape {tuple(got.shape)}"
    hf_cos, hf_sin = hf.model.rotary_emb(x, positions[None])
    causal = torch.full((T, T), float("-inf"), device=DEVICE).triu(1)[None, None]   # additive mask
    want, _ = hf.model.layers[0].self_attn(x[None], (hf_cos, hf_sin), causal)
    report("attention", got, want[0])
    torch.testing.assert_close(got, want[0], atol=1e-4, rtol=1e-4)
