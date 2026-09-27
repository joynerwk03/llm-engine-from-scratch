"""Qwen3 forward pass, from scratch.

How to read this file
---------------------
Each section opens with an EXPLAINER block (the idea, from the ground up),
then the code, and names the test that checks it against Hugging Face:

    .venv/bin/python -m pytest tests/test_parts.py -q -s -k rms_norm   # section 1
    .venv/bin/python -m pytest tests/test_parts.py -q -s -k rope       # section 2
    .venv/bin/python -m pytest tests/test_parts.py -q -s -k attention  # section 3
    .venv/bin/python -m pytest tests/test_parts.py -q -s -k mlp        # section 4
    .venv/bin/python -m pytest tests/test_reference.py -q -s           # section 5: the whole model

EXPLAINER: what a forward pass is
---------------------------------
A language model turns a sequence of tokens into, for every position, a score
for every possible next token. For Qwen3-0.6B:

    token ids  [T]                  e.g. "The history of" -> [785, 3840, 315]
      -> embedding lookup  [T, 1024]    each token becomes a vector of 1,024 numbers
      -> 28 identical blocks, each:
             x = x + attention(norm(x))  tokens look at earlier tokens
             x = x + mlp(norm(x))        each token is processed on its own
      -> final norm        [T, 1024]
      -> output layer      [T, 151936]  one score (logit) per vocabulary entry

The vector x that flows through the blocks is called the **residual stream**.
Each block reads it (through a norm), computes something, and *adds* its
result back. Nothing overwrites x, so information from early layers survives
to the end, and gradients flowed easily during training. At inference we only
care that we reproduce the same arithmetic.

Row i of the output scores what comes after token i. Generating text is just:
run the model, take the best-scoring token at the last row, append it, repeat.
(That loop, and the KV cache that makes it fast, come in checkpoint 1b.)

Shapes: we handle one sequence at a time, so tensors are [T, ...]: no batch
dimension. Hugging Face uses [batch, heads, T, dim]; the tests translate.
"""

import json
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


# ── Config (plumbing, written) ───────────────────────────────────────────────

@dataclass
class Config:
    vocab_size: int            # 151,936 possible tokens
    hidden_size: int           # 1,024: the width of the residual stream
    intermediate_size: int     # 3,072: the MLP's inner width
    num_hidden_layers: int     # 28 blocks
    num_attention_heads: int   # 16 query heads
    num_key_value_heads: int   # 8 key/value heads (grouped-query attention, section 3)
    head_dim: int              # 128 numbers per head
    rms_norm_eps: float        # 1e-6
    rope_theta: float          # 1,000,000 (section 2)


def load_config(model_dir: str) -> Config:
    with open(f"{model_dir}/config.json") as f:
        c = json.load(f)
    return Config(**{k: c[k] for k in Config.__dataclass_fields__})


# ── 1. RMSNorm ───────────────────────────────────────────────────────────────
#
# EXPLAINER
# As x passes through 28 blocks, each adding to it, its size drifts: some
# tokens' vectors grow large, others stay small. A norm rescales a vector to a
# standard size before a block reads it, so every block sees inputs of a
# predictable scale.
#
# RMSNorm ("root mean square") is the simplest version:
#     rms   = sqrt(mean of x_i^2 over the vector's 1,024 numbers)
#     out_i = x_i / rms * weight_i
# Dividing by rms makes the vector's typical entry about 1; `weight` is a
# learned per-dimension scale, so the model can make some dimensions matter
# more. A tiny `eps` (1e-6) is added inside the square root so we never divide
# by zero.
#
# Precision detail (this is what makes you match Hugging Face exactly): the
# model's weights are bf16, a 16-bit format with only ~3 significant digits.
# Squaring and averaging 1,024 numbers in bf16 loses accuracy, so Qwen3 does it
# in float32: convert x to float32, normalize, convert back to x's original
# dtype, and only then multiply by `weight`.
#
# The same function is used three ways: on the residual stream (over 1,024
# numbers), and inside attention on each head's query and key (over 128
# numbers). "Over the last dimension" covers all three.

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """x: [..., d] (any leading shape), weight: [d]. Returns the same shape and dtype as x.
    Test: pytest tests/test_parts.py -k rms_norm
    """
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x.to(dtype)


# ── 2. Rotary position embeddings (RoPE) ─────────────────────────────────────
#
# EXPLAINER
# Attention (section 3) compares tokens by dot products of vectors. On its own
# it has no idea of order: "dog bites man" and "man bites dog" would look the
# same. RoPE injects position by *rotating* each query and key vector by an
# angle that grows with the token's position.
#
# Picture a 2-D vector (a, b) rotated by angle phi:
#     (a, b) -> (a cos phi - b sin phi,  a sin phi + b cos phi)
# If the query at position m is rotated by m*w and the key at position n by n*w,
# their dot product depends only on (m - n)*w: the *relative* distance. That is
# exactly what attention should care about ("the word 3 tokens back").
#
# A head vector has 128 numbers, so we rotate 64 pairs, each at its own
# frequency:
#     inv_freq[j] = 1 / theta^(2j/128),  j = 0..63,  theta = 1,000,000
# Pair 0 rotates fast (about 1 radian per token): it sees fine, nearby
# differences. Pair 63 rotates extremely slowly: it can tell apart positions
# thousands of tokens apart. Together they encode position at every scale.
#
# Which numbers form a pair? Hugging Face's convention ("rotate half") pairs
# dimension i with dimension i + 64. So with x1 = x[..., :64], x2 = x[..., 64:]:
#     rotated = x * cos + cat(-x2, x1) * sin
# where cos and sin are 128 wide: cos(cat(angle, angle)), because both members
# of a pair use the same angle. Check it against the 2-D formula: dimension i
# gets x1*cos - x2*sin, and dimension i+64 gets x2*cos + x1*sin.

_INV_FREQ = {}


def _inv_freq(head_dim: int, theta: float, device) -> torch.Tensor:
    """1 / theta^(2j / head_dim), computed once on the CPU (as Hugging Face does:
    the GPU's float32 pow rounds some entries differently in the last bit, which
    shows up in cos at large angles) and kept on `device`."""
    key = (head_dim, theta, str(device))
    if key not in _INV_FREQ:
        exponents = torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim
        _INV_FREQ[key] = (1.0 / (theta ** exponents)).to(device)
    return _INV_FREQ[key]


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """positions: LongTensor [T]. Returns (cos, sin), each float32 [T, head_dim].
    Test: pytest tests/test_parts.py -k rope
    """
    inv_freq = _inv_freq(head_dim, theta, positions.device)  # [head_dim // 2]
    angle = positions.float()[:, None] * inv_freq[None, :]   # [T, head_dim // 2]
    angle = torch.cat([angle, angle], dim=-1)                # both members of a pair share an angle
    return angle.cos(), angle.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [T, n_heads, head_dim]; cos, sin: [T, head_dim] float32. Returns x's shape and dtype.
    (Also used for a batch of single decode tokens: then T is the batch.)
    Test: pytest tests/test_parts.py -k rope
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    cos, sin = cos[:, None, :].to(x.dtype), sin[:, None, :].to(x.dtype)   # broadcast over heads
    return x * cos + rotated * sin


# ── 3. Attention ─────────────────────────────────────────────────────────────
#
# EXPLAINER
# Attention is how a token gathers information from earlier tokens. Each token
# produces three vectors from its residual-stream vector:
#     query q: "what am I looking for?"
#     key   k: "what do I contain?"
#     value v: "what do I pass on if someone attends to me?"
# Token i scores every token j by the dot product q_i . k_j (scaled by
# 1/sqrt(128) so the scores don't grow with the vector length), turns the
# scores into weights with softmax (positive, summing to 1), and outputs the
# weighted average of the values v_j.
#
# Causal: a token may only look at itself and earlier tokens (it can't see the
# future it's trying to predict). Set the scores for j > i to -infinity before
# the softmax; exp(-inf) = 0, so those tokens get zero weight.
#
# Heads: instead of one big attention, Qwen3 runs 16 small ones in parallel,
# each on its own 128-number slice (16 x 128 = 2,048). Different heads learn to
# look for different things (the previous word, the subject of the sentence,
# a matching bracket...). Their outputs are concatenated back to 2,048 numbers
# and mixed by o_proj into the residual stream's 1,024.
#
# Grouped-query attention (GQA): there are 16 query heads but only 8 key/value
# heads. Query heads 0 and 1 share key/value head 0, heads 2 and 3 share head
# 1, and so on (query head h uses kv head h // 2). This halves the keys and
# values the model must store per token: a big saving in the KV cache later.
#
# Qwen3's own twist: before RoPE, each head's query and key vectors get their
# own RMSNorm (weights q_norm and k_norm, 128 numbers each). It keeps the dot
# products well-scaled.
#
# Shapes, step by step, for T tokens:
#     x                      [T, 1024]
#     q = x @ q_proj.T       [T, 2048] -> view as [T, 16, 128]
#     k = x @ k_proj.T       [T, 1024] -> view as [T, 8, 128]   (v likewise)
#     q, k = rms_norm per head, then apply_rope
#     k, v -> repeat each kv head twice -> [T, 16, 128]
#     move heads first: [16, T, 128], so each head is its own [T, 128] matrix
#     scores = q @ k^T / sqrt(128)       [16, T, T]
#     causal mask, softmax (in float32), @ v   [16, T, 128]
#     back to [T, 16, 128] -> [T, 2048] -> @ o_proj.T -> [T, 1024]
# (In PyTorch, x @ W.T is exactly what an nn.Linear layer computes; the weight
# files store W as [out, in].)

#
# Implementation note: the scores/mask/softmax/weighted-sum steps are done by
# torch's fused scaled_dot_product_attention (one GPU kernel instead of five;
# it computes the softmax in float32 internally). enable_gqa=True applies
# "query head h uses kv head h // 2" (checked equal to repeat_interleave).
# The attention is split in two so the KV cache (checkpoint 1b) can sit in
# between: project_qkv makes this call's queries, keys and values; attend
# runs attention against whatever keys and values are visible.

def project_qkv(x: torch.Tensor, w: dict, cos: torch.Tensor, sin: torch.Tensor, cfg: Config):
    """x: [T, hidden_size], already normed. Returns q [T, n_heads, head_dim] and
    k, v [T, n_kv_heads, head_dim], with the per-head norms and RoPE applied to q and k."""
    T = x.shape[0]
    q = (x @ w["q_proj"].T).view(T, cfg.num_attention_heads, cfg.head_dim)
    k = (x @ w["k_proj"].T).view(T, cfg.num_key_value_heads, cfg.head_dim)
    v = (x @ w["v_proj"].T).view(T, cfg.num_key_value_heads, cfg.head_dim)
    q = apply_rope(rms_norm(q, w["q_norm"], cfg.rms_norm_eps), cos, sin)
    k = apply_rope(rms_norm(k, w["k_norm"], cfg.rms_norm_eps), cos, sin)
    return q, k, v


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    """q: [T, n_heads, head_dim], the queries of positions start_pos..start_pos+T-1.
    k, v: [S, n_kv_heads, head_dim] for positions 0..S-1, where S = start_pos + T.
    Causal: query i sees keys 0..start_pos+i. Returns [T, n_heads * head_dim]."""
    T, S = q.shape[0], k.shape[0]
    qh, kh, vh = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)   # heads first
    if S == T:   # the whole sequence in one call: the plain causal triangle
        out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, enable_gqa=True)
    else:        # new tokens after a cached prefix: query i sees keys 0..start_pos+i
        visible = torch.arange(S, device=q.device)[None, :] <= (start_pos + torch.arange(T, device=q.device))[:, None]
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=visible, enable_gqa=True)
    return out.transpose(0, 1).reshape(T, -1)


def attention(x: torch.Tensor, w: dict, cos: torch.Tensor, sin: torch.Tensor, cfg: Config) -> torch.Tensor:
    """x: [T, hidden_size], already normed. w: one layer's weights. Returns [T, hidden_size].
    Test: pytest tests/test_parts.py -k attention
    """
    q, k, v = project_qkv(x, w, cos, sin, cfg)
    return attend(q, k, v) @ w["o_proj"].T


# ── 4. MLP ───────────────────────────────────────────────────────────────────
#
# EXPLAINER
# Attention moves information between tokens; the MLP then processes each
# token on its own, and it's where much of a model's knowledge is thought to
# live (60% of each block's parameters here: 9.4M vs attention's 6.3M). Qwen3
# uses a "gated" MLP
# (SwiGLU):
#     gate = silu(x @ gate_proj.T)      [T, 3072]
#     up   = x @ up_proj.T              [T, 3072]
#     out  = (gate * up) @ down_proj.T  [T, 1024]
# silu(z) = z * sigmoid(z): a smooth ReLU. The gate decides, per inner
# dimension, how much of `up` to let through. Expanding to 3,072 and
# projecting back to 1,024 gives the network room to compute.

def mlp(x: torch.Tensor, w: dict) -> torch.Tensor:
    """x: [T, hidden_size], already normed. w: keys gate_proj, up_proj, down_proj.
    Returns [T, hidden_size].
    Test: pytest tests/test_parts.py -k mlp
    """
    return (F.silu(x @ w["gate_proj"].T) * (x @ w["up_proj"].T)) @ w["down_proj"].T


# ── 5. The model ─────────────────────────────────────────────────────────────

class Qwen3:
    def __init__(self, model_dir: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        """Loads the config and all 311 weight tensors. (Plumbing, written.)

        self.layers is a list of 28 dicts, one per block, with short keys:
            input_layernorm, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm,
            post_attention_layernorm, gate_proj, up_proj, down_proj
        self.embed [vocab, hidden], self.norm [hidden], self.lm_head [vocab, hidden]
        (lm_head is identical to embed in this model: "tied" weights).
        """
        self.cfg = load_config(model_dir)
        self.device, self.dtype = device, dtype
        raw = load_file(f"{model_dir}/model.safetensors", device=device)
        t = {name: tensor.to(dtype) for name, tensor in raw.items()}
        self.embed = t["model.embed_tokens.weight"]
        self.norm = t["model.norm.weight"]
        self.lm_head = t["lm_head.weight"]
        self.layers = []
        for i in range(self.cfg.num_hidden_layers):
            p = f"model.layers.{i}."
            self.layers.append({
                "input_layernorm": t[p + "input_layernorm.weight"],
                "q_proj": t[p + "self_attn.q_proj.weight"],
                "k_proj": t[p + "self_attn.k_proj.weight"],
                "v_proj": t[p + "self_attn.v_proj.weight"],
                "o_proj": t[p + "self_attn.o_proj.weight"],
                "q_norm": t[p + "self_attn.q_norm.weight"],
                "k_norm": t[p + "self_attn.k_norm.weight"],
                "post_attention_layernorm": t[p + "post_attention_layernorm.weight"],
                "gate_proj": t[p + "mlp.gate_proj.weight"],
                "up_proj": t[p + "mlp.up_proj.weight"],
                "down_proj": t[p + "mlp.down_proj.weight"],
            })

    # EXPLAINER
    # forward() wires the sections together, following the diagram at the top
    # of this file. The embedding lookup is indexing: self.embed[token_ids]
    # picks one row (one 1,024-number vector) per token id.
    #
    # The KV cache (engine/cache.py). To generate token 101, the model needs
    # attention over tokens 0..100. Their keys and values never change once
    # computed (each depends only on that token and the ones before it), so
    # we store them: after the prompt is processed once (the "prefill"), each
    # new token needs one small forward pass over just itself (a "decode"
    # step), reading everyone else's keys and values from the cache. Without
    # it, token 101 would recompute the whole sequence: quadratic work. That's
    # what start_pos is for: the tokens in this call sit at positions
    # start_pos, start_pos+1, ...

    @torch.inference_mode()
    def forward(self, token_ids: torch.Tensor, start_pos: int = 0, cache=None,
                last_only: bool = False) -> torch.Tensor:
        """One sequence. token_ids: LongTensor [T] on self.device, at positions
        start_pos..start_pos+T-1. cache: an engine.cache.SeqCache, or None to
        process the tokens on their own (then start_pos must be 0).
        Returns float32 logits [T, vocab_size] (only the last row if last_only).
        Tests: tests/test_reference.py (logits, and greedy generation through the cache).
        """
        cfg, eps = self.cfg, self.cfg.rms_norm_eps
        T = token_ids.shape[0]
        positions = torch.arange(start_pos, start_pos + T, device=token_ids.device)
        cos, sin = rope_cos_sin(positions, cfg.head_dim, cfg.rope_theta)
        x = self.embed[token_ids]
        for i, w in enumerate(self.layers):
            q, k, v = project_qkv(rms_norm(x, w["input_layernorm"], eps), w, cos, sin, cfg)
            if cache is not None:   # store this call's keys/values, then attend over the whole prefix
                K, V = cache.pool.k[i][cache.slot], cache.pool.v[i][cache.slot]   # [max_len, kv_heads, head_dim]
                K[start_pos:start_pos + T], V[start_pos:start_pos + T] = k, v
                k, v = K[:start_pos + T], V[:start_pos + T]
            x = x + attend(q, k, v, start_pos) @ w["o_proj"].T
            x = x + mlp(rms_norm(x, w["post_attention_layernorm"], eps), w)
        if last_only:
            x = x[-1:]
        return (rms_norm(x, self.norm, eps) @ self.lm_head.T).float()

    # EXPLAINER: batched decode (the heart of continuous batching)
    # Serving many users, each running sequence needs one new token per step.
    # Instead of B separate forward passes, stack their newest tokens into one
    # [B] batch: every matrix multiply (the projections, the MLP, the output
    # layer) now reads each weight once for all B sequences. That's the whole
    # reason batching raises throughput: a decode step at batch 1 is limited by
    # reading 1.2 GB of weights, and the same read serves the entire batch.
    #
    # Attention is the one per-sequence part: each sequence attends only to its
    # own cached keys and values. The engine keeps running sequence b in cache
    # slot b ("compact" slots), so the batch's cache is simply the first B rows
    # of the pool: a view, no copying. Each row is masked past its own length.
    #
    # Grouped heads without copies: with one new token per sequence, query
    # heads 2g and 2g+1 both read kv head g. Reshaping the 16 query heads to
    # [8 kv groups, 2 queries] lets every query attend to its own group's keys
    # directly, instead of duplicating every key and value to 16 heads.

    @torch.inference_mode()
    def decode(self, token_ids: torch.Tensor, positions: torch.Tensor, pool, attn_len: int) -> torch.Tensor:
        """One new token for each of B sequences in cache slots 0..B-1.
        token_ids, positions: LongTensor [B] on the GPU (positions[b] is where
        sequence b's new token sits). attn_len: how many cache positions to
        attend over (>= max(positions) + 1; the CUDA-graph path uses the full
        pool length so shapes never change). Returns float32 logits [B, vocab_size].
        Creates no tensors from Python values, so it can be captured in a CUDA graph."""
        cfg, eps = self.cfg, self.cfg.rms_norm_eps
        B, group = token_ids.shape[0], cfg.num_attention_heads // cfg.num_key_value_heads
        rows = torch.arange(B, device=token_ids.device)
        cos, sin = rope_cos_sin(positions, cfg.head_dim, cfg.rope_theta)
        visible = torch.arange(attn_len, device=token_ids.device)[None, :] <= positions[:, None]
        visible = visible[:, None, None, :]                            # [B, 1, 1, attn_len]
        x = self.embed[token_ids]                                      # [B, hidden]
        for i, w in enumerate(self.layers):
            q, k, v = project_qkv(rms_norm(x, w["input_layernorm"], eps), w, cos, sin, cfg)  # the batch acts as T
            pool.k[i][rows, positions] = k                             # each sequence's new key/value
            pool.v[i][rows, positions] = v
            K = pool.k[i][:B, :attn_len].transpose(1, 2)               # [B, kv_heads, attn_len, head_dim], a view
            V = pool.v[i][:B, :attn_len].transpose(1, 2)
            qg = q.view(B, cfg.num_key_value_heads, group, cfg.head_dim)   # [B, kv_heads, 2, head_dim]
            out = F.scaled_dot_product_attention(qg, K, V, attn_mask=visible)
            x = x + out.reshape(B, -1) @ w["o_proj"].T
            x = x + mlp(rms_norm(x, w["post_attention_layernorm"], eps), w)
        return (rms_norm(x, self.norm, eps) @ self.lm_head.T).float()
