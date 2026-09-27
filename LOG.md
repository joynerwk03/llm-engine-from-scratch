# Log

Dated entries: what was built or measured, the prediction (before running),
the result, the verdict. Newest at the bottom.

## 2026-09-27: kickoff

- **Why:** to understand what serving engines like vLLM and SGLang do
  inside by building a minimal one: start from a basic server and add a few
  of their core features, checking each against Hugging Face and measuring
  against vLLM.
- **How it was built:** planned as a guided build, with skeletons and
  explainer blocks for William to fill in. William then asked Claude Code to
  build it, which is how this version was made. The explainer blocks stay in
  the code.
- **Decisions (William):** the Qwen3 forward pass written from scratch (no
  transformers model code) and checked against Hugging Face; the name
  `llm-engine-from-scratch`.
- **Model facts, checked in the files** (Qwen3-0.6B from Hugging Face):
  - 311 bf16 tensors, 28 layers, hidden 1024, 16 query heads, 8 key/value
    heads, head_dim 128, MLP 3072, vocab 151,936, RoPE theta 1e6, RMSNorm
    eps 1e-6.
  - `lm_head.weight` is present and identical to `model.embed_tokens.weight`
    (tied, as `config.json` says).

## 2026-09-27: Checkpoint 1, the forward pass and KV cache match Hugging Face

- `tests/test_parts.py` (each piece vs its Hugging Face counterpart, real
  weights, fp32): RMSNorm, RoPE, MLP identical (0.0); attention 2e-6.
- `tests/test_reference.py`, the whole model in fp32:
  - logits **bit-identical** to transformers 5.17 on 3 prompts (13-29
    tokens);
  - 64 greedy tokens through the KV cache identical on all 3.
- One fix on the way: the RoPE test first failed by 1.2e-5 at position 8,000.
  Hugging Face computes `inv_freq` on the CPU; the GPU's float32 `pow`
  rounds some entries differently in the last bit, and a large angle
  (~1,146 rad) amplifies it. Now computed on the CPU and cached.

## 2026-09-27: Checkpoints 2-3, served, batched, CUDA graphs (functional)

- **Continuous batching** (`engine/engine.py`): each step is one batched
  decode for every running sequence. Finished ones leave; waiting ones join
  via prefill.
  - `tests/test_engine.py` (fp32, 2 slots, 4 requests, including staggered
    arrivals that wait for a slot): every request's tokens identical to the
    prompt generated alone.
- **Server** (`server.py`): OpenAI-compatible `/v1/completions`, SSE
  streaming with one chunk per token and a final usage chunk, the format
  `vllm bench serve` reads (checked in its source).
- **CUDA graphs** (`engine/graphs.py`): the decode step recorded once per
  batch bucket (1-64) and replayed. It needed compact slots (sequence b in
  cache slot b, survivors shifted down when others finish) and attention
  over the full cache length, masked.
  - The same engine tests pass on both paths (eager and graphs).
  - Also replaced `enable_gqa` with a masked decode: that combination made
    PyTorch fall back to copying keys and values to all 16 heads. Instead,
    query heads are regrouped as [8 kv groups x 2].
- **Trial runs, not controlled** (random 256/256, bf16):

  | | c=1 tok/s | c=1 TPOT | c=16 tok/s |
  |---|---|---|---|
  | eager | 29 | 34.2 ms | 391 |
  | CUDA graphs | 134 | 7.1 ms | 1,254 |

## 2026-09-27: this engine vs vLLM, pre-registration (written before the run)

**Setup:**
- Qwen3-0.6B bf16, RTX 3090 (WSL2), same session.
- Client: `vllm bench serve`, random 256 in / 256 out, EOS ignored,
  greedy; c=1/4/16/64 with 16/32/64/128 prompts, a fresh seed per level.
- Ours: `server.py`, 64 slots x 1,024 positions.
- vLLM 0.30.0 Docker (the Inference Lab's `docker-vllm.sh`): its default V2
  runner with `VLLM_WSL2_ENABLE_PIN_MEMORY=1` (the WSL2 fix), prefix caching
  off like ours, 11.6 GB KV.
- Arms: ours-graphs, vllm, ours-graphs, vllm, ours-eager. Windows GPU
  monitor and disk watchdog on. Runner: `scripts/compare.sh`.

**Predictions:**
- **E1:** graphs vs eager (ours): 3-6x faster per token at c=1.
- **E2:** vLLM vs ours-graphs at c=1: vLLM 1.5-3x faster per token. Its
  kernels are fused and compiled, and it has no per-token Python
  detokenize-and-send loop like ours.
- **E3:** at c=16 and c=64: vLLM 2-5x more throughput. It batches prefills
  and chunks them into decode steps; ours prefills requests one at a time
  and stalls decode meanwhile, so TTFT suffers most.
- **E4:** ours-graphs throughput scales ≥ 8x from c=1 to c=16.

## 2026-09-27: result, vLLM is 2.4-3.5x faster, and all four predictions held (20:53-21:09 UTC)

Raw: `results/compare/<arm>/c<N>.json`, timeline `logs/timeline.txt`. Zero
failed requests in every arm.

| Output tok/s (median TPOT, median TTFT) | c=1 | c=4 | c=16 | c=64 |
|---|---|---|---|---|
| ours, graphs, run 1 | 136 (7.14 ms, 37 ms) | 460 (8.10) | 1,220 (10.79, 554) | 1,936 (23.39, 2,345) |
| ours, graphs, run 2 | 136 (7.19 ms, 37 ms) | 466 (8.02) | 1,251 (10.41, 556) | 1,973 (23.45, 2,300) |
| vLLM, run 1 | 339 (2.94 ms, 18 ms) | 1,202 (3.20) | 3,355 (4.38, 92) | 6,853 (8.38, 184) |
| vLLM, run 2 | 354 (2.76 ms, 18 ms) | 1,188 (3.19) | 2,980 (4.80, 83) | 6,157 (9.01, 273) |
| ours, eager | 35 (28.55 ms, 38 ms) | 126 (30.64) | 459 (32.19, 612) | 1,345 (38.69, 2,360) |

**Verdicts, all confirmed:**
- **E1:** graphs vs eager at c=1: 28.55 → 7.14-7.19 ms per token (**4.0x**;
  3-6x predicted). At c=64 the gain shrinks to 1.45x: bigger batches do
  more work per launch, and graphs attend over the full 1,024 positions.
- **E2:** vLLM at c=1: 2.76-2.94 vs 7.14-7.19 ms (**2.4-2.6x**; 1.5-3x
  predicted).
- **E3:** throughput, vLLM **2.4-2.7x at c=16, 3.1-3.5x at c=64** (2-5x
  predicted). The widest gap is time to first token at c=64: 2,300-2,345 vs
  184-273 ms (8-13x), because ours prefills arrivals one by one while
  everyone else's decode waits.
- **E4:** ours-graphs c=1 → c=16: 136 → 1,220-1,251 tok/s (**9.0-9.2x**;
  ≥ 8x predicted).

**Caveats:**
- The Windows GPU monitor shows Chrome on the GPU throughout (mean 5% of
  the 3D engine, max 25%, 13 samples above 10%) and the compositor (mean
  4.5%).
- The replicates agree within 3% (ours) and 11% (vLLM at c=16). Every
  conclusion above holds across both replicates.
- Run note: during vllm-2, `server.py`'s default `--model` changed from the
  D: path to `~/models/Qwen3-0.6B`, made a symlink to the same directory
  before graphs-3 started. Same files, same weights.

**Where vLLM's 2.4x at batch 1 comes from** (not measured here, the
candidates): fused kernels (norms, rotary and activations are several
PyTorch kernels each here), torch.compile, and no per-token Python work.
This server decodes the text and sends one HTTP chunk per token on the same
process as the engine loop. The Inference Lab's profiles would tell;
unexamined.
