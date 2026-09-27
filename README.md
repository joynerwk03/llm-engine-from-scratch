# llm-engine-from-scratch

A minimal LLM inference engine in plain PyTorch, to understand what serving
engines like [vLLM](https://github.com/vllm-project/vllm) and
[SGLang](https://github.com/sgl-project/sglang) do inside. It serves
Qwen3-0.6B over an OpenAI-compatible streaming API, with:

- **a forward pass written from scratch** (RMSNorm, rotary position
  embeddings, grouped-query attention, the gated MLP), loading the weights
  straight from the safetensors file;
- **a KV cache**, so each new token costs one small forward pass instead of
  recomputing the whole sequence;
- **continuous batching**: every step decodes one token for every running
  request at once, and requests join and leave between steps;
- **CUDA graphs** for the decode step: record it once, replay it with a single
  call instead of ~2,000 kernel launches;
- **a streaming server** that vLLM's own benchmark client can drive.

The code is commented for learning: each file opens with an explainer of the
idea it implements.

## It's correct

Checked against Hugging Face transformers on the same weights (`tests/`):

- **Logits are bit-identical in float32**, on three prompts.
- **Greedy generation through the KV cache** matches Hugging Face token for
  token (64 tokens, three prompts).
- **Batching changes nothing.** Each request's tokens are identical to that
  prompt generated alone, including requests that arrive mid-flight and wait
  for a free cache slot. Tested with and without CUDA graphs.
- Each building block (RMSNorm, RoPE, attention, MLP) also has its own test
  against the matching Hugging Face module.

## How fast, against vLLM

Same GPU (RTX 3090), same model (Qwen3-0.6B, bf16), same client (vLLM's own
`vllm bench serve`: random 256-token prompts, 256 tokens out, greedy), one
session with the runs alternated. Output tokens per second, two runs each:

| Concurrent requests | This engine | vLLM 0.30.0 | vLLM ahead by |
|---|---|---|---|
| 1 | 136, 136 | 339, 354 | 2.5x |
| 4 | 460, 466 | 1,202, 1,188 | 2.6x |
| 16 | 1,220, 1,251 | 3,355, 2,980 | 2.6x |
| 64 | 1,936, 1,973 | 6,853, 6,157 | 3.3x |

- **CUDA graphs made one user's decoding 4x faster**: 28.6 → 7.1 ms per
  token. Without them, the GPU spends most of each step waiting for Python
  to launch its next small kernel.
- **Continuous batching raised throughput 9x** from 1 to 16 users. Each
  decode step reads the model's weights once for the whole batch.
- **vLLM is 2.5-3.3x faster.** At one user its step takes 2.8-2.9 ms against
  7.1 here: fused kernels, compilation, and none of this server's per-token
  Python work.
- **The widest gap is the first token under load.** At 64 users: 2.3 s here
  vs 0.18-0.27 s in vLLM, because this engine processes new prompts one at a
  time while everyone else's decoding waits (vLLM batches and chunks them).

I wrote down each of these predictions before measuring, and all four held.
The setups, raw numbers and caveats are in `LOG.md` (another program used a
little of the GPU during the run).

## What it leaves out (and vLLM has)

- **Paged KV cache.** Each running request reserves a fixed 1,024-token slot,
  so memory, not compute, caps concurrency. vLLM's PagedAttention hands out
  small blocks on demand.
- **Batched and chunked prefill.** New prompts are processed one at a time
  between decode steps, so a burst of arrivals stalls everyone's streaming.
- **Fused kernels.** Norms, rotary embeddings and activations each run as
  several small PyTorch kernels.
- **Prefix caching, quantization, speculative decoding, multi-GPU.**

## Run it

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest tests -q                  # correctness vs Hugging Face (needs the weights)
.venv/bin/python server.py --model /path/to/Qwen3-0.6B --cuda-graphs --port 8001
curl -N http://127.0.0.1:8001/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 16, "temperature": 0, "stream": true}'
```

Weights: `huggingface-cli download Qwen/Qwen3-0.6B --local-dir /path/to/Qwen3-0.6B`.

## Layout

| Path | What |
|---|---|
| `engine/model.py` | Qwen3: config, weights, the forward pass, batched decode |
| `engine/cache.py` | the KV cache (a pool of per-sequence slots) |
| `engine/generate.py` | the single-sequence decode loop, sampling |
| `engine/engine.py` | the continuous-batching scheduler |
| `engine/graphs.py` | CUDA graphs for the decode step |
| `server.py` | OpenAI-compatible HTTP + streaming |
| `tests/` | correctness against Hugging Face |
| `scripts/compare.sh` | the benchmark against vLLM |
| `LOG.md` | what was built and measured, with predictions made before each run |

## How this was built

Claude Code (Anthropic's coding agent) wrote this code and its explainer
comments at my direction. I chose what to build, and how to check each piece
against Hugging Face and vLLM, and reviewed the results. It's a companion to
my [LLM Serving Lab](https://github.com/joynerwk03/llm-serving-lab), where I
measured vLLM and SGLang themselves.

## License

MIT
