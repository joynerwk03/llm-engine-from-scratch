# llm-engine-from-scratch

Working notes for Claude Code sessions in this repo. `README.md` is the
overview; `LOG.md` is the dated record of every build step, prediction and
measurement.

## Rules

1. **Describe it accurately.** Claude Code built this engine at William's
   direction ("How this was built" in the README).
   - "Minimal" and "from scratch" (no transformers model code) are accurate.
   - "Competitive with vLLM" is not, unless a measurement says so.
   - For new work, ask William whether he wants it built or guided. Guided
     means skeletons with EXPLAINER blocks and per-section tests, with him
     filling in the marked sections.
2. **Correctness before speed.** Every change passes `tests/`, which compare
   against Hugging Face transformers on the same weights: the parts, the
   whole model, greedy generation, and batching on both decode paths.
3. **Measure carefully.**
   - Write predictions in `LOG.md` before running.
   - Compare within one session, with arms alternated.
   - Keep animated windows off the GPU and log other GPU users.
   - Never quote a number without its setup: engine versions, model,
     precision, GPU, concurrency, prompts.

## Layout

| Path | What |
|---|---|
| `engine/model.py` | Qwen3: config, weights, the forward pass, batched decode |
| `engine/cache.py` | the KV cache (a pool of per-sequence slots) |
| `engine/generate.py` | the single-sequence decode loop, sampling |
| `engine/engine.py` | the continuous-batching scheduler (`add_request` / `step`) |
| `engine/graphs.py` | CUDA graphs for the decode step |
| `server.py` | OpenAI-compatible HTTP + streaming (SSE) |
| `tests/test_parts.py` | each building block vs its Hugging Face counterpart |
| `tests/test_reference.py` | the whole model vs Hugging Face (logits, greedy generation) |
| `tests/test_engine.py` | batching must not change any request's output (eager and graphs) |
| `scripts/compare.sh` | the benchmark against vLLM (needs `vllm bench serve` and a vLLM server script via `LAB`) |

## Commands

```bash
.venv/bin/python -m pytest tests -q -s                              # everything vs Hugging Face
.venv/bin/python -m pytest tests/test_parts.py -q -s -k rms_norm    # one piece at a time
.venv/bin/python server.py --cuda-graphs --port 8001                # serve
```

Weights: `$MODEL_DIR` (default `~/models/Qwen3-0.6B`). Developed on an RTX
3090 under WSL2.
