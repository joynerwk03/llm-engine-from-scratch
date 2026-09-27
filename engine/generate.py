"""The decode loop for a single sequence, and token sampling.

EXPLAINER
Generation is a loop around the forward pass:
  1. prefill: run the whole prompt once, filling the KV cache; the last row of
     logits scores the first new token.
  2. pick a token from those scores (greedy: the highest; sampled: at random,
     in proportion to softmax(logits / temperature)).
  3. decode: run just that token through the model (T = 1) at the next
     position; the cache supplies everything before it. Repeat from 2.
Greedy decoding is deterministic, which is why it's what we compare with
Hugging Face token for token.
"""

import torch

from engine.cache import KVPool, SeqCache


def sample(logits: torch.Tensor, temperature: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """logits: [B, vocab] float32. temperature 0 means greedy. Returns LongTensor [B]."""
    if temperature <= 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1, generator=generator)[:, 0]


@torch.inference_mode()
def greedy_generate(model, prompt_ids: list, max_new_tokens: int) -> list:
    """Generate max_new_tokens greedily (no stop on end-of-text; callers compare
    prefixes). Uses a one-slot KV cache, exactly like one request in the engine."""
    pool = KVPool(model.cfg, 1, len(prompt_ids) + max_new_tokens, model.device, model.dtype)
    cache = SeqCache(pool, pool.alloc())
    ids = torch.tensor(prompt_ids, device=model.device)
    token = sample(model.forward(ids, 0, cache, last_only=True), 0)       # prefill
    out = [token.item()]
    for pos in range(len(prompt_ids), len(prompt_ids) + max_new_tokens - 1):
        token = sample(model.forward(token, pos, cache), 0)                # decode one token
        out.append(token.item())
    return out
