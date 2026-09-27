"""The KV cache: a pool of fixed-size slots, one slot per running sequence.

EXPLAINER
For each of the 28 layers the pool holds two tensors, keys and values, shaped
[num_slots, max_len, kv_heads, head_dim]. A running sequence owns one slot
(one row of every tensor) and writes the key and value of its token at
position p into row [slot, p]. When the sequence finishes, the slot is freed
for the next request.

Memory, for Qwen3-0.6B in bf16: 2 (k and v) x 28 layers x 8 heads x 128 x 2
bytes = 114,688 bytes per token. 64 slots x 1,024 positions is 65,536 tokens,
7.5 GB. This is the simple "contiguous slot" layout. vLLM's PagedAttention
instead hands out small blocks on demand, so a short request doesn't reserve
max_len rows it never uses. That's the next thing a real engine would add
(and why vLLM fits more concurrent requests in the same memory).
"""

import torch


class KVPool:
    def __init__(self, cfg, num_slots: int, max_len: int, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        shape = (num_slots, max_len, cfg.num_key_value_heads, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.num_hidden_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.num_hidden_layers)]
        self.num_slots, self.max_len = num_slots, max_len
        self._free = list(range(num_slots - 1, -1, -1))

    def alloc(self) -> int:
        if not self._free:
            raise RuntimeError("no free cache slot")
        return self._free.pop()

    def release(self, slot: int) -> None:
        self._free.append(slot)

    @property
    def free_slots(self) -> int:
        return len(self._free)


class SeqCache:
    """One sequence's slot in a pool: what Qwen3.forward(..., cache=) takes."""

    def __init__(self, pool: KVPool, slot: int):
        self.pool, self.slot = pool, slot
