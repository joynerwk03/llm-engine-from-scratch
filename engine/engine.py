"""The engine: a request queue and a continuous-batching scheduler.

EXPLAINER
A server receives requests at random times, each wanting a different number of
tokens. The simplest engine serves one at a time: every other user waits.
Static batching groups requests and runs them together, but the whole group
waits for its longest member, and newcomers wait for the group.

Continuous batching (the idea behind vLLM and SGLang's throughput) works one
step at a time instead of one request at a time. Every step:
  1. one batched decode: every running sequence gets its next token in a
     single forward pass (Qwen3.decode_batch);
  2. finished sequences leave immediately, freeing their cache slots;
  3. waiting requests join as soon as a slot is free: their prompt is
     processed (prefill) and they're decoding from the next step on.
So the batch changes shape every step, nobody waits for anyone else's
long answer, and the GPU's weight reads are shared by everyone running.

This scheduler is deliberately simple: a new request's prefill runs on its
own, between decode steps. (vLLM also splits long prompts into chunks and
mixes them into decode steps, so one long prompt can't stall everyone.)

Compact slots: running sequence b always lives in cache slot b, so a decode
step reads the first B rows of the cache as one view (and CUDA graphs can
record it; see engine/graphs.py). When sequences finish, the survivors are
shifted down to fill the gaps, copying their cached keys and values: about
115 KB per cached token, well under a millisecond per move.
"""

import json
from collections import deque
from dataclasses import dataclass, field

import torch

from engine.cache import KVPool, SeqCache
from engine.graphs import CudaGraphDecoder
from engine.model import Qwen3


@dataclass
class Request:
    id: str
    prompt_ids: list
    max_tokens: int
    temperature: float = 0.0
    ignore_eos: bool = False
    slot: int = -1
    next_pos: int = 0            # position the next fed token will occupy
    last_token: int = -1
    output_ids: list = field(default_factory=list)
    finish_reason: str | None = None


class Engine:
    def __init__(self, model_dir: str, max_slots: int = 64, max_len: int = 1024,
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16, seed: int = 0,
                 cuda_graphs: bool = False):
        self.model = Qwen3(model_dir, device, dtype)
        self.pool = KVPool(self.model.cfg, max_slots, max_len, device, dtype)
        self.max_len, self.device = max_len, device
        self.graphs = CudaGraphDecoder(self.model, self.pool) if cuda_graphs else None
        with open(f"{model_dir}/generation_config.json") as f:
            eos = json.load(f)["eos_token_id"]
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.generator = torch.Generator(device).manual_seed(seed)

    def add_request(self, req_id: str, prompt_ids: list, max_tokens: int,
                    temperature: float = 0.0, ignore_eos: bool = False) -> None:
        if not prompt_ids:
            raise ValueError("empty prompt")
        if len(prompt_ids) + max_tokens > self.max_len:
            raise ValueError(f"prompt ({len(prompt_ids)} tokens) + max_tokens ({max_tokens}) "
                             f"exceeds this server's max length ({self.max_len})")
        self.waiting.append(Request(req_id, list(prompt_ids), max_tokens, temperature, ignore_eos))

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def step(self) -> list[tuple[str, int, str | None]]:
        """One engine iteration. Returns (request_id, token_id, finish_reason or None)
        for every token produced this step, in the order produced."""
        events = []
        if self.running:                                   # 1. one batched decode for everyone running
            reqs = self.running                            # reqs[b] is in cache slot b
            tokens, positions = [r.last_token for r in reqs], [r.next_pos for r in reqs]
            if self.graphs is not None:
                logits = self.graphs(tokens, positions)
            else:
                logits = self.model.decode(torch.tensor(tokens, device=self.device),
                                           torch.tensor(positions, device=self.device),
                                           self.pool, attn_len=max(positions) + 1)
            for r, t in zip(reqs, self._sample(logits, [r.temperature for r in reqs])):
                r.next_pos += 1
                self._emit(r, t, events)
        self._compact()                                    # 2. finished sequences leave
        while self.waiting and len(self.running) < self.pool.num_slots:   # 3. newcomers join
            r = self.waiting.popleft()
            r.slot = len(self.running)
            ids = torch.tensor(r.prompt_ids, device=self.device)
            logits = self.model.forward(ids, 0, SeqCache(self.pool, r.slot), last_only=True)   # prefill
            r.next_pos = len(r.prompt_ids)
            self.running.append(r)
            self._emit(r, self._sample(logits, [r.temperature])[0], events)
        self._compact()                                    # (a one-token request can finish at once)
        return events

    def _compact(self) -> None:
        """Drop finished sequences and shift survivors down so running[b] is in slot b."""
        survivors = [r for r in self.running if not r.finish_reason]
        for b, r in enumerate(survivors):                  # ascending, so a destination is always free
            if r.slot != b:
                n = r.next_pos                             # positions 0..n-1 hold cached keys/values
                for i in range(len(self.pool.k)):
                    self.pool.k[i][b, :n] = self.pool.k[i][r.slot, :n]
                    self.pool.v[i][b, :n] = self.pool.v[i][r.slot, :n]
                r.slot = b
        self.running = survivors

    def _sample(self, logits: torch.Tensor, temperatures: list) -> list:
        """Greedy rows (temperature 0) take the argmax; the rest sample from
        softmax(logits / temperature). One GPU->CPU copy for the whole batch."""
        greedy = logits.argmax(-1)
        if all(t <= 0 for t in temperatures):
            return greedy.tolist()
        temps = torch.tensor([max(t, 1e-5) for t in temperatures], device=logits.device)[:, None]
        sampled = torch.multinomial(torch.softmax(logits / temps, -1), 1, generator=self.generator)[:, 0]
        is_greedy = torch.tensor([t <= 0 for t in temperatures], device=logits.device)
        return torch.where(is_greedy, greedy, sampled).tolist()

    def _emit(self, r: Request, token: int, events: list) -> None:
        r.output_ids.append(token)
        r.last_token = token
        if not r.ignore_eos and token in self.eos_ids:
            r.finish_reason = "stop"
        elif len(r.output_ids) >= r.max_tokens:
            r.finish_reason = "length"
        events.append((r.id, token, r.finish_reason))
