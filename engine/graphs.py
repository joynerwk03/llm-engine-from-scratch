"""CUDA graphs for the decode step.

EXPLAINER
The GPU doesn't run Python. For every small operation in a decode step (a
matrix multiply, a norm, an add) the CPU tells the GPU "run this kernel": a
launch. One decode step of this model is about two thousand launches, and on
this machine (Python, WSL2) each costs roughly as much time as the GPU needs
to run it. So the GPU spends most of each step waiting for the CPU to hand it
the next piece of work. (The Inference Lab found the same problem inside
vLLM's speculative decoding: 358 launches per step, GPU idle half the time.)

A CUDA graph fixes this by recording. Run the step once while the GPU
records every launch, with every tensor's memory address; afterwards,
graph.replay() re-runs all two thousand kernels with a single call. The CPU
cost of a step drops to one launch, and the GPU runs back to back.

The catch: a replay repeats exactly what was recorded, the same shapes and
the same memory. So:
  - inputs go into fixed buffers (static_tokens, static_positions), copied in
    before each replay, and the output appears in a fixed buffer;
  - the batch size must match, so we record one graph per bucket
    (1, 2, 4, 8, 16, 32, 64) and pad a batch of 5 up to 8 (the padding rows
    write into unused cache slots, which a new request overwrites anyway);
  - attention covers the full cache length every step, masked, because the
    real length changes every step. vLLM does the same: it records graphs
    for a list of batch sizes at startup (that's the "Capturing CUDA graphs"
    it logs) and pads to the next one.
"""

import torch


class CudaGraphDecoder:
    def __init__(self, model, pool, buckets=(1, 2, 4, 8, 16, 32, 64)):
        self.model, self.pool = model, pool
        self.buckets = [b for b in buckets if b <= pool.num_slots]
        device, top = model.device, self.buckets[-1]
        self.static_tokens = torch.zeros(top, dtype=torch.long, device=device)
        self.static_positions = torch.zeros(top, dtype=torch.long, device=device)
        self.graphs, self.outputs = {}, {}
        mempool = torch.cuda.graph_pool_handle()
        for b in reversed(self.buckets):           # largest first, so smaller ones reuse its memory
            args = (self.static_tokens[:b], self.static_positions[:b], pool, pool.max_len)
            side = torch.cuda.Stream()             # warm up off the main stream, as PyTorch recommends
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    model.decode(*args)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=mempool):
                self.outputs[b] = model.decode(*args)
            self.graphs[b] = graph

    def __call__(self, tokens: list, positions: list) -> torch.Tensor:
        """Same contract as model.decode for sequences in slots 0..B-1."""
        B = len(tokens)
        b = next(size for size in self.buckets if size >= B)
        self.static_tokens[:B] = torch.tensor(tokens)
        self.static_positions[:B] = torch.tensor(positions)
        if b > B:
            self.static_tokens[B:b] = 0
            self.static_positions[B:b] = 0
        self.graphs[b].replay()
        return self.outputs[b][:B]
