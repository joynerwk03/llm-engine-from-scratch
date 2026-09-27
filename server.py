"""An OpenAI-compatible HTTP server around the engine.

    .venv/bin/python server.py --model ~/models/Qwen3-0.6B --port 8001 --cuda-graphs

Endpoints: POST /v1/completions (streaming or not), GET /v1/models, GET /health.
Accepts the fields `vllm bench serve` sends (prompt, max_tokens, temperature,
stream, stream_options.include_usage, ignore_eos) and ignores the rest.

EXPLAINER: how the pieces talk
The engine is a plain loop (add_request, step) that must run continuously on
one thread, because every step serves every running request at once. HTTP
requests arrive on the web server's event loop (asyncio). So:
  - each HTTP request tokenizes its prompt, gets an asyncio.Queue, and hands
    (request, queue) to the engine thread through a thread-safe queue.Queue;
  - the engine thread admits new requests between steps, runs step(), and
    pushes each produced token into its request's asyncio.Queue
    (call_soon_threadsafe, the one safe way to touch an event loop from
    another thread);
  - the HTTP handler turns tokens into text and streams them to the client
    as server-sent events (SSE): lines of `data: {json}`, then `data: [DONE]`.
Streaming is why a chatbot's words appear one by one: each token is sent the
moment its step finishes, not when the whole answer is done.
"""

import argparse
import asyncio
import json
import os
import queue
import threading
import time
import traceback
import uuid

import torch
import uvicorn
from fastapi import FastAPI, Request as HTTPRequest
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer

from engine.engine import Engine


class EngineRunner:
    """Owns the engine and runs its loop on a background thread."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.inbox: queue.Queue = queue.Queue()
        self.outboxes: dict = {}                    # request id -> (asyncio.Queue, event loop)
        threading.Thread(target=self._loop, daemon=True, name="engine").start()

    def submit(self, rid, prompt_ids, max_tokens, temperature, ignore_eos, out_queue, loop):
        self.inbox.put((rid, prompt_ids, max_tokens, temperature, ignore_eos, out_queue, loop))

    def _admit(self, item):
        rid, prompt_ids, max_tokens, temperature, ignore_eos, q, loop = item
        try:
            self.engine.add_request(rid, prompt_ids, max_tokens, temperature, ignore_eos)
            self.outboxes[rid] = (q, loop)
        except ValueError as e:
            loop.call_soon_threadsafe(q.put_nowait, ("error", str(e)))

    def _loop(self):
        while True:
            if not self.engine.has_work():
                self._admit(self.inbox.get())      # idle: sleep until a request arrives
            while True:                            # take everything else that has arrived
                try:
                    self._admit(self.inbox.get_nowait())
                except queue.Empty:
                    break
            try:
                events = self.engine.step()
            except Exception:
                traceback.print_exc()
                continue
            for rid, token, reason in events:
                q, loop = self.outboxes[rid]
                loop.call_soon_threadsafe(q.put_nowait, (token, reason))
                if reason:
                    del self.outboxes[rid]


def build_app(runner: EngineRunner, tokenizer, served_name: str) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": served_name, "object": "model", "owned_by": "llm-engine-from-scratch"}]}

    @app.post("/v1/completions")
    async def completions(http_request: HTTPRequest):
        body = await http_request.json()
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):             # a list of token ids, or a batch of one string
            prompt = prompt[0] if prompt and isinstance(prompt[0], (str, list)) else prompt
        prompt_ids = prompt if isinstance(prompt, list) else tokenizer(prompt, add_special_tokens=False)["input_ids"]
        max_tokens = int(body.get("max_tokens") or 16)
        temperature = float(body.get("temperature", 1.0))
        stream = bool(body.get("stream", False))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage", False))
        rid, created = f"cmpl-{uuid.uuid4().hex}", int(time.time())
        q: asyncio.Queue = asyncio.Queue()
        runner.submit(rid, prompt_ids, max_tokens, temperature, bool(body.get("ignore_eos", False)),
                      q, asyncio.get_running_loop())

        first = await q.get()
        if first[0] == "error":
            return JSONResponse({"error": {"message": first[1], "type": "invalid_request_error"}}, status_code=400)

        async def tokens():
            item = first
            while True:
                yield item
                if item[1]:                      # finish_reason set: done
                    return
                item = await q.get()

        def chunk(text, reason):
            return {"id": rid, "object": "text_completion", "created": created, "model": served_name,
                    "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": reason}]}

        if stream:
            async def sse():
                ids, shown = [], ""
                async for token, reason in tokens():
                    ids.append(token)
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    delta = "" if text.endswith("�") else text[len(shown):]   # hold back a split character
                    shown += delta
                    yield f"data: {json.dumps(chunk(delta, reason))}\n\n"          # one chunk per token
                if include_usage:
                    usage = {"prompt_tokens": len(prompt_ids), "completion_tokens": len(ids),
                             "total_tokens": len(prompt_ids) + len(ids)}
                    yield f"data: {json.dumps({**chunk('', None), 'choices': [], 'usage': usage})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        ids, reason = [], None
        async for token, reason in tokens():
            ids.append(token)
        body = chunk(tokenizer.decode(ids, skip_special_tokens=True), reason)
        body["usage"] = {"prompt_tokens": len(prompt_ids), "completion_tokens": len(ids),
                         "total_tokens": len(prompt_ids) + len(ids)}
        return JSONResponse(body)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL_DIR", os.path.expanduser("~/models/Qwen3-0.6B")))
    ap.add_argument("--served-name", default="qwen3-0.6b")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--max-slots", type=int, default=64, help="max concurrent sequences (KV cache slots)")
    ap.add_argument("--max-len", type=int, default=1024, help="max prompt + output tokens per sequence")
    ap.add_argument("--cuda-graphs", action="store_true", help="record the decode step as CUDA graphs")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()
    engine = Engine(args.model, args.max_slots, args.max_len, dtype=getattr(torch, args.dtype), cuda_graphs=args.cuda_graphs)
    app = build_app(EngineRunner(engine), AutoTokenizer.from_pretrained(args.model), args.served_name)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
