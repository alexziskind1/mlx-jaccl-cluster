#!/usr/bin/env python3
import os
import time
import json
import socket
import struct
import threading
import asyncio
import uuid
from typing import Optional, Union, AsyncGenerator

import mlx.core as mx
import sys
from pathlib import Path
from mlx_lm.utils import sharded_load, load_model

# generate() import differs across mlx-lm branches
try:
    from mlx_lm.utils import generate
except Exception:
    from mlx_lm.generate import generate

# stream_generate for SSE streaming
try:
    from mlx_lm.utils import stream_generate
except ImportError:
    try:
        from mlx_lm.generate import stream_generate
    except ImportError:
        stream_generate = None

from fastapi import FastAPI, HTTPException
from starlette.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn


# -------------------------
# Custom tokenizer support
# -------------------------
class TokenizerWrapper:
    """Wrapper to handle encode kwargs that some custom tokenizers don't support."""
    def __init__(self, tokenizer):
        self._tok = tokenizer

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def encode(self, text, **kwargs):
        return self._tok.encode(text)

    def decode(self, tokens, **kwargs):
        return self._tok.decode(tokens)


def load_custom_tokenizer(model_path):
    """Load custom tokenizer directly when AutoTokenizer fails."""
    model_path = Path(model_path)
    sys.path.insert(0, str(model_path))

    for tok_file in model_path.glob("tokenization_*.py"):
        module_name = tok_file.stem
        mod = __import__(module_name)
        for attr in dir(mod):
            cls = getattr(mod, attr)
            if isinstance(cls, type) and hasattr(cls, 'from_pretrained'):
                try:
                    tok = cls.from_pretrained(model_path)
                    return TokenizerWrapper(tok)
                except:
                    continue
    raise RuntimeError(f"Could not load custom tokenizer from {model_path}")


def sharded_load_with_fallback(repo):
    """Load model with fallback for custom tokenizers."""
    model_path = Path(repo)

    try:
        return sharded_load(repo)
    except Exception as e:
        if "tokenizer" not in str(e).lower() and "NoneType" not in str(e):
            raise

    # Fallback: load model and tokenizer separately
    tok = load_custom_tokenizer(model_path)
    model, config = load_model(model_path, lazy=True, strict=False)

    tensor_group = mx.distributed.init()
    if hasattr(model, "shard"):
        model.shard(tensor_group)

    mx.eval(model.parameters())

    x = mx.zeros((1,))
    mx.eval(mx.distributed.all_sum(x))

    return model, tok


# -------------------------
# Configuration (env vars)
# -------------------------
MODEL_DIR = os.environ["MODEL_DIR"]  # REQUIRED
MODEL_ID = os.environ.get("MODEL_ID", os.path.basename(MODEL_DIR.rstrip("/")))

HOST = os.environ.get("HOST", "0.0.0.0")      # HTTP bind on rank0
PORT = int(os.environ.get("PORT", "8080"))    # HTTP port on rank0

# Control-plane (rank0 <-> workers) for coordinating "everyone call generate()"
CTRL_PORT = int(os.environ.get("CTRL_PORT", "18080"))

def _default_ctrl_host() -> str:
    c = os.environ.get("MLX_JACCL_COORDINATOR", "")
    if ":" in c:
        return c.split(":", 1)[0]
    return "macstudio1.local"

CTRL_HOST = os.environ.get("CTRL_HOST", _default_ctrl_host())

DEFAULT_MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))

# Backpressure / queueing
QUEUE_MAX = int(os.environ.get("QUEUE_MAX", "8"))          # max queued requests
REQ_TIMEOUT = float(os.environ.get("REQ_TIMEOUT", "120"))  # per request timeout (seconds)

# -------------------------
# Globals
# -------------------------
app = FastAPI()
_model = None
_tok = None
_world = None

_queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX)  # rank0 only uses it

# -------------------------
# Tiny framed JSON protocol
# -------------------------
def _recvall(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

def send_msg(sock: socket.socket, obj: dict) -> None:
    data = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack("!I", len(data)))
    sock.sendall(data)

def recv_msg(sock: socket.socket) -> Optional[dict]:
    hdr = _recvall(sock, 4)
    if hdr is None:
        return None
    (n,) = struct.unpack("!I", hdr)
    body = _recvall(sock, n)
    if body is None:
        return None
    return json.loads(body.decode("utf-8"))

# -------------------------
# OpenAI-ish schemas
# -------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionsReq(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

class CompletionsReq(BaseModel):
    model: Optional[str] = None
    prompt: Union[str, list[str]]
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

def _build_chat_prompt(messages: list[ChatMessage]) -> str:
    # Prefer tokenizer chat template when available
    if hasattr(_tok, "apply_chat_template"):
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        return _tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # Fallback: simple "ROLE: content" format
    parts = [f"{m.role.upper()}: {m.content}" for m in messages]
    parts.append("ASSISTANT:")
    return "\n".join(parts)

def _tok_len(text: str) -> int:
    return len(_tok.encode(text))

# -------------------------
# Rank0 worker connections
# -------------------------
_worker_socks: dict[int, socket.socket] = {}  # rank -> socket
_worker_lock = threading.Lock()

def rank0_accept_workers(expected_world_size: int) -> None:
    """
    Rank0 listens for worker control-plane connections.
    Each worker sends {"type":"hello","rank":N}.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, CTRL_PORT))
    srv.listen(16)
    print(f"[rank0] control-plane listening on {HOST}:{CTRL_PORT}", flush=True)

    while True:
        conn, addr = srv.accept()
        hello = recv_msg(conn)
        if not hello or hello.get("type") != "hello" or "rank" not in hello:
            conn.close()
            continue
        r = int(hello["rank"])
        with _worker_lock:
            _worker_socks[r] = conn
        print(f"[rank0] worker connected rank={r} from {addr}", flush=True)

def rank0_wait_for_workers(expected_world_size: int, timeout_s: int = 60) -> bool:
    t0 = time.time()
    while True:
        with _worker_lock:
            ok = all(r in _worker_socks for r in range(1, expected_world_size))
        if ok:
            print("[rank0] all workers connected", flush=True)
            return True
        if time.time() - t0 > timeout_s:
            return False
        time.sleep(0.1)

def rank0_broadcast_task(task: dict) -> None:
    """
    Send the same task to all worker ranks (1..N-1).
    """
    with _worker_lock:
        items = list(_worker_socks.items())
    for r, s in items:
        send_msg(s, {"type": "task", **task})

def rank0_wait_done(expected_world_size: int) -> None:
    """
    Wait for {"type":"done"} from all workers.
    """
    done: set[int] = set()
    while len(done) < (expected_world_size - 1):
        with _worker_lock:
            items = list(_worker_socks.items())
        for r, s in items:
            if r in done:
                continue
            s.settimeout(0.2)
            try:
                msg = recv_msg(s)
            except Exception:
                msg = None
            if msg and msg.get("type") == "done":
                done.add(r)

# -------------------------
# Worker loop
# -------------------------
def worker_loop(rank: int) -> None:
    """
    Workers connect to rank0 control-plane, block waiting for tasks.
    For each task: call generate() (so collectives match rank0), then send done.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((CTRL_HOST, CTRL_PORT))
    send_msg(s, {"type": "hello", "rank": rank})
    print(f"[worker {rank}] connected to control-plane {CTRL_HOST}:{CTRL_PORT}", flush=True)

    while True:
        msg = recv_msg(s)
        if not msg:
            continue
        if msg.get("type") != "task":
            continue

        prompt = msg["prompt"]
        max_tokens = int(msg["max_tokens"])

        _ = generate(_model, _tok, prompt, max_tokens=max_tokens)
        mx.eval()
        send_msg(s, {"type": "done", "rank": rank})

# -------------------------
# Queue worker (rank0 only)
# -------------------------
def _run_stream_blocking(loop: asyncio.AbstractEventLoop, kind: str, prompt: str, max_t: int,
                          chunk_queue: asyncio.Queue) -> None:
    """
    Runs on a worker thread: broadcasts the task, drives stream_generate(), and
    pushes SSE chunks back onto the event loop thread-safely. Kept off the event
    loop so /health, /queue, and REQ_TIMEOUT keep working while a request runs.
    """
    def _put(chunk: str) -> None:
        loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)

    rank0_broadcast_task({"prompt": prompt, "max_tokens": max_t})

    req_id = f"chatcmpl-{uuid.uuid4().hex[:24]}" if kind == "chat" else f"cmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    for response in stream_generate(_model, _tok, prompt, max_tokens=max_t):
        token_text = response.text  # GenerationResponse.text contains the decoded text
        if kind == "chat":
            chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [{
                    "index": 0,
                    "delta": {"content": token_text},
                    "finish_reason": None,
                }],
            }
        else:  # completions
            chunk = {
                "id": req_id,
                "object": "text_completion",
                "created": created,
                "model": MODEL_ID,
                "choices": [{
                    "index": 0,
                    "text": token_text,
                    "finish_reason": None,
                    "logprobs": None,
                }],
            }
        _put(f"data: {json.dumps(chunk)}\n\n")

    mx.eval()

    # Send final chunk with finish_reason
    if kind == "chat":
        final_chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }
    else:
        final_chunk = {
            "id": req_id,
            "object": "text_completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "text": "",
                "finish_reason": "stop",
                "logprobs": None,
            }],
        }
    _put(f"data: {json.dumps(final_chunk)}\n\n")
    _put("data: [DONE]\n\n")
    _put(None)  # Signal end of stream

    rank0_wait_done(_world.size())


def _run_once_blocking(kind: str, prompt: str, max_t: int) -> dict:
    """Runs on a worker thread: broadcasts the task, calls generate(), and builds the response."""
    rank0_broadcast_task({"prompt": prompt, "max_tokens": max_t})

    t0 = time.time()
    out_text = generate(_model, _tok, prompt, max_tokens=max_t)
    mx.eval()
    t1 = time.time()

    rank0_wait_done(_world.size())

    completion = out_text[len(prompt):] if out_text.startswith(prompt) else out_text
    pt = _tok_len(prompt)
    ct = _tok_len(completion)

    timing = {
        "seconds": round(t1 - t0, 3),
        "tokens_per_sec": round(ct / max(t1 - t0, 1e-9), 3),
    }

    if kind == "chat":
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": completion},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
            "timing": timing,
        }
    elif kind == "completions":
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "text": completion,
                "finish_reason": "stop",
                "logprobs": None,
            }],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
            "timing": timing,
        }
    else:
        raise RuntimeError(f"Unknown request kind: {kind}")


async def _queue_worker() -> None:
    """
    Processes queued requests sequentially.
    Each request triggers:
      - broadcast task to workers
      - rank0 generate() or stream_generate()
      - wait for worker completion
      - fulfill per-request future with an OpenAI-shaped response (or stream chunks)

    The actual generation runs on an executor thread (via loop.run_in_executor) so
    this coroutine never blocks the event loop: /health, /queue, and new incoming
    requests keep being served, and REQ_TIMEOUT can still fire while a request runs.
    """
    loop = asyncio.get_running_loop()
    while True:
        item = await _queue.get()
        if item is None:
            _queue.task_done()
            continue

        kind, prompt, max_t, result_target, is_stream = item  # kind: "chat" | "completions"
        try:
            if is_stream and stream_generate is not None:
                chunk_queue: asyncio.Queue = result_target
                await loop.run_in_executor(
                    None, _run_stream_blocking, loop, kind, prompt, max_t, chunk_queue
                )
            else:
                fut: asyncio.Future = result_target
                resp = await loop.run_in_executor(None, _run_once_blocking, kind, prompt, max_t)
                fut.set_result(resp)

        except Exception as e:
            if is_stream:
                chunk_queue: asyncio.Queue = result_target
                await chunk_queue.put(f"data: {json.dumps({'error': str(e)})}\n\n")
                await chunk_queue.put("data: [DONE]\n\n")
                await chunk_queue.put(None)
            else:
                result_target.set_exception(e)
        finally:
            _queue.task_done()

@app.on_event("startup")
async def _startup() -> None:
    # Only rank0 runs the HTTP server, so only rank0 starts the queue worker
    if _world and _world.rank() == 0:
        asyncio.create_task(_queue_worker())

# -------------------------
# HTTP endpoints (rank0 only)
# -------------------------
@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "world_size": _world.size(),
        "rank": _world.rank(),
        "model": MODEL_ID,
        "queue_max": QUEUE_MAX,
        "queue_size": _queue.qsize(),
    }

@app.get("/v1/models")
def list_models() -> dict:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}

@app.get("/queue")
def queue_status() -> dict:
    return {"size": _queue.qsize(), "max": QUEUE_MAX}

async def _stream_generator(chunk_queue: asyncio.Queue) -> AsyncGenerator[str, None]:
    """Yield SSE chunks from the queue until None is received."""
    while True:
        chunk = await chunk_queue.get()
        if chunk is None:
            break
        yield chunk


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionsReq):
    if req.stream and stream_generate is None:
        raise HTTPException(status_code=400, detail="stream=true not supported (stream_generate not available)")
    if req.model and req.model != MODEL_ID:
        raise HTTPException(status_code=400, detail=f"Only model '{MODEL_ID}' is served")

    if _world.rank() != 0:
        raise HTTPException(status_code=500, detail="Rank != 0 received HTTP request")

    prompt = _build_chat_prompt(req.messages)
    max_t = req.max_tokens or DEFAULT_MAX_TOKENS

    if req.stream:
        # Streaming mode: return SSE response
        chunk_queue: asyncio.Queue = asyncio.Queue()
        try:
            _queue.put_nowait(("chat", prompt, max_t, chunk_queue, True))
        except asyncio.QueueFull:
            raise HTTPException(status_code=429, detail="Server busy (queue full). Try again later.")

        return StreamingResponse(
            _stream_generator(chunk_queue),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Non-streaming mode: return JSON response
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()

        try:
            _queue.put_nowait(("chat", prompt, max_t, fut, False))
        except asyncio.QueueFull:
            raise HTTPException(status_code=429, detail="Server busy (queue full). Try again later.")

        try:
            return await asyncio.wait_for(fut, timeout=REQ_TIMEOUT)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timed out")

@app.post("/v1/completions")
async def completions(req: CompletionsReq):
    if req.stream and stream_generate is None:
        raise HTTPException(status_code=400, detail="stream=true not supported (stream_generate not available)")
    if req.model and req.model != MODEL_ID:
        raise HTTPException(status_code=400, detail=f"Only model '{MODEL_ID}' is served")

    if _world.rank() != 0:
        raise HTTPException(status_code=500, detail="Rank != 0 received HTTP request")

    if isinstance(req.prompt, list):
        # Keep it simple + safe for distributed mode: one prompt at a time.
        if len(req.prompt) != 1:
            raise HTTPException(status_code=400, detail="Only a single prompt string is supported (prompt must be a string, or a list of length 1).")
        prompt = req.prompt[0]
    else:
        prompt = req.prompt

    max_t = req.max_tokens or DEFAULT_MAX_TOKENS

    if req.stream:
        # Streaming mode: return SSE response
        chunk_queue: asyncio.Queue = asyncio.Queue()
        try:
            _queue.put_nowait(("completions", prompt, max_t, chunk_queue, True))
        except asyncio.QueueFull:
            raise HTTPException(status_code=429, detail="Server busy (queue full). Try again later.")

        return StreamingResponse(
            _stream_generator(chunk_queue),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Non-streaming mode: return JSON response
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()

        try:
            _queue.put_nowait(("completions", prompt, max_t, fut, False))
        except asyncio.QueueFull:
            raise HTTPException(status_code=429, detail="Server busy (queue full). Try again later.")

        try:
            return await asyncio.wait_for(fut, timeout=REQ_TIMEOUT)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timed out")

# -------------------------
# Main
# -------------------------
def main() -> None:
    global _model, _tok, _world
    _world = mx.distributed.init()
    _model, _tok = sharded_load_with_fallback(MODEL_DIR)

    if _world.rank() == 0:
        th = threading.Thread(target=rank0_accept_workers, args=(_world.size(),), daemon=True)
        th.start()

        if not rank0_wait_for_workers(_world.size(), timeout_s=60):
            raise RuntimeError("Workers did not connect to control-plane in time")

        uvicorn.run(app, host=HOST, port=PORT, log_level="info")
    else:
        worker_loop(_world.rank())

if __name__ == "__main__":
    main()

