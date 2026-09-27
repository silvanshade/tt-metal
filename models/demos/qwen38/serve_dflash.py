# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible chat server for Qwen3.8 at TP=2 with captured DFlash2 rounds.

One request runs at a time, greedily: every round verifies the anchor and seven drafts, and the
target's own greedy tokens decide the output, so the text equals plain greedy decoding. Sampling
parameters are accepted and ignored. Thinking is split into `reasoning_content` at `</think>`;
`<tool_call>` blocks in the answer become OpenAI `tool_calls`. A request whose client leaves stops
at the next round.

Environment: QWEN38_MODEL (target checkpoint dir), QWEN38_DRAFTER (DFlash2 checkpoint dir),
QWEN38_MAX_CONTEXT (default 131072), SERVED_MODEL_NAME, HOST, PORT, VLLM_API_KEY (optional).
The mesh follows MESH_DEVICE / TT_VISIBLE_DEVICES, as the TP=2 service sets them.
"""

import asyncio
import faulthandler
import json
import logging
import os
import re
import signal
import threading
import time
import uuid
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

import ttnn

TT_CONFIG = {
    "physical_device_ids": [1, 0],
    "fabric_config": "FABRIC_1D",
    "trace_region_size": int(os.environ.get("TRACE_REGION", str(3 << 29))),
    "l1_small_size": 24576,
}
PAGE = 64
log = logging.getLogger("serve_dflash")


class Engine:
    """Target, verifier, drafter and one captured round, driven by one thread at a time."""

    def __init__(self, model_dir: Path, drafter_dir: Path, max_context: int) -> None:
        from transformers import AutoConfig, AutoTokenizer
        from vllm_tt_plugin.loader import weight_downcast
        from vllm_tt_plugin.worker import open_mesh_device

        from models.demos.qwen38.tt.dflash import Qwen38DFlash
        from models.demos.qwen38.tt.dflash_round import Qwen38DFlashRound
        from models.demos.qwen38.tt.mtp import Qwen38MTPVerifier
        from models.demos.qwen38.tt.qwen38_vllm import Qwen38ForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.stops = {self.tokenizer.convert_tokens_to_ids("<|im_end|>"), self.tokenizer.eos_token_id}
        self.max_context = max_context
        self.mesh = open_mesh_device(TT_CONFIG, "all")
        os.environ["HF_MODEL"] = str(model_dir)
        hf = AutoConfig.from_pretrained(model_dir)
        text = getattr(hf, "text_config", hf)
        with weight_downcast("bfloat4_b"):
            generator = Qwen38ForCausalLM.initialize_vllm_model(hf, self.mesh, 1, max_context)
        self.model = model = generator.model[0]
        self.vocab = model.vocab_size
        blocks = max_context // PAGE
        heads = text.num_key_value_heads // self.mesh.get_num_devices()
        model.allocate_kv_caches((blocks, heads, PAGE, text.head_dim), ttnn.bfloat4_b, batch_size=1)
        self.page_table = torch.arange(blocks, dtype=torch.int32).reshape(1, blocks)
        self.drafter = Qwen38DFlash(model, drafter_dir, max_context)
        self.verifier = Qwen38MTPVerifier(model, self.drafter.block, taps=self.drafter.taps)
        self.round = Qwen38DFlashRound(model, self.verifier, self.drafter, blocks)
        self.block = self.drafter.block
        self.window = self.drafter.window
        self.composer = ttnn.ConcatMeshToTensor(self.mesh, dim=3)
        self._taps: dict[int, list[torch.Tensor]] | None = None
        self._tap_rows: dict[int, int] = {}
        self._tap_first = 0
        self._hook_taps()
        self.lock = threading.Lock()
        # Every program an eager request path can reach compiles before the round trace is captured:
        # a compile after capture allocates over the parked trace and the next request hangs
        # (model.warmup_prefill_masked_buckets). Prefill buckets, the chunked path past one chunk,
        # and the drafter's context and proposal programs; then the round itself is captured.
        started = time.monotonic()
        self.model.warmup_prefill_masked_buckets(self.page_table)
        self._prefill([self.tokenizer.eos_token_id] * (2 * self.window + 100))
        for _ in self.generate(self.tokenizer("warm up the round", add_special_tokens=False).input_ids, 2 * self.block):
            pass
        # From here a program-cache miss raises instead of silently clobbering the trace.
        self.mesh.set_program_cache_misses_allowed(False)
        counting = " ".join(str(n) for n in range(150))
        for _ in self.generate(self.tokenizer(counting, add_special_tokens=False).input_ids, 4 * self.block):
            pass
        log.info(
            "warm in %.0f s, %d cached programs", time.monotonic() - started, self.mesh.num_program_cache_entries()
        )

    def _hook_taps(self) -> None:
        """Keep each tap layer's prefill output rows at positions >= _tap_first on the host."""
        for index in self.drafter.taps:
            layer = self.model.layers[index]
            forward = layer.forward

            def hooked(*args, _forward=forward, _index=index, **kwargs):
                out = _forward(*args, **kwargs)
                if self._taps is not None:
                    rows = out.shape[-2]
                    start = self._tap_rows.get(_index, 0)
                    self._tap_rows[_index] = start + rows
                    if start + rows > self._tap_first:
                        host = ttnn.to_torch(out, mesh_composer=self.composer).float().reshape(rows, -1)
                        self._taps.setdefault(_index, []).append(host[max(0, self._tap_first - start) :])
                return out

            layer.forward = hooked

    def _prefill(self, ids: list[int]) -> tuple[int, torch.Tensor]:
        """Prefill the prompt; return the first generated token and the last `window` tap rows."""
        length = len(ids)
        self._tap_first = max(0, length - self.window)
        self._taps, self._tap_rows = {}, {}
        try:
            logits = self.model.prefill_traced_chunked(torch.tensor([ids]), self.page_table, actual_len=length)
            rows = []
            for index in self.drafter.taps:
                kept = torch.cat(self._taps[index])
                # A padded final chunk leaves rows past the prompt; drop them.
                rows.append(kept[: length - self._tap_first])
            taps = torch.cat(rows, dim=-1)
        finally:
            self._taps = None
        host = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
        anchor = int(host.reshape(-1, host.shape[-1])[0, : self.vocab].float().argmax())
        return anchor, taps

    def generate(self, ids: list[int], budget: int):
        """Yield lists of new token ids, greedily, until a stop token or `budget` tokens."""
        length = len(ids)
        budget = min(budget, self.max_context - length - 2 * self.block)
        if budget < 1:
            raise ValueError("prompt leaves no room in the context")
        started = time.monotonic()
        anchor, taps = self._prefill(ids)
        log.info("prefill %d tokens in %.2f s", length, time.monotonic() - started)
        yield [anchor]
        produced = 1
        if anchor in self.stops or produced >= budget:
            return
        first = length - taps.shape[0]
        self.drafter.reset(first)
        # Whole 32-row chunks keep the context programs at one shape. The padding rows land on
        # positions at or past `length`: masked until verified rows overwrite them.
        padded = torch.nn.functional.pad(taps, (0, 0, 0, -taps.shape[0] % 32))
        self.drafter.append(padded, first)
        block = [anchor] + self.drafter.propose(anchor, length)
        self.round.begin(block, length, self.page_table)
        started, self.rounds, self.decoded = time.monotonic(), 0, 0
        try:
            yield from self._rounds(block, produced, budget)
        finally:
            elapsed = time.monotonic() - started
            log.info(
                "decode %d tokens in %d rounds, %.2f s: %.1f tok/s, %.2f tokens/round",
                self.decoded, self.rounds, elapsed, self.decoded / max(elapsed, 1e-9), self.decoded / max(self.rounds, 1),
            )

    def _rounds(self, block: list[int], produced: int, budget: int):
        while produced < budget:
            warm = self.round.trace is None
            egress = self.round.run()
            accepted, correction, drafts = self.round.select(egress)
            new = block[1:accepted] + [correction]
            self.rounds += 1
            self.decoded += len(new)
            block = [correction] + drafts
            self.round.stage(block)
            if warm:
                self.round.capture()
            for index, token in enumerate(new):
                if token in self.stops or produced + index + 1 >= budget:
                    yield new[: index + 1]
                    return
            produced += len(new)
            yield new


def create_app(engine: Engine, name: str, key: str | None) -> FastAPI:
    app = FastAPI()
    tokenizer = engine.tokenizer

    def authorize(request: Request) -> None:
        if key and request.headers.get("authorization") != f"Bearer {key}":
            raise HTTPException(status_code=401, detail="invalid API key")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models(request: Request):
        authorize(request)
        return {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "local", "max_model_len": engine.max_context}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        authorize(request)
        body = await request.json()
        kwargs = dict(body.get("chat_template_kwargs") or {})
        tools = body.get("tools") or None
        messages = [_template_message(m) for m in body["messages"]]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
                **kwargs,
            )
        except Exception as error:  # noqa: BLE001 - the template's own message is the client's answer
            raise HTTPException(status_code=400, detail=f"chat template: {error}") from error
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
        budget = int(body.get("max_completion_tokens") or body.get("max_tokens") or engine.max_context)
        if len(ids) + 2 * engine.block >= engine.max_context:
            raise HTTPException(status_code=400, detail=f"prompt of {len(ids)} tokens exceeds the context")
        thinking = prompt.rstrip().endswith("<think>")
        created, rid = int(time.time()), f"chatcmpl-{uuid.uuid4().hex}"
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        # Set once the client stops listening: the engine serves one request at a time, so a
        # generation nobody reads must not hold the lock until its budget runs out.
        cancelled = threading.Event()

        def work() -> None:
            with engine.lock:
                if cancelled.is_set():
                    log.info("dropped a request whose client left while it waited")
                    return
                try:
                    for tokens in engine.generate(ids, budget):
                        if cancelled.is_set():
                            log.info("client left; generation stopped")
                            break
                        loop.call_soon_threadsafe(queue.put_nowait, tokens)
                    loop.call_soon_threadsafe(queue.put_nowait, None)
                except Exception as error:  # noqa: BLE001 - reported to the client
                    loop.call_soon_threadsafe(queue.put_nowait, error)

        threading.Thread(target=work, daemon=True).start()
        splitter = _Splitter(tokenizer, engine.stops, thinking)
        calls = _ToolCalls(tools)

        async def events():
            started = time.monotonic()
            try:
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=1.0)
                    except TimeoutError:
                        # Streaming disconnects cancel this task; a waiting or non-streaming
                        # client is only seen by polling.
                        if await request.is_disconnected():
                            return
                        continue
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    reasoning, content = splitter.feed(item)
                    yield reasoning, calls.feed(content)
            finally:
                cancelled.set()
            reasoning, content = splitter.finish()
            content = calls.feed(content)
            yield reasoning, content + calls.finish()
            splitter.elapsed = time.monotonic() - started

        def usage():
            return {
                "prompt_tokens": len(ids),
                "completion_tokens": splitter.count,
                "total_tokens": len(ids) + splitter.count,
            }

        if body.get("stream"):

            async def stream():
                def chunk(delta, finish=None):
                    return "data: " + json.dumps(
                        {
                            "id": rid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": name,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                        }
                    ) + "\n\n"

                yield chunk({"role": "assistant", "content": ""})
                async for reasoning, content in events():
                    delta = {}
                    if reasoning:
                        delta["reasoning_content"] = reasoning
                    if content:
                        delta["content"] = content
                    if delta:
                        yield chunk(delta)
                if calls.parsed:
                    yield chunk({"tool_calls": [{"index": i, **call} for i, call in enumerate(calls.parsed)]})
                yield chunk({}, "tool_calls" if calls.parsed else splitter.reason)
                if (body.get("stream_options") or {}).get("include_usage"):
                    yield "data: " + json.dumps(
                        {"id": rid, "object": "chat.completion.chunk", "created": created, "model": name, "choices": [], "usage": usage()}
                    ) + "\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")

        reasoning_parts, content_parts = [], []
        async for reasoning, content in events():
            reasoning_parts.append(reasoning)
            content_parts.append(content)
        content = "".join(content_parts)
        message = {"role": "assistant", "content": content or (None if calls.parsed else "")}
        if any(reasoning_parts):
            message["reasoning_content"] = "".join(reasoning_parts)
        if calls.parsed:
            message["tool_calls"] = calls.parsed
        return JSONResponse(
            {
                "id": rid,
                "object": "chat.completion",
                "created": created,
                "model": name,
                "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls.parsed else splitter.reason}],
                "usage": usage(),
            }
        )

    return app


class _Splitter:
    """Incremental detokenizer that routes text before `</think>` to reasoning."""

    def __init__(self, tokenizer, stops: set[int], thinking: bool) -> None:
        self.tokenizer, self.stops = tokenizer, stops
        self.ids: list[int] = []
        self.sent = 0
        self.thinking = thinking
        self.count = 0
        self.reason = "length"
        self.elapsed = 0.0

    def feed(self, tokens: list[int]) -> tuple[str, str]:
        for token in tokens:
            self.count += 1
            if token in self.stops:
                self.reason = "stop"
                break
            self.ids.append(token)
        return self._drain(final=False)

    def finish(self) -> tuple[str, str]:
        return self._drain(final=True)

    def _drain(self, final: bool) -> tuple[str, str]:
        text = self.tokenizer.decode(self.ids, skip_special_tokens=False)
        if not final and text.endswith("\ufffd"):
            return "", ""
        if not final and self.thinking:
            # Hold back a possible partial "</think>" so the split lands on it exactly.
            hold = next((n for n in range(len("</think>"), 0, -1) if text.endswith("</think>"[:n])), 0)
            if hold and not text.endswith("</think>"):
                text = text[:-hold]
        new, self.sent = text[self.sent :], max(self.sent, len(text))
        if not self.thinking:
            return "", new
        marker = new.find("</think>")
        if marker < 0:
            return new, ""
        self.thinking = False
        return new[:marker], new[marker + len("</think>") :].lstrip("\n")


_TOOL_OPEN = "<tool_call>"
# The chat template's call format: <tool_call><function=NAME><parameter=KEY>\nVALUE\n</parameter>
# ...</function></tool_call>. A call cut off by the output budget keeps what it has.
_FUNCTION = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=([^>\n]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)", re.DOTALL)


def _template_message(message: dict) -> dict:
    """OpenAI tool-call arguments arrive as JSON text; the template iterates them as a mapping."""
    if not message.get("tool_calls"):
        return message
    converted = []
    for call in message["tool_calls"]:
        function = dict(call.get("function") or {})
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                function["arguments"] = json.loads(arguments) if arguments.strip() else {}
            except ValueError as error:
                raise HTTPException(status_code=400, detail=f"tool call arguments are not JSON: {error}") from error
        converted.append({**call, "function": function})
    return {**message, "tool_calls": converted}


class _ToolCalls:
    """Passes answer text through until the first `<tool_call>`, then holds the rest and parses it
    into OpenAI `tool_calls` at the end. Held text that parses to no call is returned as text."""

    def __init__(self, tools: list[dict] | None) -> None:
        self.enabled = bool(tools)
        self.types: dict[str, dict[str, object]] = {}
        for tool in tools or []:
            function = tool.get("function", tool)
            properties = (function.get("parameters") or {}).get("properties") or {}
            self.types[function.get("name")] = {key: (spec or {}).get("type") for key, spec in properties.items()}
        self.pending = ""
        self.held: str | None = None
        self.parsed: list[dict] = []

    def feed(self, text: str) -> str:
        if not self.enabled:
            return text
        if self.held is not None:
            self.held += text
            return ""
        text = self.pending + text
        at = text.find(_TOOL_OPEN)
        if at >= 0:
            self.held, self.pending = text[at:], ""
            return text[:at]
        # Hold back a possible partial marker so the cut lands on it exactly.
        hold = next((n for n in range(len(_TOOL_OPEN) - 1, 0, -1) if text.endswith(_TOOL_OPEN[:n])), 0)
        self.pending = text[len(text) - hold :]
        return text[: len(text) - hold]

    def finish(self) -> str:
        """Parse the held calls into `parsed`; return any text that is not a call."""
        if self.held is None:
            tail, self.pending = self.pending, ""
            return tail
        for block in self.held.split(_TOOL_OPEN)[1:]:
            match = _FUNCTION.search(block.split("</tool_call>")[0])
            if match is None:
                continue
            name = match.group(1).strip()
            kinds = self.types.get(name, {})
            arguments = {}
            for parameter in _PARAMETER.finditer(match.group(2)):
                key = parameter.group(1).strip()
                arguments[key] = _parameter_value(parameter.group(2), kinds.get(key))
            self.parsed.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                }
            )
        return "" if self.parsed else self.held


def _parameter_value(raw: str, kind: object) -> object:
    """The template writes each value between newlines, strings verbatim and the rest as JSON."""
    value = raw[1:] if raw.startswith("\n") else raw
    value = value[:-1] if value.endswith("\n") else value
    if kind == "string":
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value



def main() -> None:
    # SIGUSR1 dumps every thread's stack to stderr: the diagnostic for a stalled request.
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    model_dir = Path(os.environ["QWEN38_MODEL"])
    drafter_dir = Path(os.environ["QWEN38_DRAFTER"])
    max_context = int(os.environ.get("QWEN38_MAX_CONTEXT", "131072"))
    name = os.environ.get("SERVED_MODEL_NAME", "qwen3.8-27b")
    engine = Engine(model_dir, drafter_dir, max_context)
    app = create_app(engine, name, os.environ.get("VLLM_API_KEY"))
    print(f"listening on {os.environ.get('HOST', '0.0.0.0')}:{os.environ.get('PORT', '18030')} as {name}", flush=True)
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "18030")), log_level="info")


if __name__ == "__main__":
    main()
