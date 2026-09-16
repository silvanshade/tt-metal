# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 single-layer MTP with shared target embedding and output projection."""

import copy
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet
from models.demos.blackhole.qwen36.tt.gdn.verification import GDNVerification
from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.distributed_norm import DistributedNorm

if TYPE_CHECKING:
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model


class Qwen36MTP:
    """Predict from a token and the preceding final-normalized hidden state.

    # Specification
    - requires: a live TP-path target and one full-attention MTP checkpoint layer.
    - ensures: target embedding, LM head, RoPE and Q/K rotation are shared; decoder weights and KV are independent.
    - provides: final-normalized draft hidden states for recursive proposal or the shared target LM head.
    - fails: unsupported checkpoint configurations raise ValueError; weight and device failures propagate.
    - panics: none.

    # Adequacy
    - hypothesis: reference agreement detects reversed concatenation, missing unit offsets and shifted positions.
    - hypothesis: rejection replay detects stale KV visibility.
    """

    def __init__(
        self,
        target: "Qwen36Model",
        state_dict: dict[str, torch.Tensor],
        cache_path: Path,
        *,
        max_verify_tokens: int = 32,
    ) -> None:
        """Load independent draft weights without duplicating shared target weights.

        # Specification
        - requires: state_dict contains the stripped mtp.* checkpoint namespace; cache_path identifies those weights.
        - ensures: target configuration and state remain unchanged; draft uses a full-attention layer at index zero.
        - fails: non-TP raises ValueError; missing weights raise KeyError; device and cache I/O failures propagate.
        - panics: none.

        # Adequacy
        - hypothesis: reference outputs detect wrong layers and cache collisions; target continuation detects aliasing.
        """
        if not 1 <= max_verify_tokens <= 32:
            raise ValueError(f"MTP verification bound must be in [1, 32], got {max_verify_tokens}")
        self.max_verify_tokens = max_verify_tokens
        if not target.tp_path:
            raise ValueError("MTP requires the TP model path")
        self.target = target
        self.args = copy.copy(target.args)
        self.args.attention_type_list = ["full_attention"]
        self.args.n_layers = 1
        cache_path.mkdir(parents=True, exist_ok=True)
        self.pre_fc_norm_embedding = self._make_norm(state_dict, "pre_fc_norm_embedding", cache_path, fractured=True)
        self.pre_fc_norm_hidden = self._make_norm(state_dict, "pre_fc_norm_hidden", cache_path, fractured=False)
        self.fc = tpc.shard_w(
            state_dict["fc.weight"],
            target.mesh_device,
            -1,
            ttnn.DRAM_MEMORY_CONFIG,
            cache_path / "fc.column",
        )
        self.layer = Qwen36DecoderLayer(
            target.mesh_device,
            self.args,
            state_dict,
            0,
            cache_path,
            tt_ccl=target.tt_ccl,
            qk_rotation=target.qk_rotation,
        )
        self.norm = self._make_norm(state_dict, "norm", cache_path, fractured=True)
        self.previous_hidden: dict[int, ttnn.Tensor] = {}
        self._prefill_tokens: torch.Tensor | None = None
        self._prefill_slot = 0
        self._prefill_inputs: dict[int, tuple[ttnn.Tensor, ...]] = {}
        self._proposal_inputs: tuple[ttnn.Tensor, ...] = ()
        self._proposal_outputs: tuple[ttnn.Tensor, ttnn.Tensor] | None = None
        self._proposal_trace: ttnn.MeshTraceId | None = None
        self._kv_caches: tuple[ttnn.Tensor, ...] = ()
        self.verifier: Qwen36MTPVerifier | None = None

    def allocate_kv_caches(self, kv_cache_shape: tuple[int, ...]) -> None:
        """Allocate independent BFP8 draft KV and target verification tapes.

        requires: target caches and stable recurrent state allocated; no model traces;
            no existing draft allocation; shape matches target physical page pool.
        ensures: draft pages use target block IDs without sharing target KV storage.
            Target verification supports the configured maximum input count.
        """
        assert not self._kv_caches and self.verifier is None
        attention = self.layer.attention
        assert isinstance(attention, TPAttention)
        self._kv_caches = tuple(
            ttnn.as_tensor(
                torch.zeros(kv_cache_shape, dtype=torch.bfloat16),
                device=self.target.mesh_device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.target.mesh_device),
            )
            for _ in range(2)
        )
        attention.set_paged_kv_cache(*self._kv_caches)
        self.verifier = Qwen36MTPVerifier(self.target, self.max_verify_tokens)
        initial_hidden = ttnn.from_torch(
            torch.zeros((1, 1, 1, self.target.args.dim), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.target.mesh_device),
        )
        self.previous_hidden = {
            slot: ttnn.to_device(initial_hidden, self.target.mesh_device)
            for slot in range(self.target.args.max_batch_size)
        }

    def free_kv_caches(self) -> None:
        """Release proposal trace, feedback, verifier tapes, then draft KV.

        requires: caller released other traces referencing verification or draft buffers.
        ensures: no speculative state committed; weights retained for reallocation.
        """
        self.release_proposal()
        for hidden in self.previous_hidden.values():
            ttnn.deallocate(hidden)
        self.previous_hidden.clear()
        for buffers in self._prefill_inputs.values():
            for tensor in buffers:
                ttnn.deallocate(tensor)
        self._prefill_inputs.clear()
        if self.verifier is not None:
            self.verifier.release()
            self.verifier = None
        attention = self.layer.attention
        assert isinstance(attention, TPAttention)
        attention.paged_k = attention.paged_v = None
        for tensor in self._kv_caches:
            ttnn.deallocate(tensor)
        self._kv_caches = ()

    def remap_slots(self, remap: torch.Tensor) -> None:
        """Move owned hidden feedback with the scheduler's gather permutation.

        requires: no pending verification; remap is a permutation of scheduler slots.
        ensures: slot i owns the prior hidden at remap[i]; padded slots stay put.
            No device allocation or copy; paged KV follows request block tables.
        """
        if self.verifier is not None:
            assert self.verifier.slot == -1
        indices = [int(value) for value in remap]
        assert sorted(indices) == list(range(len(indices)))
        old = self.previous_hidden
        self.previous_hidden = {
            slot: old[source]
            for slot in range(self.target.args.max_batch_size)
            if (source := indices[slot] if slot < len(indices) else slot) in old
        }

    def _make_norm(
        self, state_dict: dict[str, torch.Tensor], name: str, cache_path: Path, *, fractured: bool
    ) -> RMSNorm | DistributedNorm:
        """Build zero-centered normalization for fractured or replicated hidden states.

        # Specification
        - requires: name identifies a hidden-width checkpoint norm; fractured matches the input mesh distribution.
        - ensures: applies RMSNorm with gamma=1+weight and the target epsilon; output is replicated across devices.
        - fails: missing weights raise KeyError; device and cache I/O failures propagate.
        - panics: none.

        # Adequacy
        - hypothesis: nonuniform vectors versus checkpoint RMSNorm detect offset and cross-device statistics faults.
        """
        norm = RMSNorm(
            device=self.target.mesh_device,
            dim=self.args.dim,
            state_dict=state_dict,
            weight_key=name,
            weight_cache_path=cache_path,
            weight_dtype=ttnn.bfloat16,
            add_unit_offset=True,
            eps=self.args.norm_eps,
            is_distributed=self.args.is_distributed_norm if fractured else None,
            ccl_topology=self.args.ccl_topology(),
            tt_ccl=self.target.tt_ccl,
        )
        if fractured:
            return DistributedNorm(norm, self.args, tt_ccl=self.target.tt_ccl, TG=self.args.is_galaxy)
        return norm

    @classmethod
    def from_pretrained(cls, target: "Qwen36Model", checkpoint: Path, *, max_verify_tokens: int = 32) -> "Qwen36MTP":
        """Read only MTP checkpoint tensors and share the loaded target.

        # Specification
        - requires: checkpoint is a local indexed safetensors checkpoint matching the target.
        - ensures: loads one MTP layer without loading another embedding or LM head.
        - fails: unsupported MTP count or dedicated embeddings raise ValueError.
        - fails: malformed metadata, missing weights and I/O failures propagate.
        - panics: none.

        # Adequacy
        - hypothesis: reference outputs detect missing or misnamed weights.
        - hypothesis: unsupported configurations detect silent architecture substitution.
        """
        config = json.loads((checkpoint / "config.json").read_text())
        text = config.get("text_config", config)
        if text.get("mtp_num_hidden_layers") != 1 or text.get("mtp_use_dedicated_embeddings", False):
            raise ValueError("MTP requires one layer and shared embeddings")
        weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        names = {name: shard for name, shard in weight_map.items() if name.startswith("mtp.")}
        weights = {}
        for shard_name in sorted(set(names.values())):
            with safe_open(checkpoint / shard_name, framework="pt", device="cpu") as shard:
                for name, source in names.items():
                    if source == shard_name:
                        weights[name.removeprefix("mtp.")] = shard.get_tensor(name)
        return cls(target, weights, target.args.weight_cache_path() / "mtp", max_verify_tokens=max_verify_tokens)

    def forward(
        self,
        token_ids: ttnn.Tensor,
        previous_hidden: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        cache_positions: ttnn.Tensor,
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Advance one draft token per slot using independent paged attention KV.

        # Specification
        - requires: draft KV is attached; token IDs and previous_hidden identify matching slots.
        - requires: hidden is replicated [1,1,B,dim], final-normalized by the preceding target or draft step.
        - requires: cos/sin encode absolute token positions; active cache_positions equal absolute positions minus one.
        - requires: the first draft token has absolute position one; inactive slots use cache position -1.
        - ensures: returns replicated final-normalized draft hidden states and preserves caller-owned inputs.
        - ensures: updates only draft KV at active positions.
        - fails: tensor, allocation and device operation failures propagate.
        - panics: none.

        # Adequacy
        - hypothesis: reference agreement across positions detects concatenation, normalization and rotary faults.
        - hypothesis: cropped-prefix replay detects rejected KV visibility and overwrite faults.
        """
        fused = self._project_inputs(token_ids, previous_hidden, Mode.DECODE)
        norm_config = dict(self.args.get_norm_config("lm_head", Mode.DECODE))
        norm_config["output_mem_config"] = ttnn.DRAM_MEMORY_CONFIG
        hidden = self.layer.forward(
            fused, cos, sin, mode="decode", position_tensor=cache_positions, page_table=page_table
        )
        ttnn.deallocate(fused)
        result = self.norm(hidden, mode=Mode.DECODE, norm_config=norm_config)
        ttnn.deallocate(hidden)
        return result

    def prepare_proposal(
        self,
        token_ids: ttnn.Tensor,
        previous_hidden: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        cache_positions: ttnn.Tensor,
        page_table: ttnn.Tensor,
    ) -> None:
        """Allocate stable proposal inputs and warm forward, projection and copies.

        requires: draft KV attached; no prepared proposal; inputs satisfy forward.
        ensures: caller retains inputs; persistent copies survive capture and replay.
            Warmup writes draft KV and must precede every model's trace capture.
        hypothesis: recursive replay matches eager hidden feedback and token choices.
        """
        assert not self._proposal_inputs and self._proposal_trace is None
        sources = (token_ids, previous_hidden, cos, sin, cache_positions, page_table)
        self._proposal_inputs = tuple(ttnn.clone(tensor) for tensor in sources)
        for source, target in zip(sources, self._proposal_inputs, strict=True):
            ttnn.copy(source, target)
        hidden = self.forward(*self._proposal_inputs)
        logits = self.target._lm_head(hidden)
        self._proposal_outputs = ttnn.clone(logits), ttnn.clone(hidden)
        for source, destination in zip((logits, hidden), self._proposal_outputs, strict=True):
            ttnn.copy(source, destination)
        # Recursive feedback may come directly from the previous replay's output.
        ttnn.copy(hidden, self._proposal_inputs[1])
        ttnn.copy(previous_hidden, self._proposal_inputs[1])
        ttnn.deallocate(logits)
        ttnn.deallocate(hidden)

    def capture_proposal(self) -> None:
        """Capture one shared draft step and LM projection after global warmup.

        requires: prepare_proposal completed; no existing proposal trace.
        ensures: owns captured outputs until release_proposal; capture writes draft KV.
            Caller restores request cache contents before inference if warmup touched them.
        hypothesis: program-cache-frozen capture and replay perform no JIT compilation.
        """
        assert self._proposal_inputs and self._proposal_trace is None
        assert self._proposal_outputs is not None
        mesh = self.target.mesh_device
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        hidden = self.forward(*self._proposal_inputs)
        logits = self.target._lm_head(hidden)
        for source, destination in zip((logits, hidden), self._proposal_outputs, strict=True):
            ttnn.copy(source, destination)
            ttnn.deallocate(source)
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        self._proposal_trace = trace_id

    def proposal_step(
        self,
        token_ids: ttnn.Tensor,
        previous_hidden: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        cache_positions: ttnn.Tensor,
        page_table: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Replay a shared proposal step with request-selected pages and positions.

        requires: captured trace; input shapes, dtypes and layouts match preparation.
            Caller device buffers must predate trace capture; upload host values into
            persistent staging buffers rather than allocating device inputs on replay.
            Inputs satisfy forward; request draft KV prefix is valid.
        ensures: returns borrowed logits and hidden, overwritten by the next replay.
            Prior borrowed hidden may be passed directly as recursive feedback.
            Enqueues on CQ0; host consumption must synchronize through normal tensor read.
        hypothesis: alternating page tables and crossing page boundaries match eager
            recursive proposals without changing another request's KV.
        """
        assert self._proposal_trace is not None and self._proposal_outputs is not None
        sources = (token_ids, previous_hidden, cos, sin, cache_positions, page_table)
        for source, target in zip(sources, self._proposal_inputs, strict=True):
            ttnn.copy(source, target)
        ttnn.execute_trace(self.target.mesh_device, self._proposal_trace, cq_id=0, blocking=False)
        return self._proposal_outputs

    def release_proposal(self) -> None:
        """Release trace before its persistent buffers; preserve weights and draft KV.

        requires: no concurrent proposal consumption; borrowed outputs no longer used.
        ensures: repeated release is harmless; preparation can allocate a fresh trace.
        hypothesis: release and recapture preserve subsequent proposal results.
        """
        if self._proposal_trace is not None:
            ttnn.release_trace(self.target.mesh_device, self._proposal_trace)
            self._proposal_trace = None
        if self._proposal_outputs is not None:
            for tensor in self._proposal_outputs:
                ttnn.deallocate(tensor)
            self._proposal_outputs = None
        for tensor in self._proposal_inputs:
            ttnn.deallocate(tensor)
        self._proposal_inputs = ()

    def refresh(
        self,
        token_ids: ttnn.Tensor,
        previous_hidden: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        cache_positions: ttnn.Tensor,
        page_table: ttnn.Tensor,
    ) -> None:
        """Replace accepted draft KV with teacher-forced target feedback.

        requires: 1..32 consecutive input tokens [T,1]; previous_hidden contains
            their preceding final-normalized target rows, not recursive draft rows.
        requires: every page-table row names the same sequence; cache positions
            equal absolute token positions minus one; cos/sin use absolute positions.
        ensures: accepted input KV replaces speculative entries, including the last
            proposal when all drafts were accepted; caller-owned tensors unchanged.
        hypothesis: continuation after each accepted prefix matches sequential
            teacher-forced decode even when rejected entries contain different KV.
        """
        assert 1 <= token_ids.shape[0] <= 32
        fused = self._project_inputs(token_ids, previous_hidden, Mode.DECODE)
        hidden = self.layer.forward(
            fused, cos, sin, mode="verify", position_tensor=cache_positions, page_table=page_table
        )
        ttnn.deallocate(fused)
        ttnn.deallocate(hidden)

    def _project_inputs(self, token_ids: ttnn.Tensor, previous_hidden: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        """Fuse aligned tokens and preceding normalized hidden rows; caller owns output.

        requires: replicated hidden rows match flattened token rows; mode selects
            prefill or decode normalization. Caller-owned inputs remain unchanged.
        hypothesis: teacher-forced prefill and decode agree within precision bounds.
        """
        embedding = self.target.embd(token_ids)
        embedding = ttnn.reshape(embedding, (1, 1, embedding.shape[0] * embedding.shape[1], embedding.shape[-1]))
        norm_config = dict(self.args.get_norm_config("lm_head", mode))
        norm_config["output_mem_config"] = ttnn.DRAM_MEMORY_CONFIG
        normalized_embedding = self.pre_fc_norm_embedding(embedding, mode=mode, norm_config=norm_config)
        ttnn.deallocate(embedding)
        normalized_hidden = self.pre_fc_norm_hidden(previous_hidden, mode=mode)
        joined = ttnn.concat([normalized_embedding, normalized_hidden], dim=-1)
        ttnn.deallocate(normalized_embedding)
        ttnn.deallocate(normalized_hidden)
        fused = ttnn.linear(
            joined, self.fc, compute_kernel_config=tpc.COMPUTE_HIFI2, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(joined)
        return fused

    def prefill(
        self,
        token_ids: ttnn.Tensor,
        previous_hidden: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        page_table: ttnn.Tensor,
        chunk_page_table: ttnn.Tensor,
        chunk_start: int,
        chunk_start_tensor: ttnn.Tensor | None = None,
    ) -> None:
        """Seed independent draft KV from teacher-forced target hidden rows.

        requires: target hidden rows at positions c..c+S-1 align with input tokens
            c+1..c+S; cos/sin encode those token positions; chunk_start=c.
        requires: c is page-aligned; padded S fits chunk_page_table; draft KV attached.
        requires: a supplied chunk_start_tensor contains c and selects dynamic-offset SDPA.
        ensures: fills draft cache positions c..c+S-1; target state and inputs unchanged.
            Padded suffix is invisible to subsequent position-bounded decode.
        fails: tensor and device failures propagate.
        hypothesis: continuation after multiple chunks detects shifted RoPE, token
            alignment, page offsets, and accidental visibility of padded KV.
        """
        fused = self._project_inputs(token_ids, previous_hidden, Mode.PREFILL)
        hidden = self.layer.forward(
            fused,
            cos,
            sin,
            mode="prefill",
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start,
            chunk_start_idx_tensor=chunk_start_tensor,
        )
        ttnn.deallocate(fused)
        ttnn.deallocate(hidden)

    @contextmanager
    def prefill_request(self, tokens: torch.Tensor, slot: int) -> Iterator[None]:
        """Bind one prompt's hidden feedback until prefill completes.

        requires: unpadded [1,T] tokens; no nested prefill; slot owns this request.
        ensures: slot feedback storage is reused; prompt binding clears on failure.
        hypothesis: successive prompts and slot reuse cannot consume stale feedback.
        """
        assert self._prefill_tokens is None and tokens.shape[0] == 1
        self._prefill_tokens = tokens
        self._prefill_slot = slot
        try:
            yield
        finally:
            self._prefill_tokens = None

    def observe_prefill(self, hidden: ttnn.Tensor, start: int, pages: torch.Tensor) -> None:
        """Seed draft KV before the target chunk output can be overwritten.

        requires: hidden contains final-layer, unnormalized rows starting at start;
            active prompt binding supplies the next token across chunk boundaries.
        ensures: draft consumes target-normalized rows and shifted prompt tokens;
            no hidden carry between chunks; caller retains hidden ownership.
        hypothesis: chunk boundaries and padded tails match teacher-forced continuation.
        """
        tokens = self._prefill_tokens
        if tokens is None:
            return
        size = hidden.shape[-2]
        valid = min(size, tokens.shape[1] - start - 1)
        if valid <= 0:
            return
        mapper = ttnn.ReplicateTensorToMesh(self.target.mesh_device)

        def host(value: torch.Tensor, dtype: ttnn.DataType, layout: ttnn.Layout) -> ttnn.Tensor:
            return ttnn.from_torch(value.contiguous(), dtype=dtype, layout=layout, mesh_mapper=mapper)

        shifted = torch.zeros((1, size), dtype=torch.int32)
        shifted[:, :valid] = tokens[:, start + 1 : start + 1 + valid]
        assert isinstance(self.layer.attention, TPAttention) and self.layer.attention.paged_k is not None
        block_size = self.layer.attention.paged_k.shape[2]
        first_block = start // block_size
        chunk_pages = pages[:, first_block : first_block + size // block_size]
        # Fixed full width prevents per-position page-table programs after capture.
        width = ((self.args.max_seq_len + size + block_size * 32 - 1) // (block_size * 32)) * 32
        full_pages = torch.zeros((1, width), dtype=torch.int32)
        full_pages[:, : pages.shape[1]] = pages
        cos_values, sin_values = self.target._rope_tp_cos_sin_torch(start + 1, size)
        sources = (
            host(shifted, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            host(chunk_pages, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            host(full_pages, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            host(torch.tensor([start], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            host(cos_values, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            host(sin_values, ttnn.bfloat16, ttnn.TILE_LAYOUT),
        )
        buffers = self._prefill_inputs.get(size)
        if buffers is None:
            buffers = tuple(ttnn.to_device(source, self.target.mesh_device) for source in sources)
            self._prefill_inputs[size] = buffers
        else:
            for source, buffer in zip(sources, buffers, strict=True):
                ttnn.copy_host_to_device_tensor(source, buffer)
        ids, chunk_pages, table, offset, cos, sin = buffers
        normalized = self.target.norm(hidden, mode=Mode.PREFILL)
        self.prefill(ids, normalized, cos, sin, table, chunk_pages, start, offset)
        ttnn.deallocate(normalized)

    def retain_prefill_hidden(self, hidden: ttnn.Tensor) -> None:
        """Retain the exact normalized row used for the prompt's target logits.

        requires: hidden is the selected last row; caller retains its tensor.
        ensures: active slot owns an independent copy for its first draft anchor.
        hypothesis: first-anchor prediction detects wrong row selection or alias reuse.
        """
        if self._prefill_tokens is not None:
            previous = self.previous_hidden.get(self._prefill_slot)
            if previous is None:
                self.previous_hidden[self._prefill_slot] = ttnn.clone(hidden)
            else:
                ttnn.copy(hidden, previous)


class Qwen36MTPVerifier:
    """Verify one sequence against target layers without committing rejected GDN state.

    requires: target paged caches and stable GDN state allocated before construction;
        construct before trace capture; one pending verification across all slots.
    ensures: target logits and normalized hidden rows accompany a prefix-fold operation;
        fold returns next absolute KV position and commits only accepted verifier inputs.
    hypothesis: sequential target logits and continuation across rejection boundaries;
        rejected suffix differs from subsequent real input.
    """

    def __init__(self, target: "Qwen36Model", max_tokens: int) -> None:
        """Allocate every GDN checkpoint and tape before target trace capture."""
        assert target.tp_path and 1 <= max_tokens <= 32
        self.target = target
        self.max_tokens = max_tokens
        self.gdn: dict[int, GDNVerification] = {}
        for index, layer in enumerate(target.layers):
            if not layer.is_full_attention:
                assert isinstance(layer.attention, TPGatedDeltaNet)
                self.gdn[index] = GDNVerification(layer.attention, max_tokens)
        self.count = 0
        self.start_position = 0
        self.slot = -1

    def release(self) -> None:
        """Discard verifier buffers after caller-owned verification traces are released.

        requires: no concurrent verifier execution or fold.
        ensures: target committed state unchanged; pending speculative state discarded.
            Repeated release is harmless; released verifier must not be reused.
        """
        for verifier in self.gdn.values():
            verifier.release()
        self.count = 0
        self.slot = -1
        self.max_tokens = 0

    def verify(
        self,
        token_ids: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        positions: ttnn.Tensor,
        page_table: ttnn.Tensor,
        slot: int,
        start_position: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Return target logits and final-normalized hidden rows for consecutive tokens.

        requires: tokens [T,1], 1 <= T <= max_tokens; positions start at start_position;
            every page-table row names slot's sequence; no pending verification.
        ensures: GDN committed state unchanged; attention suffix remains logically
            speculative until fold returns next KV position. Caller owns both outputs.
        """
        self.prepare(slot, start_position)
        return self.verify_prepared(token_ids, cos, sin, positions, page_table)

    def prepare(self, slot: int, start_position: int) -> None:
        """Select and checkpoint a sequence before eager execution or trace replay.

        requires: no pending verification; valid slot and nonnegative position.
        ensures: committed-slot copies stay outside the shared verification trace.
        """
        assert self.count == 0 and self.slot == -1 and start_position >= 0
        assert self.max_tokens > 0
        for verifier in self.gdn.values():
            verifier.prepare(slot)
        self.slot = slot
        self.start_position = start_position

    def verify_prepared(
        self,
        token_ids: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        positions: ttnn.Tensor,
        page_table: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Execute the slot-independent verifier from its prepared checkpoint.

        requires: prepare completed; 1..max_tokens consecutive input rows.
        ensures: caller owns logits and hidden rows; fold commits only a chosen prefix.
        """
        assert self.count == 0 and self.slot >= 0 and 1 <= token_ids.shape[0] <= self.max_tokens
        target = self.target
        x = target.embd(token_ids)
        x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1], x.shape[-1]))
        for index, layer in enumerate(target.layers):
            output = layer.forward(
                x,
                cos,
                sin,
                mode="verify",
                position_tensor=positions,
                page_table=page_table,
                gdn_verification=self.gdn.get(index),
            )
            ttnn.deallocate(x)
            x = output
        hidden = target._final_norm_decode(x)
        ttnn.deallocate(x)
        logits = target._lm_head(hidden)
        self.record_replay(token_ids.shape[0])
        return logits, hidden

    def record_replay(self, count: int) -> None:
        """Restore host fold metadata after successful device verification.

        requires: prepare completed and eager execution or trace replay succeeded.
        ensures: every layer folds the same count into the selected committed slot.
        """
        assert self.count == 0 and self.slot >= 0 and 1 <= count <= self.max_tokens
        self.count = count
        for verifier in self.gdn.values():
            verifier.slot, verifier.count = self.slot, count

    def fold(self, accepted_inputs: int) -> int:
        """Commit accepted verifier inputs; return next absolute KV write position.

        requires: successful pending verification; accepted_inputs includes anchor,
            excludes correction/bonus; 0 <= accepted_inputs <= verified count.
        ensures: rejected KV masked by returned position; every GDN layer commits
            same prefix; correction/bonus remains next round's unprocessed anchor.
        """
        assert self.count > 0 and 0 <= accepted_inputs <= self.count
        for verifier in self.gdn.values():
            verifier.fold(accepted_inputs)
        self.count = 0
        self.slot = -1
        return self.start_position + accepted_inputs
