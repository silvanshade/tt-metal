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

    def __init__(self, target: "Qwen36Model", state_dict: dict[str, torch.Tensor], cache_path: Path) -> None:
        """Load independent draft weights without duplicating shared target weights.

        # Specification
        - requires: state_dict contains the stripped mtp.* checkpoint namespace; cache_path identifies those weights.
        - ensures: target configuration and state remain unchanged; draft uses a full-attention layer at index zero.
        - fails: non-TP raises ValueError; missing weights raise KeyError; device and cache I/O failures propagate.
        - panics: none.

        # Adequacy
        - hypothesis: reference outputs detect wrong layers and cache collisions; target continuation detects aliasing.
        """
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
    def from_pretrained(cls, target: "Qwen36Model", checkpoint: Path) -> "Qwen36MTP":
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
        return cls(target, weights, target.args.weight_cache_path() / "mtp")

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
        ensures: prior slot feedback is released; prompt binding clears on failure.
        hypothesis: successive prompts and slot reuse cannot consume stale feedback.
        """
        assert self._prefill_tokens is None and tokens.shape[0] == 1
        previous = self.previous_hidden.pop(slot, None)
        if previous is not None:
            ttnn.deallocate(previous)
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

        def upload(value: torch.Tensor, dtype: ttnn.DataType, layout: ttnn.Layout) -> ttnn.Tensor:
            return ttnn.from_torch(
                value.contiguous(),
                dtype=dtype,
                layout=layout,
                device=self.target.mesh_device,
                mesh_mapper=mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        shifted = torch.zeros((1, size), dtype=torch.int32)
        shifted[:, :valid] = tokens[:, start + 1 : start + 1 + valid]
        ids = upload(shifted, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        assert isinstance(self.layer.attention, TPAttention) and self.layer.attention.paged_k is not None
        block_size = self.layer.attention.paged_k.shape[2]
        first_block = start // block_size
        chunk_pages = upload(
            pages[:, first_block : first_block + size // block_size],
            ttnn.int32,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        # Fixed full width prevents per-position page-table programs after capture.
        width = ((self.args.max_seq_len + size + block_size * 32 - 1) // (block_size * 32)) * 32
        full_pages = torch.zeros((1, width), dtype=torch.int32)
        full_pages[:, : pages.shape[1]] = pages
        table = upload(full_pages, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        offset = upload(torch.tensor([start], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        cos_values, sin_values = self.target._rope_tp_cos_sin_torch(start + 1, size)
        cos = upload(cos_values, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        sin = upload(sin_values, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        normalized = self.target.norm(hidden, mode=Mode.PREFILL)
        self.prefill(ids, normalized, cos, sin, table, chunk_pages, start, offset)
        for tensor in (ids, normalized, cos, sin, table, chunk_pages, offset):
            ttnn.deallocate(tensor)

    def retain_prefill_hidden(self, hidden: ttnn.Tensor) -> None:
        """Retain the exact normalized row used for the prompt's target logits.

        requires: hidden is the selected last row; caller retains its tensor.
        ensures: active slot owns an independent copy for its first draft anchor.
        hypothesis: first-anchor prediction detects wrong row selection or alias reuse.
        """
        if self._prefill_tokens is not None:
            self.previous_hidden[self._prefill_slot] = ttnn.clone(hidden)


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
