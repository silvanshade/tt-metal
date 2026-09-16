# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-user GDN verification with compact finite-precision state replay."""

import os

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import fused_decay_and_write_ttnn


class GDNVerification:
    """Own persistent scratch and state-write tape, allocated before trace capture.

    requires: layer has stable batched state; no concurrent use of layer while bound.
    ensures: verification leaves committed state unchanged; fold commits only accepted
        prefix, preserving other slots and committed buffer addresses.
    hypothesis: L2 sequential decode continuation at every prefix, including zero and
        full acceptance; nonzero checkpoint and distinct neighboring live slots.
    """

    def __init__(self, layer: TPGatedDeltaNet, max_tokens: int) -> None:
        """Allocate checkpoint, scratch, convolution inputs and rank-one update tape."""
        assert 1 <= max_tokens <= 32
        assert layer._stable_state and layer.rec_state is not None and layer.conv_states is not None
        self.layer = layer
        self.max_tokens = max_tokens
        self.slot = -1
        self.count = 0
        rec_shape = (1, layer.Nv, layer.Dk, layer.Dv)
        conv_shape = (1, 1, layer.qkv_dim_tp)
        self.checkpoint = self._allocate(rec_shape, layer.rec_state.dtype)
        self.scratch = self._allocate(rec_shape, layer.rec_state.dtype)
        self.checkpoint_convs = [self._allocate(conv_shape, ttnn.bfloat16) for _ in range(layer.K)]
        self.scratch_convs = [self._allocate(conv_shape, ttnn.bfloat16) for _ in range(layer.K)]
        self.conv_inputs = [self._allocate(conv_shape, ttnn.bfloat16) for _ in range(max_tokens)]
        update_dtype = ttnn.bfloat16 if os.environ.get("QWEN35_GDN_DECODE_BF16") == "1" else ttnn.float32
        shapes = ((1, layer.Nv, layer.Dk), (1, layer.Nv, layer.Dv), (1, layer.Nv, 1, 1), (1, layer.Nv))
        self.updates = [tuple(self._allocate(shape, update_dtype) for shape in shapes) for _ in range(max_tokens)]

    def _allocate(self, shape: tuple[int, ...], dtype: ttnn.DataType) -> ttnn.Tensor:
        """Allocate persistent replicated local-shard storage before capture."""
        return ttnn.from_torch(
            torch.zeros(shape, dtype=torch.float32),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.layer.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.layer.mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _checkpoint_row(self, source: ttnn.Tensor, target: ttnn.Tensor, dim: int, slot: int) -> None:
        """Copy one committed slot without consuming source or persistent target."""
        if source.shape[dim] == 1:
            ttnn.copy(source, target)
        else:
            row = self.layer._slice_along(source, dim, slot, slot + 1)
            ttnn.copy(row, target)
            ttnn.deallocate(row)

    def verify(self, x: ttnn.Tensor, slot: int) -> ttnn.Tensor:
        """Verify token block; preserve committed state even when execution raises.

        requires: x contains 1..max_tokens consecutive tokens for valid slot.
        ensures: returned rows follow sequential recurrence from slot checkpoint;
            fold may commit exactly one prefix before next verification.
        """
        self.prepare(slot)
        return self.verify_prepared(x)

    def prepare(self, slot: int) -> None:
        """Checkpoint one slot outside the slot-independent verification trace.

        requires: no pending verification; stable committed state and valid slot.
        ensures: scratch starts at the selected checkpoint; committed state unchanged.
        """
        layer = self.layer
        assert self.count == 0 and 0 <= slot < layer.B
        assert layer.rec_state is not None and layer.conv_states is not None and layer._stable_state
        self.slot = slot
        self._checkpoint_row(layer.rec_state, self.checkpoint, 0, slot)
        ttnn.copy(self.checkpoint, self.scratch)
        for source, checkpoint, scratch in zip(
            layer.conv_states, self.checkpoint_convs, self.scratch_convs, strict=True
        ):
            self._checkpoint_row(source, checkpoint, 1, slot)
            ttnn.copy(checkpoint, scratch)

    def verify_prepared(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Execute against prepared scratch; record writes without committing.

        requires: prepare selected the checkpoint; 1..max_tokens consecutive rows.
        ensures: device commands contain no committed-slot selection.
        """
        layer = self.layer
        assert self.count == 0 and self.slot >= 0 and 1 <= x.shape[-2] <= self.max_tokens
        saved = layer.B, layer.rec_state, layer.conv_states, layer._stable_state
        layer.B, layer.rec_state, layer.conv_states, layer._stable_state = 1, self.scratch, self.scratch_convs, True
        try:
            output = layer.forward_verify(x, self.updates, self.conv_inputs)
        finally:
            layer.B, layer.rec_state, layer.conv_states, layer._stable_state = saved
        self.count = x.shape[-2]
        return output

    def fold(self, accepted: int) -> None:
        """Commit accepted verifier inputs, including anchor, using recorded writes.

        requires: successful verification; 0 <= accepted <= verified token count;
            committed layer state has not advanced since verification.
        ensures: zero acceptance changes nothing; positive acceptance replays identical
            decay/write rounding and convolution inputs for exactly that prefix.
            Rejected suffix cannot affect committed state or subsequent decode.
        intension: no projection, normalization, query or state-read matmul during fold.
        """
        assert self.count > 0 and 0 <= accepted <= self.count
        self.count = 0
        if accepted == 0:
            return
        layer = self.layer
        h = ttnn.clone(self.checkpoint, memory_config=ttnn.L1_MEMORY_CONFIG)
        for k, delta, g, beta in self.updates[:accepted]:
            if h.dtype != k.dtype:
                converted = ttnn.typecast(h, k.dtype)
                ttnn.deallocate(h)
                h = converted
            decayed = ttnn.multiply(
                h, g, input_tensor_b_activations=[ttnn.UnaryOpType.EXP], memory_config=ttnn.L1_MEMORY_CONFIG
            )
            ttnn.deallocate(h)
            h = fused_decay_and_write_ttnn(decayed, k, delta, g, beta, device=layer.mesh, apply_decay=False)
            ttnn.deallocate(decayed)
            if h.dtype != self.checkpoint.dtype:
                converted = ttnn.typecast(h, self.checkpoint.dtype)
                ttnn.deallocate(h)
                h = converted
        convs = [
            ttnn.clone(
                self.checkpoint_convs[index + accepted]
                if index + accepted < layer.K
                else self.conv_inputs[index + accepted - layer.K],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for index in range(layer.K)
        ]
        layer.write_slot(self.slot, h, convs)
