# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-user GDN verification with per-row state snapshots and constant-cost prefix folds."""

import torch

import ttnn
from models.demos.qwen38.tt.gdn.tp import TPGatedDeltaNet
from models.demos.qwen38.tt.gdn.verify_recurrence import select_rows, select_state


class GDNVerification:
    """Own the checkpoint, per-row state snapshots and tap history, allocated before trace capture.

    requires: layer has stable batched recurrent state and four BF16 convolution taps; no
        concurrent use of layer while bound.
    ensures: verification leaves committed state unchanged; fold commits only the accepted
        prefix, preserving other slots and committed buffer addresses.
    hypothesis: L2 sequential decode continuation at every prefix, including zero and
        full acceptance; nonzero checkpoint and distinct neighboring live slots.
    """

    def __init__(self, layer: TPGatedDeltaNet, max_tokens: int) -> None:
        """Allocate checkpoint, snapshots and one tap history per verification width."""
        assert 1 <= max_tokens and layer.K + max_tokens <= 32
        assert layer._stable_state and layer.rec_state is not None and layer.conv_states is not None
        assert layer.K == 4
        self.layer = layer
        self.max_tokens = max_tokens
        self.slot = -1
        self.count = 0
        # One FP32 slot: verification only reads committed state and the fold rewrites it from
        # snapshots and history (whose first rows are the checkpoint taps), so the committed
        # buffers serve as the checkpoint and preparation copies nothing.
        self.aliased = layer.B == 1 and layer.rec_state.dtype == ttnn.float32
        self.checkpoint = None if self.aliased else self._allocate((1, layer.Nv, layer.Dk, layer.Dv), ttnn.float32)
        self.checkpoint_convs = (
            None if self.aliased else [self._allocate((1, 1, layer.qkv_dim_tp), ttnn.bfloat16) for _ in range(layer.K)]
        )
        self.snapshots = self._allocate((max_tokens, layer.Nv, layer.Dk, layer.Dv), ttnn.float32)
        self.histories = {
            count: self._allocate((1, layer.K + count, layer.qkv_dim_tp), ttnn.bfloat16)
            for count in range(1, max_tokens + 1)
        }
        # BF16 committed state (QWEN35_GDN_STATE_BF16=1) folds through an FP32 staging row.
        self.staging = (
            None
            if layer.rec_state.dtype == ttnn.float32
            else self._allocate((1, layer.Nv, layer.Dk, layer.Dv), ttnn.float32)
        )
        self._released = False

    def release(self) -> None:
        """Discard checkpoint, snapshots and histories without committing pending state.

        requires: no live trace references these buffers; no concurrent verification.
        ensures: committed layer state survives; repeated release is harmless.
        """
        if self._released:
            return
        staging = () if self.staging is None else (self.staging,)
        owned = () if self.aliased else (self.checkpoint, *self.checkpoint_convs)
        for tensor in (*owned, self.snapshots, *self.histories.values(), *staging):
            ttnn.deallocate(tensor)
        self._released = True
        self.count = 0
        self.slot = -1

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
        row = source if source.shape[dim] == 1 else self.layer._slice_along(source, dim, slot, slot + 1)
        operand = row if row.dtype == target.dtype else ttnn.typecast(row, target.dtype)
        ttnn.copy(operand, target)
        if operand is not row:
            ttnn.deallocate(operand)
        if row is not source:
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
        ensures: the checkpoint holds the selected slot; committed state unchanged.
        """
        layer = self.layer
        assert not self._released
        assert self.count == 0 and 0 <= slot < layer.B
        assert layer.rec_state is not None and layer.conv_states is not None and layer._stable_state
        self.slot = slot
        if self.aliased:
            self.checkpoint, self.checkpoint_convs = layer.rec_state, layer.conv_states
            return
        self._checkpoint_row(layer.rec_state, self.checkpoint, 0, slot)
        for source, checkpoint in zip(layer.conv_states, self.checkpoint_convs, strict=True):
            self._checkpoint_row(source, checkpoint, 1, slot)

    def verify_prepared(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Execute against the prepared checkpoint; record snapshots without committing.

        requires: prepare selected the checkpoint; 1..max_tokens consecutive rows.
        ensures: device commands contain no committed-slot selection.
        """
        layer = self.layer
        count = x.shape[-2]
        assert self.count == 0 and self.slot >= 0 and 1 <= count <= self.max_tokens
        saved = layer.B, layer.rec_state, layer.conv_states, layer._stable_state
        layer.B, layer.rec_state, layer.conv_states, layer._stable_state = 1, self.checkpoint, self.checkpoint_convs, True
        try:
            output = layer.forward_verify(x, self.snapshots, self.histories[count])
        finally:
            layer.B, layer.rec_state, layer.conv_states, layer._stable_state = saved
        self.count = count
        return output

    def fold(self, accepted: int | ttnn.Tensor) -> None:
        """Commit the state after the accepted verifier inputs into the selected slot.

        requires: pending verification; accepted is an integer or device FP32 tiled
            scalar in [0, count]; committed state has not advanced since verification.
        ensures: only accepted inputs change state; rejected writes never contribute.
            Device acceptance stays on device; cost is independent of the accepted count.
        """
        assert self.count > 0
        if not isinstance(accepted, ttnn.Tensor):
            assert 0 <= accepted <= self.count
        count, self.count = self.count, 0
        if isinstance(accepted, int) and accepted == 0:
            return
        layer = self.layer
        if self.staging is None:
            select_state(self.snapshots, self.checkpoint, accepted, layer.rec_state, self.slot)
        else:
            select_state(self.snapshots, self.checkpoint, accepted, self.staging, 0)
            layer._write_index(layer.rec_state, ttnn.typecast(self.staging, layer.rec_state.dtype), self.slot, dim=0)
        select_rows(self.histories[count], accepted, layer.conv_states, self.slot)
