# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Captured greedy DFlash2 rounds: verify, accept, fold, extend context and draft in one trace."""

from typing import TYPE_CHECKING

import torch

import ttnn
from models.demos.qwen38.tt.dflash import Qwen38DFlash
from models.demos.qwen38.tt.greedy import greedy_tokens
from models.demos.qwen38.tt.mtp import Qwen38MTPVerifier

if TYPE_CHECKING:
    from models.demos.qwen38.tt.model import Qwen38Model


class Qwen38DFlashRound:
    """One slot's greedy DFlash2 rounds, each a single replay of one captured program.

    # Specification
    - requires: target paged caches allocated; the verifier taps the drafter's layers and
      verifies exactly one drafter block; construction, `begin` and `capture` precede replay.
    - ensures: a round verifies the staged block, commits its accepted prefix to target and
      drafter state, advances the device-resident position, and drafts the next block from
      the correction token. Nothing allocated after capture outlives a replay.
    - provides: per device, one FP32 egress row per block position in tile-aligned segments:
      `[accepted, correction, 0...](32) | candidate values(C) | candidate shard-local ids(C) |
      selector projection(P)`; rows 1.. carry the next block's selector inputs.
    """

    def __init__(
        self, target: "Qwen38Model", verifier: Qwen38MTPVerifier, drafter: Qwen38DFlash, page_width: int, slot: int = 0
    ) -> None:
        assert verifier.max_tokens == drafter.block and verifier.taps == frozenset(drafter.taps)
        self.target, self.verifier, self.drafter, self.slot = target, verifier, drafter, slot
        self.mesh = target.mesh_device
        self.mapper = ttnn.ReplicateTensorToMesh(self.mesh)
        block = drafter.block
        self.rotary = tuple(
            self._tensor(table.to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
            for table in (target.rope.cos_cpu, target.rope.sin_cpu)
        )
        self.mask_ids = self._tensor(torch.full((1, 1, block - 1, 1), float(drafter.mask_token)), ttnn.float32)
        # Header columns: accepted count in column 0, correction token in column 1.
        header = torch.zeros(2, 1, 1, block, 32)
        header[0, ..., 0] = 1.0
        header[1, ..., 1] = 1.0
        self.header = tuple(self._tensor(basis, ttnn.float32) for basis in header)
        self.one = self._tensor(torch.ones(1, 1, 1, 1), ttnn.float32)
        self.candidates = drafter.local_candidates
        width = 32 + 2 * self.candidates + drafter.selector_width
        self.frame = {
            "ids": self._tensor(torch.zeros(block, 1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "pages": self._tensor(torch.zeros(block, page_width, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            "start": self._tensor(torch.zeros(1, 1, 1, 1), ttnn.float32),
            "egress": self._tensor(torch.zeros(1, 1, block, width), ttnn.float32),
        }
        self.trace = None

    def _tensor(self, value: torch.Tensor, dtype: ttnn.DataType, layout: ttnn.Layout = ttnn.TILE_LAYOUT) -> ttnn.Tensor:
        return ttnn.from_torch(value, dtype=dtype, layout=layout, device=self.mesh, mesh_mapper=self.mapper)

    def _upload(self, name: str, value: torch.Tensor) -> None:
        destination = self.frame[name]
        host = ttnn.from_torch(value, dtype=destination.dtype, layout=destination.layout, mesh_mapper=self.mapper)
        ttnn.copy_host_to_device_tensor(host, destination)

    @staticmethod
    def _float(tensor: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.typecast(ttnn.to_layout(tensor, ttnn.TILE_LAYOUT), ttnn.float32)

    @staticmethod
    def _indices(tensor: ttnn.Tensor, shape: tuple[int, ...], dtype: ttnn.DataType = ttnn.uint32) -> ttnn.Tensor:
        return ttnn.reshape(ttnn.to_layout(ttnn.typecast(tensor, dtype), ttnn.ROW_MAJOR_LAYOUT), shape)

    def _accept(self, predicted: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Accepted input count (anchor included) and the target token after that prefix, from
        the target's greedy tokens (FP32 [1, 1, block, 1])."""
        block = self.drafter.block
        inputs = ttnn.reshape(self._float(self.frame["ids"]), (1, 1, block, 1))
        accepted = ttnn.clone(self.one)
        prefix = ttnn.clone(self.one)
        correction = predicted[:, :, 0:1, :]
        for index in range(1, block):
            prefix = ttnn.multiply(prefix, ttnn.eq(inputs[:, :, index : index + 1, :], predicted[:, :, index - 1 : index, :]))
            accepted = ttnn.add(accepted, prefix)
            correction = ttnn.where(prefix, predicted[:, :, index : index + 1, :], correction)
        return accepted, correction

    def _round(self) -> None:
        accepted, correction, taps = self._verify()
        self._extend(taps)
        self._draft(accepted, correction)

    def _verify(self) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Verify the staged block and commit its accepted prefix to target state."""
        frame, block = self.frame, self.drafter.block
        absolute = ttnn.add(self.drafter.offsets, frame["start"])
        rope_index = self._indices(absolute, (block, 1))
        cos, sin = (
            ttnn.reshape(
                ttnn.embedding(rope_index, table, layout=ttnn.TILE_LAYOUT), (1, block, 1, self.target.args.rope_head_dim)
            )
            for table in self.rotary
        )
        self.verifier.prepare(self.slot, 0)
        logits, hidden, taps = self.verifier.verify_prepared(
            frame["ids"], cos, sin, self._indices(absolute, (block,), ttnn.int32), frame["pages"], gather_logits=False
        )
        target = self.target
        predicted = greedy_tokens(logits, self.mesh, target.tt_ccl, target.args.ccl_topology())
        for tensor in (logits, hidden, cos, sin, rope_index):
            ttnn.deallocate(tensor)
        accepted, correction = self._accept(predicted)
        for verification in self.verifier.gdn.values():
            verification.fold(accepted)
        self.verifier.count, self.verifier.slot = 0, -1
        return accepted, correction, taps

    def _extend(self, taps: ttnn.Tensor) -> None:
        """Every verified row enters the drafter context; rows past the accepted prefix stay
        masked until the next block overwrites them."""
        start = self.frame["start"]
        self.drafter.extend(taps, self.drafter.device_positions(start), self.drafter.device_base(start))
        ttnn.deallocate(taps)

    def _draft(self, accepted: ttnn.Tensor, correction: ttnn.Tensor) -> None:
        """Draft the block after the accepted prefix and export its selector inputs."""
        frame, drafter, block = self.frame, self.drafter, self.drafter.block
        following = ttnn.add(frame["start"], accepted)
        block_ids = self._indices(ttnn.concat([correction, self.mask_ids], dim=2), (block, 1))
        drafted = drafter.draft(block_ids, drafter.device_positions(following), drafter.device_mask(following))
        values, indices, projected = drafter.candidates(drafted)
        header = ttnn.add(ttnn.multiply(self.header[0], accepted), ttnn.multiply(self.header[1], correction))
        egress = ttnn.concat(
            [header, ttnn.typecast(values, ttnn.float32), ttnn.typecast(indices, ttnn.float32), projected], dim=3
        )
        ttnn.copy(egress, frame["egress"])
        ttnn.copy(following, frame["start"])

    def begin(self, tokens: list[int], position: int, pages: torch.Tensor) -> None:
        """Stage a request's first block (anchor plus drafts) at `position` with its page row."""
        block = self.drafter.block
        self._upload("pages", pages.reshape(1, -1).repeat(block, 1).to(torch.int32))
        self._upload("start", torch.full((1, 1, 1, 1), float(position)))
        self.stage(tokens)

    def stage(self, tokens: list[int]) -> None:
        assert len(tokens) == self.drafter.block
        self._upload("ids", torch.tensor(tokens, dtype=torch.int32).reshape(-1, 1))

    def run(self) -> torch.Tensor:
        """Execute one round (replay once captured, else eagerly); returns egress()."""
        if self.trace is None:
            self._round()
        else:
            ttnn.execute_trace(self.mesh, self.trace, cq_id=0, blocking=False)
        return self.egress()

    def egress(self) -> torch.Tensor:
        """Every device's egress rows [devices, block, width]."""
        parts = ttnn.get_device_tensors(self.frame["egress"])
        return torch.stack([ttnn.to_torch(part).float().reshape(self.drafter.block, -1) for part in parts])

    def capture(self) -> None:
        """Record the round after an eager warm round compiled every program."""
        assert self.trace is None
        trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self._round()
        ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
        self.trace = trace

    def select(self, egress: torch.Tensor) -> tuple[int, int, list[int]]:
        """Accepted input count, correction token, and the next block's drafts."""
        c = self.candidates
        accepted, correction = int(egress[0, 0, 0]), int(egress[0, 0, 1])
        rows = egress[:, 1:]
        values, ids = self.drafter.merge(rows[..., 32 : 32 + c], rows[..., 32 + c : 32 + 2 * c])
        drafts = self.drafter.select(rows[0, :, 32 + 2 * c :], values, ids, correction)
        return accepted, correction, drafts

    def release(self) -> None:
        if self.trace is not None:
            ttnn.release_trace(self.mesh, self.trace)
            self.trace = None
        for tensor in (*self.frame.values(), *self.rotary, *self.header, self.mask_ids, self.one):
            ttnn.deallocate(tensor)
        self.frame.clear()
