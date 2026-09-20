# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Captured MTP rounds with device-resident drafting and accepted-state selection."""

import torch

import ttnn
from models.demos.qwen38.tt.mtp import Qwen38MTP


class Qwen38MTPRound:
    """Own fixed-address frames for greedy rounds and host-policy target boundaries.

    requires: target and draft caches allocated; construction and warmup precede all
        trace capture. One round runs at a time across request slots.
    ensures: greedy rounds export only token IDs and counts; target hidden rows,
        accepted-prefix replay, teacher alignment and recursive drafting stay on device.
        Host-policy rounds expose target logits once, then consume count/correction.
    hypothesis: sequential target continuation after every rejection prefix; seeded
        policy sampling consumes no discarded rows; slot reuse preserves isolation.
    """

    def __init__(self, mtp: Qwen38MTP, page_width: int) -> None:
        self.mtp = mtp
        self.target = mtp.target
        self.mesh = self.target.mesh_device
        self.verifier = mtp.verifier
        assert self.verifier is not None
        self.drafts = mtp.max_verify_tokens - 1
        assert self.drafts > 0
        self.mapper = ttnn.ReplicateTensorToMesh(self.mesh)
        self.composer = ttnn.ConcatMeshToTensor(self.mesh, dim=0)
        self.frames = {}
        self.traces = {}
        self.pending = None
        # Embedding lookup consumes row-major weights. Convert once, never per token.
        self.rotary = tuple(
            self._tensor(table.to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
            for table in (self.target.rope.cos_cpu, self.target.rope.sin_cpu)
        )
        for count in range(1, self.drafts + 2):
            self.frames[count] = {
                "ids": self._tensor(torch.zeros(count, 1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                "pages": self._tensor(
                    torch.zeros(count, page_width, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
                ),
                "base": self._scalar(1),
                "delta": self._scalar(0),
                "budget": self._scalar(self.drafts + 2),
                "accepted": self._scalar(1),
                "accepted_egress": self._tensor(
                    torch.zeros(1, 1, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
                ),
                "correction": self._scalar(0),
                "extent": self._scalar(self.drafts),
                "offsets": self._tensor(torch.arange(count).reshape(1, 1, count, 1).float(), ttnn.float32),
                "previous": self._tensor(torch.zeros(1, 1, 1, self.target.args.dim), ttnn.bfloat16),
                "hidden": self._tensor(torch.zeros(1, 1, count, self.target.args.dim), ttnn.bfloat16),
                "tokens": self._tensor(torch.zeros(count, 1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                "drafts": self._tensor(
                    torch.zeros(self.drafts, 1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
                ),
                "egress": self._tensor(
                    torch.zeros(self.drafts + 3, 1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
                ),
            }

    def _tensor(self, value: torch.Tensor, dtype: ttnn.DataType, layout: ttnn.Layout = ttnn.TILE_LAYOUT) -> ttnn.Tensor:
        return ttnn.from_torch(value, dtype=dtype, layout=layout, device=self.mesh, mesh_mapper=self.mapper)

    def _scalar(self, value: int) -> ttnn.Tensor:
        return self._tensor(torch.full((1, 1, 1, 1), value, dtype=torch.float32), ttnn.float32)

    @staticmethod
    def _float(tensor: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.typecast(ttnn.to_layout(tensor, ttnn.TILE_LAYOUT), ttnn.float32)

    @staticmethod
    def _indices(tensor: ttnn.Tensor, shape: tuple[int, ...], dtype: ttnn.DataType = ttnn.uint32) -> ttnn.Tensor:
        return ttnn.reshape(ttnn.to_layout(ttnn.typecast(tensor, dtype), ttnn.ROW_MAJOR_LAYOUT), shape)

    @staticmethod
    def _row(tensor: ttnn.Tensor, index: int) -> ttnn.Tensor:
        return tensor if tensor.shape[-2] == 1 else tensor[:, :, index : index + 1, :]

    def _rotations(self, positions: ttnn.Tensor, delta: ttnn.Tensor, count: int) -> tuple[ttnn.Tensor, ...]:
        indices = self._indices(ttnn.add(positions, delta), (count, 1))
        return tuple(
            ttnn.reshape(
                ttnn.embedding(indices, table, layout=ttnn.TILE_LAYOUT), (1, count, 1, self.target.args.rope_head_dim)
            )
            for table in self.rotary
        )

    def _argmax(self, logits: ttnn.Tensor) -> ttnn.Tensor:
        # Padded vocabulary entries are not legal tokens, even if their logits win.
        valid = ttnn.to_layout(logits[..., : self.target.vocab_size], ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.reshape(ttnn.argmax(valid, dim=-1, keepdim=True), (logits.shape[-2], 1))

    def _verify(self, count: int, slot: int) -> None:
        frame = self.frames[count]
        positions = ttnn.add(frame["base"], frame["offsets"])
        cos, sin = self._rotations(positions, frame["delta"], count)
        self.verifier.prepare(slot, 0)
        logits, hidden = self.verifier.verify_prepared(
            frame["ids"], cos, sin, self._indices(positions, (count,), ttnn.int32), frame["pages"]
        )
        if "logits" not in frame:
            frame["logits"] = ttnn.clone(logits)
        ttnn.copy(logits, frame["logits"])
        ttnn.copy(hidden, frame["hidden"])
        ttnn.copy(self._argmax(logits), frame["tokens"])
        ttnn.deallocate(logits)
        ttnn.deallocate(hidden)

    def _accept(self, count: int) -> None:
        frame = self.frames[count]
        tokens = ttnn.reshape(self._float(frame["tokens"]), (1, 1, count, 1))
        inputs = ttnn.reshape(self._float(frame["ids"]), (1, 1, count, 1))
        accepted = ttnn.add(ttnn.multiply(frame["accepted"], 0), 1)
        prefix = ttnn.clone(accepted)
        correction = self._row(tokens, 0)
        for index in range(1, count):
            prefix = ttnn.multiply(prefix, ttnn.eq(self._row(inputs, index), self._row(tokens, index - 1)))
            accepted = ttnn.add(accepted, prefix)
            correction = ttnn.where(prefix, self._row(tokens, index), correction)
        ttnn.copy(accepted, frame["accepted"])
        ttnn.copy(self._indices(accepted, (1, 1), ttnn.int32), frame["accepted_egress"])
        ttnn.copy(correction, frame["correction"])

    def _propose(self, count: int, anchor: ttnn.Tensor, hidden: ttnn.Tensor, position: ttnn.Tensor) -> None:
        frame = self.frames[count]
        pages = frame["pages"] if count == 1 else frame["pages"][:1, :]
        rows = []
        current = anchor
        for step in range(self.drafts):
            active = ttnn.gt(frame["extent"], step)
            absolute = ttnn.add(position, step)
            # Invalid tail lanes neither write KV nor index outside the rotary table.
            safe_position = ttnn.where(active, absolute, 1)
            cos, sin = self._rotations(safe_position, frame["delta"], 1)
            cache_position = ttnn.where(active, ttnn.subtract(absolute, 1), -1)
            hidden = self.mtp.forward(
                self._indices(current, (1, 1)),
                hidden,
                cos,
                sin,
                self._indices(cache_position, (1,), ttnn.int32),
                pages,
            )
            logits = self.target._lm_head(hidden)
            token = self._argmax(logits)
            ttnn.deallocate(logits)
            rows.append(token)
            current = ttnn.reshape(self._float(token), (1, 1, 1, 1))
        ttnn.copy(ttnn.concat(rows, dim=0), frame["drafts"])

    def _finish(self, count: int, slot: int, *, full: bool) -> None:
        """Finalize the selected prefix; full acceptance reuses verified scratch.

        requires: pending verification; full implies every verifier input was accepted.
        ensures: partial acceptance retains device-masked repair; feedback and drafts
            follow the same accepted prefix on both captured paths.
        """

        frame = self.frames[count]
        accepted = frame["accepted"]
        for verifier in self.verifier.gdn.values():
            verifier.fold(count if full else accepted)
        self.verifier.count = 0
        self.verifier.slot = -1

        selected = self._row(frame["hidden"], 0)
        for index in range(1, count):
            active = ttnn.typecast(ttnn.eq(accepted, index + 1), selected.dtype)
            selected = ttnn.where(active, self._row(frame["hidden"], index), selected)
        preceding = (
            frame["previous"]
            if count == 1
            else ttnn.concat([frame["previous"], frame["hidden"][:, :, : count - 1, :]], dim=2)
        )
        positions = ttnn.add(frame["base"], frame["offsets"])
        cos, sin = self._rotations(positions, frame["delta"], count)
        active = ttnn.lt(frame["offsets"], accepted)
        cache_positions = ttnn.where(active, ttnn.subtract(positions, 1), -1)
        self.mtp.refresh(
            frame["ids"],
            preceding,
            cos,
            sin,
            self._indices(cache_positions, (count,), ttnn.int32),
            frame["pages"],
        )
        next_position = ttnn.add(frame["base"], accepted)
        extent = ttnn.minimum(ttnn.maximum(ttnn.subtract(frame["budget"], ttnn.add(accepted, 1)), 0), self.drafts)
        extent = ttnn.minimum(
            extent, ttnn.maximum(ttnn.add(ttnn.neg(next_position), self.target.args.max_seq_len - 1), 0)
        )
        ttnn.copy(extent, frame["extent"])
        self._propose(count, frame["correction"], selected, next_position)

        ttnn.copy(selected, frame["previous"])
        self._pack(frame)

    def _pack(self, frame: dict[str, ttnn.Tensor]) -> None:
        """Export one compact message: accepted count, correction, extent and drafts."""
        header = [self._indices(frame[name], (1, 1)) for name in ("accepted", "correction", "extent")]
        ttnn.copy(ttnn.concat([*header, frame["drafts"]], dim=0), frame["egress"])

    def warmup(self) -> None:
        """Compile every admitted count/slot before any trace retains addresses."""
        assert not self.traces and self.pending is None
        for count in self.frames:
            for slot in range(self.target.args.max_batch_size):
                self._verify(count, slot)
                self._accept(count)
                self._finish(count, slot, full=False)
                self.verifier.slot = slot
                self.verifier.record_replay(count)
                self._finish(count, slot, full=True)
        frame = self.frames[1]
        self._propose(1, frame["correction"], frame["previous"], frame["base"])

    def capture(self) -> None:
        """Share verification/finalization traces across greedy and host-policy rounds."""
        assert not self.traces and all("logits" in frame for frame in self.frames.values())
        for count in self.frames:
            for slot in range(self.target.args.max_batch_size):
                for phase, full in (("verify", False), ("finish", False), ("finish", True)):
                    if phase == "finish":
                        self.verifier.slot = slot
                        self.verifier.record_replay(count)
                    trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                    if phase == "verify":
                        self._verify(count, slot)
                        self._accept(count)
                    else:
                        self._finish(count, slot, full=full)
                    ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
                    self.traces[phase, count, slot, full] = trace
                    if phase == "verify":
                        # Capture executed the verifier and left its checkpoint pending.
                        # Finalization capture consumes that exact pending block.
                        self.verifier.count = 0
        frame = self.frames[1]
        trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self._propose(1, frame["correction"], frame["previous"], frame["base"])
        self._pack(frame)
        ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
        self.traces["bridge", 1, 0, False] = trace

    def stage(
        self, slot: int, tokens: list[int], position: int, pages: torch.Tensor, budget: int, rope_delta: int = 0
    ) -> None:
        """Upload request ingress into persistent buffers; retain feedback on device."""
        assert self.pending is None and 1 <= len(tokens) <= self.drafts + 1
        assert len(tokens) <= budget and position > 0
        count = len(tokens)
        frame = self.frames[count]
        assert position + count <= self.target.args.max_seq_len
        values = {
            "ids": torch.tensor(tokens, dtype=torch.int32).reshape(count, 1),
            "pages": pages.reshape(1, -1).repeat(count, 1).to(torch.int32),
            "base": torch.full((1, 1, 1, 1), position, dtype=torch.float32),
            "delta": torch.full((1, 1, 1, 1), rope_delta, dtype=torch.float32),
            "budget": torch.full((1, 1, 1, 1), budget, dtype=torch.float32),
        }
        for name, value in values.items():
            destination = frame[name]
            host = ttnn.from_torch(value, dtype=destination.dtype, layout=destination.layout, mesh_mapper=self.mapper)
            ttnn.copy_host_to_device_tensor(host, destination)
        ttnn.copy(self.mtp.previous_hidden[slot], frame["previous"])
        self.pending = count, slot

    def bridge(
        self, slot: int, anchor: int, position: int, pages: torch.Tensor, budget: int, rope_delta: int = 0
    ) -> dict[str, ttnn.Tensor]:
        """Draft after final prefill; the target anchor remains unprocessed."""
        self.stage(slot, [anchor], position, pages, budget, rope_delta)
        frame = self.frames[1]
        extent = min(self.drafts, budget - 1, self.target.args.max_seq_len - position - 1)
        for name, value in (("accepted", 0), ("correction", anchor), ("extent", extent)):
            host = ttnn.from_torch(
                torch.full((1, 1, 1, 1), value, dtype=torch.float32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
            )
            ttnn.copy_host_to_device_tensor(host, frame[name])
        ttnn.execute_trace(self.mesh, self.traces["bridge", 1, 0, False], cq_id=0, blocking=False)
        self.pending = None
        return frame

    def execute(self, *, greedy: bool) -> dict[str, ttnn.Tensor]:
        """Verify a staged round and choose finalization from its accepted count.

        requires: one staged request; no overlapping frame use.
        ensures: greedy execution synchronously reads one int32 count before
            submitting full or partial finalization; policy execution leaves
            finalization pending. Borrowed results survive until frame reuse.
        """
        assert self.pending is not None
        count, slot = self.pending
        frame = self.frames[count]
        ttnn.execute_trace(self.mesh, self.traces["verify", count, slot, False], cq_id=0, blocking=False)
        if greedy:
            accepted = int(ttnn.to_torch(frame["accepted_egress"], mesh_composer=self.composer).reshape(-1)[0].item())
            ttnn.execute_trace(
                self.mesh, self.traces["finish", count, slot, accepted == count], cq_id=0, blocking=False
            )
            ttnn.copy(frame["previous"], self.mtp.previous_hidden[slot])
            self.pending = None
        return frame

    def finish(self, accepted: int, correction: int) -> dict[str, ttnn.Tensor]:
        """Commit host-policy prefix and draft recursively without further host sampling."""
        assert self.pending is not None
        count, slot = self.pending
        assert 1 <= accepted <= count
        frame = self.frames[count]
        for name, value in (("accepted", accepted), ("correction", correction)):
            host = ttnn.from_torch(
                torch.full((1, 1, 1, 1), value, dtype=torch.float32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
            )
            ttnn.copy_host_to_device_tensor(host, frame[name])
        ttnn.execute_trace(self.mesh, self.traces["finish", count, slot, accepted == count], cq_id=0, blocking=False)
        ttnn.copy(frame["previous"], self.mtp.previous_hidden[slot])
        self.pending = None
        return frame

    def release(self) -> None:
        """Release caller-owned traces before frame storage and model cache teardown."""
        for trace in self.traces.values():
            ttnn.release_trace(self.mesh, trace)
        self.traces.clear()
        for frame in self.frames.values():
            for tensor in frame.values():
                ttnn.deallocate(tensor)
        self.frames.clear()
        for tensor in self.rotary:
            ttnn.deallocate(tensor)
        self.pending = None
