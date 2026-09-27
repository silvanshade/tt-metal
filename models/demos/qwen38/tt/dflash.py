# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DFlash2 block drafter on the target's tensor-parallel mesh.

The drafter reads five target layers' hidden states as context and proposes one block: the anchor
token followed by mask tokens, all positions attending to each other and to context within a
sliding window. Weights are column/row parallel like the target; the residual stream is fractured
on hidden, norms gather it. Context keys and values live in a per-layer ring of positions.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open

import ttnn
from models.demos.qwen38.tt import tp_common as tpc
from models.tt_transformers.tt.ccl import tt_all_reduce

if TYPE_CHECKING:
    from models.demos.qwen38.tt.model import Qwen38Model

_TILE = ttnn.Tile([32, 32])


def _ring_write(source: ttnn.Tensor, ring: ttnn.Tensor, base: ttnn.Tensor) -> None:
    """Write source rows into ring rows base, base+1, ... (mod ring length).

    requires: BF16 tiled L1 source [1, H, rows<=32, W]; BF16 tiled ring [1, H, R, W], R % 32 == 0;
        base an FP32 tiled scalar holding a nonnegative integer.
    ensures: only the addressed ring rows change.
    """
    _, heads, rows, width = source.shape
    ring_rows = ring.shape[2]
    assert rows <= 32 and ring_rows % 32 == 0 and width % 32 == 0 and ring.shape[1] == heads
    assert source.memory_config().buffer_type == ttnn.BufferType.L1
    device = ring.device()
    grid = device.compute_with_storage_grid_size()
    coordinates = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(heads)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coordinates])
    args = ttnn.RuntimeArgs()
    for head, core in enumerate(coordinates):
        args[core.x][core.y] = [head, base.buffer_address(), source.buffer_address(), ring.buffer_address()]
    compile_args = [rows, ring_rows, width // 32]
    for tensor in (base, source, ring):
        compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    scratch = 64 + (width // 32) * 2048
    descriptor = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=str(Path(__file__).with_name("dflash_kernels") / "ring_write.cpp"),
                core_ranges=cores,
                compile_time_args=compile_args,
                runtime_args=args,
                config=ttnn.ReaderConfigDescriptor(),
            )
        ],
        cbs=[
            ttnn.CBDescriptor(
                total_size=(scratch + 4095) // 4096 * 4096,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=0, data_format=ttnn.float32, page_size=4096, tile=ttnn.TileDescriptor(_TILE)
                    )
                ],
            )
        ],
        semaphores=[],
    )
    ttnn.generic_op([base, source, ring], descriptor)


class Qwen38DFlash:
    """DFlash2 drafter sharing the target mesh, embedding, LM head and collectives.

    # Specification
    - requires: a TP-path target on a (1, N) mesh whose hidden size matches the checkpoint.
    - ensures: target weights and state are untouched; draft weights, rotary tables and context
      rings are owned here.
    - provides: context extension from target taps, block hidden states, top-k candidates.
    - fails: unsupported checkpoint configurations raise ValueError.
    """

    def __init__(
        self,
        target: "Qwen38Model",
        checkpoint: Path,
        max_positions: int,
        weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        matmul_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2,
    ) -> None:
        config = json.loads((checkpoint / "config.json").read_text())
        draft = config["dflash_config"]
        self.block = int(draft["block_size"])
        self.mask_token = int(draft["mask_token_id"])
        self.top_k = int(draft["selector_top_k"])
        self.taps = tuple(int(i) for i in draft["target_layer_ids"])
        self.window = int(config["sliding_window"])
        self.heads, self.kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
        self.head_dim, self.hidden = config["head_dim"], config["hidden_size"]
        self.eps = config["rms_norm_eps"]
        group, kernel = int(draft["conv_group_size"]), int(draft["conv_kernel_size"])
        types = set(config["layer_types"])
        if (
            self.block != 8
            or kernel != 2
            or types != {"sliding_attention"}
            or config.get("is_causal", True)
            or self.hidden != target.args.dim
            or config["rope_parameters"].get("rope_type", "default") != "default"
        ):
            raise ValueError("unsupported DFlash2 configuration")
        mesh = target.mesh_device
        self.tp = mesh.get_num_devices()
        if self.heads % self.tp or self.kv_heads % self.tp or self.hidden % (self.tp * 32):
            raise ValueError("DFlash2 heads and hidden must split across the mesh")
        self.target, self.mesh = target, mesh

        def compute(fidelity: ttnn.MathFidelity) -> ttnn.DeviceComputeKernelConfig:
            return ttnn.init_device_compute_kernel_config(
                mesh.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
            )

        # Norms and softmax at default precision triple the hidden error against the fp32
        # reference (8% vs 2.3%); they are cheap at eight rows, so they always run at HiFi4.
        self.compute, self.precise = compute(matmul_fidelity), compute(ttnn.MathFidelity.HiFi4)
        self.groups = self.hidden // group
        self.ring_rows = self.window + 64
        self.local_heads, self.local_kv = self.heads // self.tp, self.kv_heads // self.tp
        weights = {}
        with safe_open(str(checkpoint / "model.safetensors"), framework="pt") as f:
            for name in f.keys():
                weights[name] = f.get_tensor(name)
        self.layer_count = config["num_hidden_layers"]
        rep = ttnn.ReplicateTensorToMesh(mesh)

        def dram(tensor: torch.Tensor, dtype, mapper=rep, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
            return ttnn.from_torch(
                tensor, dtype=dtype, layout=layout, device=mesh, mesh_mapper=mapper, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        def column(name):
            return tpc.shard_w(weights[name], mesh, -1, ttnn.DRAM_MEMORY_CONFIG, None, dtype=weight_dtype)

        def row(name):
            return tpc.shard_w(weights[name], mesh, 0, ttnn.DRAM_MEMORY_CONFIG, None, dtype=weight_dtype)

        def vector(tensor: torch.Tensor) -> ttnn.Tensor:
            return dram(tensor.float().reshape(1, 1, 1, -1), ttnn.bfloat16)

        # Target taps stay fractured: each device holds its local hidden slice of every tap, in tap
        # order. Row-parallel fc input rows follow [device][tap][local hidden].
        local = self.hidden // self.tp
        order = [
            tap * self.hidden + device * local + k
            for device in range(self.tp)
            for tap in range(len(self.taps))
            for k in range(local)
        ]
        self.fc = tpc.shard_w(weights["fc.weight"][:, order], mesh, 0, ttnn.DRAM_MEMORY_CONFIG, None, dtype=weight_dtype)
        self.hidden_norm = vector(weights["hidden_norm.weight"])
        self.norm = vector(weights["norm.weight"])
        self.hidden_projection = weights["candidate_selector.hidden_projection.weight"].float()
        self.predecessor = weights["candidate_selector.predecessor_codebook"]
        self.successor = weights["candidate_selector.successor_codebook"]
        expand = torch.zeros(self.groups, self.hidden)
        expand[torch.arange(self.hidden) // group, torch.arange(self.hidden)] = 1.0
        self.expand = dram(expand.reshape(1, 1, self.groups, self.hidden), ttnn.bfloat8_b)
        shift = torch.zeros(self.block, self.block)
        shift[torch.arange(1, self.block), torch.arange(self.block - 1)] = 1.0
        self.shift = dram(shift.reshape(1, 1, self.block, self.block), ttnn.bfloat16)
        half = self.head_dim // 2
        rotate = torch.zeros(self.head_dim, self.head_dim)
        rotate[torch.arange(half) + half, torch.arange(half)] = -1.0
        rotate[torch.arange(half), torch.arange(half) + half] = 1.0
        self.rotate = dram(rotate.reshape(1, 1, self.head_dim, self.head_dim), ttnn.bfloat16)
        theta = float(config["rope_parameters"]["rope_theta"])
        inv = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float64) / self.head_dim))
        angles = torch.arange(max_positions, dtype=torch.float64)[:, None] * inv[None, :]
        angles = torch.cat([angles, angles], dim=-1)
        self.rotary = tuple(
            dram(table.float().to(torch.bfloat16), ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            for table in (angles.cos(), angles.sin())
        )
        block_mask = torch.full((1, 1, 32, 32), float("-inf"))
        block_mask[..., : self.block] = 0.0
        self.block_mask = dram(block_mask, ttnn.float32)
        # Device-resident round inputs derive from one FP32 start scalar and the origin scalar.
        self.slot_index = dram(torch.arange(self.ring_rows, dtype=torch.float32).reshape(1, 1, 1, -1), ttnn.float32)
        reach = self.window - 1 - (torch.arange(32) % self.block)
        self.reach = dram(reach.float().reshape(1, 1, 32, 1), ttnn.float32)
        self.offsets = dram(torch.arange(self.block, dtype=torch.float32).reshape(1, 1, -1, 1), ttnn.float32)
        self.origin_scalar = dram(torch.zeros(1, 1, 1, 1), ttnn.float32)
        self.layers = []
        for i in range(self.layer_count):
            p = f"layers.{i}"
            layer = {
                "input_norm": vector(weights[f"{p}.input_layernorm.weight"]),
                "post_norm": vector(weights[f"{p}.post_attention_layernorm.weight"]),
                "q": column(f"{p}.self_attn.q_proj.weight"),
                "k": column(f"{p}.self_attn.k_proj.weight"),
                "v": column(f"{p}.self_attn.v_proj.weight"),
                "o": row(f"{p}.self_attn.o_proj.weight"),
                "q_norm": vector(weights[f"{p}.self_attn.q_norm.weight"]),
                "k_norm": vector(weights[f"{p}.self_attn.k_norm.weight"]),
                "gate": column(f"{p}.mlp.gate_proj.weight"),
                "up": column(f"{p}.mlp.up_proj.weight"),
                "down": row(f"{p}.mlp.down_proj.weight"),
            }
            for conv in ("attention_conv", "mlp_conv"):
                layer[conv] = {
                    "projection": dram(
                        weights[f"{p}.{conv}.kernel_projection.weight"].T.contiguous().reshape(1, 1, self.hidden, -1),
                        ttnn.bfloat16,
                    ),
                    "base": [[vector(weights[f"{p}.{conv}.base_kernel"][h, t]) for t in range(kernel)] for h in range(2)],
                }
            zeros = torch.zeros(1, self.kv_heads, self.ring_rows, self.head_dim)
            layer["ring"] = tuple(
                dram(zeros, ttnn.bfloat16, mapper=ttnn.ShardTensorToMesh(mesh, dim=1)) for _ in range(2)
            )
            self.layers.append(layer)
        self.origin = 0

    # ---------------------------------------------------------------- helpers

    def _linear(self, x: ttnn.Tensor, weight: ttnn.Tensor, **kwargs) -> ttnn.Tensor:
        return ttnn.linear(x, weight, compute_kernel_config=self.compute, **kwargs)

    def _matmul(self, a: ttnn.Tensor, b: ttnn.Tensor, **kwargs) -> ttnn.Tensor:
        return ttnn.matmul(a, b, compute_kernel_config=self.compute, **kwargs)

    def _norm(self, fractured: ttnn.Tensor, weight: ttnn.Tensor) -> ttnn.Tensor:
        """Gather the fractured residual and apply checkpoint RMSNorm (gamma = weight)."""
        # The residual outlives the norm, so gather directly (tt_all_gather frees its input).
        full = ttnn.experimental.all_gather_async(
            fractured,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.target.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=self.target.tt_ccl.get_num_links(1),
            topology=self.target.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=self.target.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        out = ttnn.rms_norm(full, weight=weight, epsilon=self.eps, compute_kernel_config=self.precise)
        ttnn.deallocate(full)
        return out

    def _reduce(self, partial: ttnn.Tensor) -> ttnn.Tensor:
        """Row-parallel partial sums -> residual fractured on hidden."""
        return tt_all_reduce(
            partial,
            self.mesh,
            self.target.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.target.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _rope(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """x [1, rows, heads, D] with cos/sin [1, rows, 1, D]: x*cos + rotate_half(x)*sin."""
        rotated = self._matmul(x, self.rotate)
        out = ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rotated, sin))
        ttnn.deallocate(rotated)
        return out

    def _rotations(self, positions: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """positions: uint32 row-major [rows, 1] -> cos/sin [1, rows, 1, D]."""
        rows = positions.shape[0]
        return tuple(
            ttnn.reshape(ttnn.embedding(positions, table, layout=ttnn.TILE_LAYOUT), (1, rows, 1, self.head_dim))
            for table in self.rotary
        )

    def _conv(self, x: ttnn.Tensor, dynamic: ttnn.Tensor, conv: dict, half: int) -> ttnn.Tensor:
        """Grouped dynamic causal convolution over the block rows (kernel two)."""
        coefficients = []
        for tap in range(2):
            start = (half * 2 + tap) * self.groups
            rows = ttnn.slice(dynamic, (0, 0, 0, start), (1, 1, dynamic.shape[2], start + self.groups))
            expanded = self._matmul(rows, self.expand)
            ttnn.deallocate(rows)
            coefficients.append(ttnn.add(expanded, conv["base"][half][tap]))
            ttnn.deallocate(expanded)
        previous = self._matmul(self.shift, x)
        out = ttnn.add(ttnn.multiply(coefficients[0], x), ttnn.multiply(coefficients[1], previous))
        for tensor in (*coefficients, previous):
            ttnn.deallocate(tensor)
        return out

    def _heads(self, projected: ttnn.Tensor, heads: int, norm: ttnn.Tensor, cos, sin) -> ttnn.Tensor:
        """[1,1,rows,heads*D] -> normalized, rotated [1, heads, rows, D]."""
        rows = projected.shape[2]
        x = ttnn.reshape(projected, (1, rows, heads, self.head_dim))
        x = ttnn.rms_norm(x, weight=norm, epsilon=self.eps, compute_kernel_config=self.precise)
        x = self._rope(x, cos, sin)
        return ttnn.permute(x, (0, 2, 1, 3))

    # ---------------------------------------------------------------- context

    def extend(self, taps: ttnn.Tensor, positions: ttnn.Tensor, base: ttnn.Tensor) -> None:
        """Append context rows: fractured taps [1,1,rows<=32,taps*hidden/tp] (per device, its local
        slice of each tap in tap order) at `positions`, written to ring rows base, base+1, ...
        (base = first position minus origin, mod ring)."""
        context = self._reduce(self._linear(taps, self.fc))
        full = self._norm(context, self.hidden_norm)
        ttnn.deallocate(context)
        cos, sin = self._rotations(positions)
        for layer in self.layers:
            keys = self._heads(self._linear(full, layer["k"]), self.local_kv, layer["k_norm"], cos, sin)
            values = ttnn.permute(
                ttnn.reshape(self._linear(full, layer["v"]), (1, taps.shape[2], self.local_kv, self.head_dim)),
                (0, 2, 1, 3),
            )
            for source, ring in zip((keys, values), layer["ring"], strict=True):
                staged = ttnn.to_memory_config(source, ttnn.L1_MEMORY_CONFIG)
                _ring_write(staged, ring, base)
                ttnn.deallocate(staged)
                ttnn.deallocate(source)
        ttnn.deallocate(full)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)

    def reset(self, origin: int) -> None:
        """Forget all context; position `origin` maps to ring row zero. Outside any trace."""
        self.origin = origin
        host = ttnn.from_torch(torch.full((1, 1, 1, 1), float(origin)), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, self.origin_scalar)
        for layer in self.layers:
            for ring in layer["ring"]:
                zeros = ttnn.from_torch(
                    torch.zeros(tuple(ring.shape)),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                )
                ttnn.copy(zeros, ring)
                ttnn.deallocate(zeros)

    def append(self, taps: torch.Tensor, first: int) -> None:
        """Extend context from host taps [rows, taps*hidden] (checkpoint order) at positions
        first..first+rows-1, 32 rows per device call. Outside any trace."""
        local = self.hidden // self.tp
        fractured = torch.cat(
            [
                taps[:, j * self.hidden + d * local : j * self.hidden + (d + 1) * local]
                for d in range(self.tp)
                for j in range(len(self.taps))
            ],
            dim=-1,
        )
        for start in range(0, taps.shape[0], 32):
            chunk = fractured[start : start + 32]
            n = chunk.shape[0]
            device_taps = ttnn.from_torch(
                chunk.reshape(1, 1, n, -1).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=3),
            )
            positions = self.positions(first + start, n)
            base = self.scalar(self.ring_base(first + start))
            self.extend(device_taps, positions, base)
            for tensor in (device_taps, positions, base):
                ttnn.deallocate(tensor)

    # ---------------------------------------------------------------- inputs

    def positions(self, first: int, rows: int) -> ttnn.Tensor:
        return ttnn.from_torch(
            torch.arange(first, first + rows, dtype=torch.int32).reshape(rows, 1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def scalar(self, value: float) -> ttnn.Tensor:
        return ttnn.from_torch(
            torch.full((1, 1, 1, 1), float(value)),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def ring_base(self, position: int) -> int:
        return (position - self.origin) % self.ring_rows

    def ring_mask(self, start: int) -> torch.Tensor:
        """Additive mask [1,1,32,R] for a block at `start`: row r is block position r % 8."""
        slots = torch.arange(self.ring_rows)
        m = torch.remainder(start - 1 - self.origin - slots, self.ring_rows)
        offsets = (torch.arange(32) % self.block)[:, None]
        visible = (m[None, :] < self.window - 1 - offsets) & (m[None, :] <= start - 1)
        mask = torch.where(visible, 0.0, float("-inf"))
        return mask.reshape(1, 1, 32, self.ring_rows)

    # ------------------------------------------------- device-resident inputs (traceable)

    def _modulo_ring(self, value: ttnn.Tensor) -> ttnn.Tensor:
        """Integer-valued FP32 -> value mod ring rows; the half-row bias keeps floor exact."""
        wraps = ttnn.floor(ttnn.multiply(ttnn.add(value, 0.5), 1.0 / self.ring_rows))
        return ttnn.subtract(value, ttnn.multiply(wraps, float(self.ring_rows)))

    def device_positions(self, start: ttnn.Tensor) -> ttnn.Tensor:
        """FP32 scalar start -> uint32 row-major positions [block, 1]."""
        absolute = ttnn.add(self.offsets, start)
        return ttnn.reshape(ttnn.to_layout(ttnn.typecast(absolute, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), (self.block, 1))

    def device_base(self, start: ttnn.Tensor) -> ttnn.Tensor:
        """FP32 scalar start -> FP32 scalar ring row of position start."""
        return self._modulo_ring(ttnn.subtract(start, self.origin_scalar))

    def device_mask(self, start: ttnn.Tensor) -> ttnn.Tensor:
        """FP32 scalar start -> additive ring mask [1,1,32,R]: ring_mask() computed on device."""
        last = ttnn.subtract(start, 1.0)
        distance = self._modulo_ring(ttnn.subtract(ttnn.subtract(last, self.origin_scalar), self.slot_index))
        visible = ttnn.logical_and(ttnn.lt(distance, self.reach), ttnn.le(distance, last))
        return ttnn.multiply(ttnn.subtract(visible, 1.0), 1e30)

    # ---------------------------------------------------------------- block

    def draft(self, token_ids: ttnn.Tensor, positions: ttnn.Tensor, ring_mask: ttnn.Tensor) -> ttnn.Tensor:
        """Block hidden states [1,1,8,hidden], replicated and final-normalized.

        token_ids: uint32 row-major [8,1] (anchor then mask tokens); positions: uint32 [8,1];
        ring_mask: FP32 [1,1,32,R] from ring_mask().
        """
        rows = self.block
        h = self.target.embd(token_ids)
        h = ttnn.reshape(h, (1, 1, h.shape[0] * h.shape[1], h.shape[-1]))
        cos, sin = self._rotations(positions)
        scale = self.head_dim**-0.5
        for layer in self.layers:
            x = self._norm(h, layer["input_norm"])
            conv = layer["attention_conv"]
            dynamic = self._linear(x, conv["projection"])
            x_conv = self._conv(x, dynamic, conv, 0)
            ttnn.deallocate(x)
            q = self._heads(self._linear(x_conv, layer["q"]), self.local_heads, layer["q_norm"], cos, sin)
            k = self._heads(self._linear(x_conv, layer["k"]), self.local_kv, layer["k_norm"], cos, sin)
            v = ttnn.permute(
                ttnn.reshape(self._linear(x_conv, layer["v"]), (1, rows, self.local_kv, self.head_dim)), (0, 2, 1, 3)
            )
            ttnn.deallocate(x_conv)
            per_group = self.local_heads // self.local_kv
            q = ttnn.reshape(q, (1, self.local_kv, per_group * rows, self.head_dim))
            k = ttnn.pad(k, [(0, 0), (0, 0), (0, 32 - rows), (0, 0)], 0.0)
            v = ttnn.pad(v, [(0, 0), (0, 0), (0, 32 - rows), (0, 0)], 0.0)
            ring_k, ring_v = layer["ring"]
            ring_scores = self._matmul(q, ring_k, transpose_b=True, dtype=ttnn.float32)
            block_scores = self._matmul(q, k, transpose_b=True, dtype=ttnn.float32)
            scores = ttnn.concat(
                [
                    ttnn.add(ttnn.multiply(ring_scores, scale), ring_mask),
                    ttnn.add(ttnn.multiply(block_scores, scale), self.block_mask),
                ],
                dim=-1,
            )
            probabilities = ttnn.typecast(
                ttnn.softmax(scores, dim=-1, numeric_stable=True, compute_kernel_config=self.precise), ttnn.bfloat16
            )
            ring_p = ttnn.slice(probabilities, (0, 0, 0, 0), (1, self.local_kv, 32, self.ring_rows))
            block_p = ttnn.slice(probabilities, (0, 0, 0, self.ring_rows), (1, self.local_kv, 32, self.ring_rows + 32))
            attended = ttnn.add(self._matmul(ring_p, ring_v), self._matmul(block_p, v))
            for tensor in (q, k, v, ring_scores, block_scores, scores, probabilities, ring_p, block_p):
                ttnn.deallocate(tensor)
            attended = ttnn.reshape(attended, (1, self.local_heads, rows, self.head_dim))
            attended = ttnn.reshape(
                ttnn.permute(attended, (0, 2, 1, 3)), (1, 1, rows, self.local_heads * self.head_dim)
            )
            partial = self._linear(attended, layer["o"])
            ttnn.deallocate(attended)
            finished = self._conv(partial, dynamic, conv, 1)
            ttnn.deallocate(partial)
            ttnn.deallocate(dynamic)
            out = self._reduce(finished)
            h = ttnn.add(h, out)
            ttnn.deallocate(out)

            x = self._norm(h, layer["post_norm"])
            conv = layer["mlp_conv"]
            dynamic = self._linear(x, conv["projection"])
            x_conv = self._conv(x, dynamic, conv, 0)
            ttnn.deallocate(x)
            gated = ttnn.multiply(ttnn.silu(self._linear(x_conv, layer["gate"])), self._linear(x_conv, layer["up"]))
            ttnn.deallocate(x_conv)
            partial = self._linear(gated, layer["down"])
            ttnn.deallocate(gated)
            finished = self._conv(partial, dynamic, conv, 1)
            ttnn.deallocate(partial)
            ttnn.deallocate(dynamic)
            out = self._reduce(finished)
            h = ttnn.add(h, out)
            ttnn.deallocate(out)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        out = self._norm(h, self.norm)
        ttnn.deallocate(h)
        return out

    def candidates(self, hidden: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Top-k logits and token ids per block row from the target LM head: [1,1,8,k] each."""
        logits = self.target._lm_head(hidden)
        valid = logits[..., : self.target.vocab_size]
        values, indices = ttnn.topk(valid, k=self.top_k, dim=-1)
        ttnn.deallocate(logits)
        return values, indices

    def select(self, hidden: torch.Tensor, values: torch.Tensor, indices: torch.Tensor, anchor: int) -> list[int]:
        """Greedy candidate-selector walk on host: hidden [7,hidden], values/indices [7,k]."""
        projected = hidden.float() @ self.hidden_projection.T
        predecessor, path = anchor, []
        for p in range(hidden.shape[0]):
            candidates = indices[p].long()
            scores = values[p].float() + self.successor[candidates].float() @ (
                self.predecessor[predecessor].float() * projected[p]
            )
            predecessor = int(candidates[int(torch.argmax(scores))])
            path.append(predecessor)
        return path

    def propose(self, anchor: int, start: int) -> list[int]:
        """Eager proposal of block - 1 tokens following `anchor` at position `start`.

        requires: context holds every position before `start` that the window reaches.
        """
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        token_ids = ttnn.from_torch(
            torch.tensor([anchor] + [self.mask_token] * (self.block - 1), dtype=torch.int32).reshape(self.block, 1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=rep,
        )
        positions = self.positions(start, self.block)
        mask = ttnn.from_torch(
            self.ring_mask(start), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh, mesh_mapper=rep
        )
        hidden = self.draft(token_ids, positions, mask)
        values, indices = self.candidates(hidden)

        def rows(tensor: ttnn.Tensor) -> torch.Tensor:
            host = ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])
            return host.reshape(-1, tensor.shape[-1])[1 : self.block]

        proposal = self.select(rows(hidden).float(), rows(values).float(), rows(indices).long(), anchor)
        for tensor in (token_ids, positions, mask, hidden, values, indices):
            ttnn.deallocate(tensor)
        return proposal

    def observe(self, taps: ttnn.Tensor, start: int) -> None:
        """Extend context with every verified row's device taps at positions start.. (rows past
        the accepted prefix stay masked until a later block overwrites them)."""
        positions = self.positions(start, taps.shape[2])
        base = self.scalar(self.ring_base(start))
        self.extend(taps, positions, base)
        ttnn.deallocate(positions)
        ttnn.deallocate(base)
