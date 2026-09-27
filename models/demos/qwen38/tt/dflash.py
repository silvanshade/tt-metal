# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DFlash2 block drafter on the target's tensor-parallel mesh.

The drafter reads five target layers' hidden states as context and proposes one block: the anchor
token followed by mask tokens, all positions attending to each other and to context within a
sliding window. Weights are column/row parallel like the target; the residual stream is fractured
on hidden, norms gather it. Context keys and values live in a per-layer ring of positions.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open

import ttnn
from models.demos.qwen38.tt import tp_common as tpc
from models.demos.qwen38.tt.greedy import shard_candidates
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.common import Mode

if TYPE_CHECKING:
    from models.demos.qwen38.tt.model import Qwen38Model

_TILE = ttnn.Tile([32, 32])


@dataclass(frozen=True)
class _Projection:
    """Matmul for at most 32 rows: per-device weight [K, N] width-sharded across DRAM banks,
    activation width-sharded in L1, DRAM-sharded program (the target's decode matmul path)."""

    weight: ttnn.Tensor
    program: ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
    activation: ttnn.MemoryConfig


def _ring_write(source: ttnn.Tensor, ring: ttnn.Tensor, base: ttnn.Tensor, modulo: int) -> None:
    """Write source rows into ring rows base, base+1, ... (mod `modulo`).

    requires: BF16 tiled L1 source [1, H, rows<=32, W]; BF16 tiled ring [1, H, L, W], L % 32 == 0;
        modulo a multiple of 32 no larger than L; base an FP32 tiled scalar holding a nonnegative
        integer.
    ensures: only the addressed ring rows change.
    """
    _, heads, rows, width = source.shape
    length = ring.shape[2]
    assert rows <= 32 and length % 32 == 0 and modulo % 32 == 0 and modulo <= length
    assert width % 32 == 0 and ring.shape[1] == heads
    assert source.memory_config().buffer_type == ttnn.BufferType.L1
    device = ring.device()
    grid = device.compute_with_storage_grid_size()
    coordinates = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(heads)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coordinates])
    args = ttnn.RuntimeArgs()
    for head, core in enumerate(coordinates):
        args[core.x][core.y] = [head, base.buffer_address(), source.buffer_address(), ring.buffer_address()]
    compile_args = [rows, modulo, length // 32, width // 32]
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
        block: int | None = None,
    ) -> None:
        """block: drafted rows per round (anchor included); default the checkpoint's block size.
        A wider block than the checkpoint's drafts the extra positions with the same model."""
        config = json.loads((checkpoint / "config.json").read_text())
        draft = config["dflash_config"]
        self.block = int(draft["block_size"]) if block is None else block
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
            not 2 <= self.block <= 32
            or kernel != 2
            or types != {"sliding_attention"}
            or config.get("is_causal", True)
            or self.hidden != target.args.dim
            or config["rope_parameters"].get("rope_type", "default") != "default"
        ):
            raise ValueError("unsupported DFlash2 configuration")
        # The drafter runs `width` rows, the least power of two holding the block. Attention folds
        # each group of up to eight drafted rows into query heads (head h, row t -> query
        # h * group + t), one SDPA batch per group over the shared ring, so a core's query tile
        # stays the block-8 shape at any width. Padding rows follow the block; their keys are
        # masked from every row, so they leave the block's rows exact, and their outputs are dropped.
        self.width = next(rows for rows in (2, 4, 8, 16, 32) if rows >= self.block)
        self.group = min(self.width, 8)
        self.batches = self.width // self.group
        # Drafted row of each group's mask-tile row: group g, tile row r -> g * group + r % group.
        self.mask_rows = torch.arange(self.batches)[:, None] * self.group + torch.arange(32)[None, :] % self.group
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
        # Each ring tensor holds the context ring, then a block slot at row ring_rows for the
        # drafted block's K/V, padded to whole attention chunks.
        self.attention_chunk = 128
        self.ring_length = -(-(self.ring_rows + 32) // self.attention_chunk) * self.attention_chunk
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

        workers = target.args.dram_sharded_workers

        def projection(tensor: torch.Tensor, dim: int | None, dtype: ttnn.DataType = weight_dtype) -> _Projection:
            """Checkpoint weight [out, in]: dim -1 column-parallel, 0 row-parallel, None replicated."""
            n, k = tensor.shape
            k = k // self.tp if dim == 0 else k
            n = n // self.tp if dim == -1 else n
            weight = ttnn.as_tensor(
                tensor.to(torch.bfloat16).T.contiguous(),
                dtype=dtype,
                device=mesh,
                mesh_mapper=rep if dim is None else ttnn.ShardTensorToMesh(mesh, dim=dim),
                layout=ttnn.TILE_LAYOUT,
                memory_config=tpc.create_dram_sharded_mem_config(k, n),
            )
            program = tpc.create_dram_sharded_matmul_program_config(32, k, n, num_workers_per_dram_bank=workers)
            return _Projection(weight, program, tpc.create_activation_shard_config(k))

        def column(name: str) -> _Projection:
            return projection(weights[name], -1)

        def row(name: str) -> _Projection:
            return projection(weights[name], 0)

        def vector(tensor: torch.Tensor) -> ttnn.Tensor:
            return dram(tensor.float().reshape(1, 1, 1, -1), ttnn.bfloat16)

        def gamma(tensor: torch.Tensor) -> ttnn.Tensor:
            """Hidden-width norm weight in the row-major [1, 1, hidden/32, 32] sharded-norm layout."""
            return dram(tensor.float().reshape(1, 1, -1, 32), ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        # Residual norms use the target's decode layout: gather straight into a width-sharded L1
        # activation, one sharded norm program across its cores.
        norm_config = target.args.get_norm_config("attn", Mode.DECODE)
        self.norm_program = norm_config["sharded_program_config"]
        self.norm_memory = norm_config["sharded_output_config"]

        # Target taps stay fractured: each device holds its local hidden slice of every tap, in tap
        # order. Row-parallel fc input rows follow [device][tap][local hidden].
        local = self.hidden // self.tp
        order = [
            tap * self.hidden + device * local + k
            for device in range(self.tp)
            for tap in range(len(self.taps))
            for k in range(local)
        ]
        self.fc = projection(weights["fc.weight"][:, order], 0)
        self.hidden_norm = gamma(weights["hidden_norm.weight"])
        self.norm = gamma(weights["norm.weight"])
        # Selector scores compare near-tied candidates: BF16 weight, HiFi4, FP32 output.
        self.selector_projection = projection(
            weights["candidate_selector.hidden_projection.weight"], None, ttnn.bfloat16
        )
        self.selector_width = self.selector_projection.weight.shape[-1]
        # Candidates come from the target's vocab-sharded LM head without gathering logits: each
        # device takes the top_k of its shard (shard_candidates) into a tile-wide local_candidates
        # segment, and the host merges devices. The global top-k lies within the per-device top-k,
        # so the merge is exact.
        if self.tp > 1 and not target._lmhead_vocab_sharded:
            raise ValueError("DFlash2 candidates need a vocab-sharded target LM head")
        self.local_vocab = target.lm_head_weight.shape[-1]
        self.local_candidates = 32 * -(-self.top_k // 32)
        self.predecessor = weights["candidate_selector.predecessor_codebook"]
        self.successor = weights["candidate_selector.successor_codebook"]
        expand = torch.zeros(self.groups, self.hidden)
        expand[torch.arange(self.hidden) // group, torch.arange(self.hidden)] = 1.0
        self.expand = dram(expand.reshape(1, 1, self.groups, self.hidden), ttnn.bfloat8_b)
        shift = torch.zeros(self.width, self.width)
        shift[torch.arange(1, self.width), torch.arange(self.width - 1)] = 1.0
        self.shift = dram(shift.reshape(1, 1, self.width, self.width), ttnn.bfloat16)
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
        # Block slot columns: the block's own rows are visible to every block row, padding is not.
        tail = torch.full((self.batches, 1, 32, self.ring_length - self.ring_rows), float("-inf"))
        tail[..., : self.block] = 0.0
        self.block_tail = dram(tail, ttnn.float32)
        self.block_slot = dram(torch.full((1, 1, 1, 1), float(self.ring_rows)), ttnn.float32)
        attention_grid = mesh.compute_with_storage_grid_size()
        self.attention_program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(min(8, attention_grid.x), min(8, attention_grid.y)),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=self.attention_chunk,
        )
        # Device-resident round inputs derive from one FP32 start scalar and the origin scalar.
        self.slot_index = dram(torch.arange(self.ring_rows, dtype=torch.float32).reshape(1, 1, 1, -1), ttnn.float32)
        reach = self.window - 1 - self.mask_rows
        self.reach = tuple(dram(row.float().reshape(1, 1, 32, 1), ttnn.float32) for row in reach)
        self.offsets = dram(torch.arange(self.block, dtype=torch.float32).reshape(1, 1, -1, 1), ttnn.float32)
        self.draft_offsets = dram(torch.arange(self.width, dtype=torch.float32).reshape(1, 1, -1, 1), ttnn.float32)
        self.padding = (
            dram(
                torch.full((self.width - self.block, 1), self.mask_token, dtype=torch.int32),
                ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            if self.width > self.block
            else None
        )
        self.origin_scalar = dram(torch.zeros(1, 1, 1, 1), ttnn.float32)
        self.layers = []
        for i in range(self.layer_count):
            p = f"layers.{i}"
            layer = {
                "input_norm": gamma(weights[f"{p}.input_layernorm.weight"]),
                "post_norm": gamma(weights[f"{p}.post_attention_layernorm.weight"]),
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
                    # Dynamic conv coefficients: kept BF16 like the reference (replicated, 13 MB).
                    "projection": projection(weights[f"{p}.{conv}.kernel_projection.weight"], None, ttnn.bfloat16),
                    "base": [[vector(weights[f"{p}.{conv}.base_kernel"][h, t]) for t in range(kernel)] for h in range(2)],
                }
            zeros = torch.zeros(1, self.kv_heads, self.ring_length, self.head_dim)
            layer["ring"] = tuple(
                dram(zeros, ttnn.bfloat16, mapper=ttnn.ShardTensorToMesh(mesh, dim=1)) for _ in range(2)
            )
            self.layers.append(layer)
        self.origin = 0

    # ---------------------------------------------------------------- helpers

    def _project(
        self, x: ttnn.Tensor, projection: _Projection, *, precise: bool = False, dtype: ttnn.DataType | None = None
    ) -> ttnn.Tensor:
        """At most 32 rows [1,1,rows,K] -> DRAM-interleaved [1,1,rows,N]."""
        assert x.shape[-2] <= 32
        sharded = ttnn.to_memory_config(x, projection.activation)
        out = ttnn.linear(
            sharded,
            projection.weight,
            program_config=projection.program,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.precise if precise else self.compute,
            **({} if dtype is None else {"dtype": dtype}),
        )
        ttnn.deallocate(sharded)
        result = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        return result

    def _matmul(self, a: ttnn.Tensor, b: ttnn.Tensor, **kwargs) -> ttnn.Tensor:
        return ttnn.matmul(a, b, compute_kernel_config=self.compute, **kwargs)

    def _attend(self, queries: ttnn.Tensor, ring_k: ttnn.Tensor, ring_v: ttnn.Tensor, mask: ttnn.Tensor, scale: float):
        """Queries [1,1,heads*rows,D] over one layer's ring under an additive mask [1,1,heads*rows,L]."""
        return ttnn.transformer.scaled_dot_product_attention_decode(
            queries,
            ring_k,
            ring_v,
            is_causal=False,
            attn_mask=mask,
            scale=scale,
            program_config=self.attention_program,
            compute_kernel_config=self.precise,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

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
            memory_config=self.norm_memory,
            barrier_semaphore=self.target.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        out = ttnn.rms_norm(
            full,
            weight=weight,
            epsilon=self.eps,
            program_config=self.norm_program,
            memory_config=self.norm_memory,
            compute_kernel_config=self.precise,
        )
        ttnn.deallocate(full)
        interleaved = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        return interleaved

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
        context = self._reduce(self._project(taps, self.fc))
        full = self._norm(context, self.hidden_norm)
        ttnn.deallocate(context)
        cos, sin = self._rotations(positions)
        for layer in self.layers:
            keys = self._heads(self._project(full, layer["k"]), self.local_kv, layer["k_norm"], cos, sin)
            values = ttnn.permute(
                ttnn.reshape(self._project(full, layer["v"]), (1, taps.shape[2], self.local_kv, self.head_dim)),
                (0, 2, 1, 3),
            )
            for source, ring in zip((keys, values), layer["ring"], strict=True):
                staged = ttnn.to_memory_config(source, ttnn.L1_MEMORY_CONFIG)
                _ring_write(staged, ring, base, self.ring_rows)
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
        """Additive mask [groups,1,32,R] for a block at `start`: group g's row r is drafted row
        g * group + r % group."""
        slots = torch.arange(self.ring_rows)
        m = torch.remainder(start - 1 - self.origin - slots, self.ring_rows)
        offsets = self.mask_rows[:, :, None]
        visible = (m[None, None, :] < self.window - 1 - offsets) & (m[None, None, :] <= start - 1)
        mask = torch.where(visible, 0.0, float("-inf"))
        return mask.reshape(self.batches, 1, 32, self.ring_rows)

    # ------------------------------------------------- device-resident inputs (traceable)

    def _modulo_ring(self, value: ttnn.Tensor) -> ttnn.Tensor:
        """Integer-valued FP32 -> value mod ring rows; the half-row bias keeps floor exact."""
        wraps = ttnn.floor(ttnn.multiply(ttnn.add(value, 0.5), 1.0 / self.ring_rows))
        return ttnn.subtract(value, ttnn.multiply(wraps, float(self.ring_rows)))

    def device_positions(self, start: ttnn.Tensor, rows: int | None = None) -> ttnn.Tensor:
        """FP32 scalar start -> uint32 row-major positions [rows, 1]: `block` rows by default,
        `width` rows for draft()."""
        rows = self.block if rows is None else rows
        offsets = {self.block: self.offsets, self.width: self.draft_offsets}[rows]
        absolute = ttnn.add(offsets, start)
        return ttnn.reshape(ttnn.to_layout(ttnn.typecast(absolute, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), (rows, 1))

    def device_base(self, start: ttnn.Tensor) -> ttnn.Tensor:
        """FP32 scalar start -> FP32 scalar ring row of position start."""
        return self._modulo_ring(ttnn.subtract(start, self.origin_scalar))

    def device_mask(self, start: ttnn.Tensor) -> ttnn.Tensor:
        """FP32 scalar start -> additive ring mask [groups,1,32,R]: ring_mask() computed on device."""
        last = ttnn.subtract(start, 1.0)
        distance = self._modulo_ring(ttnn.subtract(ttnn.subtract(last, self.origin_scalar), self.slot_index))
        masks = [
            ttnn.multiply(ttnn.subtract(ttnn.logical_and(ttnn.lt(distance, reach), ttnn.le(distance, last)), 1.0), 1e30)
            for reach in self.reach
        ]
        return masks[0] if self.batches == 1 else ttnn.concat(masks, dim=0)

    # ---------------------------------------------------------------- block

    def draft(self, token_ids: ttnn.Tensor, positions: ttnn.Tensor, ring_mask: ttnn.Tensor) -> ttnn.Tensor:
        """Block hidden states [1,1,block,hidden], replicated and final-normalized.

        token_ids: uint32 row-major [block,1] (anchor then mask tokens); positions: uint32
        [width,1]; ring_mask: FP32 [groups,1,32,R] from ring_mask().

        Attention is one SDPA-decode call per layer: the drafted rows' K/V go to each ring's block
        slot, and in batch g query head h * group + t (row g * group + t) attends over ring and
        slot under that row's mask.
        """
        rows, group, groups = self.width, self.group, self.batches
        if self.padding is not None:
            token_ids = ttnn.concat([token_ids, self.padding], dim=0)
        h = self.target.embd(token_ids)
        if self.padding is not None:
            ttnn.deallocate(token_ids)
        h = ttnn.reshape(h, (1, 1, h.shape[0] * h.shape[1], h.shape[-1]))
        cos, sin = self._rotations(positions)
        scale = self.head_dim**-0.5
        block_mask = ttnn.typecast(ttnn.concat([ring_mask, self.block_tail], dim=-1), ttnn.bfloat16)
        mask = ttnn.concat(
            [block_mask] * (self.local_heads * group // 32), dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(block_mask)
        masks = (
            [mask]
            if groups == 1
            else [ttnn.slice(mask, (g, 0, 0, 0), (g + 1, mask.shape[1], mask.shape[2], mask.shape[3])) for g in range(groups)]
        )
        if groups > 1:
            ttnn.deallocate(mask)
        for layer in self.layers:
            x = self._norm(h, layer["input_norm"])
            conv = layer["attention_conv"]
            dynamic = self._project(x, conv["projection"])
            x_conv = self._conv(x, dynamic, conv, 0)
            ttnn.deallocate(x)
            q = self._heads(self._project(x_conv, layer["q"]), self.local_heads, layer["q_norm"], cos, sin)
            k = self._heads(self._project(x_conv, layer["k"]), self.local_kv, layer["k_norm"], cos, sin)
            v = ttnn.permute(
                ttnn.reshape(self._project(x_conv, layer["v"]), (1, rows, self.local_kv, self.head_dim)), (0, 2, 1, 3)
            )
            ttnn.deallocate(x_conv)
            for source, ring in zip((k, v), layer["ring"], strict=True):
                staged = ttnn.to_memory_config(source, ttnn.L1_MEMORY_CONFIG)
                _ring_write(staged, ring, self.block_slot, self.ring_length)
                ttnn.deallocate(staged)
                ttnn.deallocate(source)
            ring_k, ring_v = layer["ring"]
            if groups == 1:
                attended = self._attend(
                    ttnn.reshape(q, (1, 1, self.local_heads * rows, self.head_dim)), ring_k, ring_v, mask, scale
                )
            else:
                # One call per group: SDPA decode's output spec takes its batch from K, so a
                # shared-cache batch would be written past a one-batch output.
                width = self.local_heads * group
                grouped = ttnn.permute(ttnn.reshape(q, (self.local_heads, groups, group, self.head_dim)), (1, 0, 2, 3))
                grouped = ttnn.reshape(grouped, (groups, 1, width, self.head_dim))
                parts = []
                for g, group_mask in enumerate(masks):
                    queries = ttnn.slice(grouped, (g, 0, 0, 0), (g + 1, 1, width, self.head_dim))
                    parts.append(self._attend(queries, ring_k, ring_v, group_mask, scale))
                    ttnn.deallocate(queries)
                ttnn.deallocate(grouped)
                attended = ttnn.concat(parts, dim=0)
                for part in parts:
                    ttnn.deallocate(part)
                attended = ttnn.permute(
                    ttnn.reshape(attended, (groups, self.local_heads, group, self.head_dim)), (1, 0, 2, 3)
                )
            ttnn.deallocate(q)
            attended = ttnn.reshape(attended, (1, self.local_heads, rows, self.head_dim))
            attended = ttnn.reshape(
                ttnn.permute(attended, (0, 2, 1, 3)), (1, 1, rows, self.local_heads * self.head_dim)
            )
            partial = self._project(attended, layer["o"])
            ttnn.deallocate(attended)
            finished = self._conv(partial, dynamic, conv, 1)
            ttnn.deallocate(partial)
            ttnn.deallocate(dynamic)
            out = self._reduce(finished)
            h = ttnn.add(h, out)
            ttnn.deallocate(out)

            x = self._norm(h, layer["post_norm"])
            conv = layer["mlp_conv"]
            dynamic = self._project(x, conv["projection"])
            x_conv = self._conv(x, dynamic, conv, 0)
            ttnn.deallocate(x)
            gated = ttnn.multiply(ttnn.silu(self._project(x_conv, layer["gate"])), self._project(x_conv, layer["up"]))
            ttnn.deallocate(x_conv)
            partial = self._project(gated, layer["down"])
            ttnn.deallocate(gated)
            finished = self._conv(partial, dynamic, conv, 1)
            ttnn.deallocate(partial)
            ttnn.deallocate(dynamic)
            out = self._reduce(finished)
            h = ttnn.add(h, out)
            ttnn.deallocate(out)
        ttnn.deallocate(cos)
        for group_mask in masks:
            ttnn.deallocate(group_mask)
        ttnn.deallocate(sin)
        out = self._norm(h, self.norm)
        ttnn.deallocate(h)
        if self.padding is None:
            return out
        block = ttnn.slice(out, (0, 0, 0, 0), (1, 1, self.block, out.shape[-1]))
        ttnn.deallocate(out)
        return block

    def candidates(self, hidden: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Per block row: per device, the top_k logits of its vocabulary shard and their
        shard-local ids as FP32 [1,1,8,C] (then -inf values; merge() maps them to tokens), and the
        selector's FP32 hidden projection [1,1,8,P] (replicated)."""
        logits = ttnn.linear(hidden, self.target.lm_head_weight)
        values, indices = shard_candidates(logits, self.top_k, self.local_candidates)
        ttnn.deallocate(logits)
        projected = self._project(hidden, self.selector_projection, precise=True, dtype=ttnn.float32)
        return values, indices, projected

    def merge(self, values: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-device shard candidates [devices, rows, C] -> global top-k values and token ids [rows, k]."""
        devices, rows, _ = values.shape
        device = torch.arange(devices).reshape(-1, 1, 1) * self.local_vocab
        ids = (indices.long() + device).permute(1, 0, 2).reshape(rows, -1)
        top = torch.topk(values.float().permute(1, 0, 2).reshape(rows, -1), self.top_k, dim=-1)
        return top.values, torch.gather(ids, 1, top.indices)

    def select(self, projected: torch.Tensor, values: torch.Tensor, indices: torch.Tensor, anchor: int) -> list[int]:
        """Greedy candidate-selector walk on host: projected [7,P], values/indices [7,k]."""
        predecessor, path = anchor, []
        for p in range(projected.shape[0]):
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
        positions = self.positions(start, self.width)
        mask = ttnn.from_torch(
            self.ring_mask(start), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh, mesh_mapper=rep
        )
        hidden = self.draft(token_ids, positions, mask)
        values, indices, projected = self.candidates(hidden)

        def rows(tensor: ttnn.Tensor) -> torch.Tensor:
            """[devices, block rows 1.., width] from every device."""
            return torch.stack(
                [
                    ttnn.to_torch(part).reshape(-1, tensor.shape[-1])[1 : self.block]
                    for part in ttnn.get_device_tensors(tensor)
                ]
            )

        top_values, top_ids = self.merge(rows(values), rows(indices))
        proposal = self.select(rows(projected)[0].float(), top_values, top_ids, anchor)
        for tensor in (token_ids, positions, mask, hidden, values, indices, projected):
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
