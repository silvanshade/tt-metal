# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode matmul layout sweep: 1D mcast on interleaved weights vs DRAM-sharded weights.

The decode projections are weight-read bound at M = 1 tile row: the program reads the whole
BF4_B weight once per token, so the only quantity that matters is the rate at which that read
retires. This sweep measures that rate for both layouts on the served shapes.

Arms:

* ``1d`` — the served configuration: DRAM-interleaved weight, ``MatmulMultiCoreReuseMultiCast1D``
  with ``mcast_in0`` on the tuned small grid (``tp_common.create_matmul_1d_decode_progcfg``).
* ``dram_sharded`` — WIDTH_SHARDED weight, one shard per DRAM bank, readers beside their bank
  (``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig``), swept over
  ``num_workers_per_dram_bank`` and every legal ``in0_block_w``.
* ``auto`` — no program config, for the shapes the model leaves to the ttnn default (LM head).

Durations are device program durations taken from the runtime profiler callback, so the reported
rate excludes host dispatch and excludes the activation reshard each arm needs; the reshard is
measured separately and reported in its own column. Effective GB/s is padded weight bytes over
the median program duration, the same one-read model the per-op timelines use; it is not a bus
counter.

Run:

    pytest models/demos/qwen38/tests/test_decode_matmul_layout_sweep.py -p no:cacheprovider

Environment:

* ``QWEN38_SWEEP_OUT`` — directory for the CSV rows (default: no CSV, table logged only).
* ``QWEN38_SWEEP_ITERS`` — measured iterations per cell (default 40, minimum 30).
* ``QWEN38_SWEEP_SHAPES`` — comma-separated shape labels to restrict the sweep.
"""

import csv
import math
import os
import statistics
import threading
import time
from pathlib import Path

try:
    import pytest
except ModuleNotFoundError:  # standalone run against an installed wheel
    pytest = None
import torch
from loguru import logger

import ttnn
from models.demos.qwen38.tt import tp_common as tpc

TILE = tpc.TILE_SIZE
DRAM_BANKS = tpc.DRAM_CORES
BF4_TILE_BYTES = 576  # 512 mantissa + 64 exponent bytes for a 32x32 BFP4_B tile

# Served decode shapes at TP=1, with the core count the model's 1D grid uses for each.
# `cores_1d = None` means the model leaves the shape to the ttnn default (the LM head).
SERVED_SHAPES = (
    # label, K, N, cores_1d, served
    ("mlp_gate_up", 5120, 17408, 44, True),
    ("mlp_down", 17408, 5120, 33, True),
    ("attn_qkv", 5120, 14336, 64, True),
    ("gdn_qkvzab", 5120, 16480, 44, True),
    ("out_proj", 6144, 5120, 33, True),
    ("lm_head", 5120, 248320, None, True),
    ("n10240", 5120, 10240, 44, False),
)

WORKERS = (1, 2, 3)
MIN_ITERS = 30


class ProgramTimer:
    """Device program durations from the runtime profiler callback."""

    def __init__(self):
        self._lock = threading.Lock()
        self._records = []
        self._handle = None

    def register(self):
        """Attach the callback once; raises if the runtime has no realtime profiler."""
        if self._handle is None:
            self._handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(self._on_batch)
            assert ttnn.device.IsProgramRealtimeProfilerActive(), "realtime profiler inactive"

    def _on_batch(self, batch):
        with self._lock:
            for record in batch.records:
                self._records.append((record.start_timestamp, record.end_timestamp, record.frequency))

    def clear(self):
        with self._lock:
            self._records.clear()

    def count(self):
        with self._lock:
            return len(self._records)

    def drain(self, limit_s=1.0):
        """Wait for the callback to settle: three quiet polls or the deadline."""
        seen = self.count()
        quiet = 0
        deadline = time.perf_counter() + limit_s
        while quiet < 3 and time.perf_counter() < deadline:
            time.sleep(0.001)
            now = self.count()
            if now == seen:
                quiet += 1
            else:
                seen, quiet = now, 0
        return seen

    def durations_us(self):
        with self._lock:
            return [(end - start) / freq / 1000 for start, end, freq in self._records]


def roundup(value, multiple):
    return multiple * math.ceil(value / multiple)


def sharded_core_counts(k):
    """Core counts the DRAM-sharded arm can use: K must divide evenly and the grid must exist.

    The activation is width-sharded over these cores and the count is the compute grid, so it is
    a sweep dimension of its own: too few cores and the per-core output block does not fit L1,
    too many and each core's share of K stops dividing.
    """
    k_tiles = k // TILE
    return [cores for cores in range(8, 81, 8) if k_tiles % cores == 0]


def activation_shard_config(k, num_cores):
    """WIDTH_SHARDED L1 activation over an explicit core count, eight columns per row."""
    cols = 8
    return ttnn.create_sharded_memory_config(
        shape=(TILE, k // num_cores),
        core_grid=ttnn.CoreGrid(x=cols, y=num_cores // cols),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def legal_in0_block_w(k, num_cores):
    """Every in0_block_w the DRAM-sharded kernel admits for this K and core grid.

    The kernel requires the activation shard width in tiles, K / num_cores, to be a whole number
    of blocks: `(shard_shape[1] / tile_width) % in0_block_w == 0`. Nothing else is legal, so the
    sweep is over the divisors of that width and the device rejects anything this misses.
    """
    per_core_k = k // TILE // num_cores
    return [bw for bw in range(1, per_core_k + 1) if per_core_k % bw == 0]


def quartiles(values):
    ordered = sorted(values)
    half = len(ordered) // 2
    lower = statistics.median(ordered[:half])
    upper = statistics.median(ordered[-half:])
    return lower, upper


def measure(timer, run, iters, warmup=5):
    """Median and IQR of the device program duration of one matmul, in microseconds."""
    for _ in range(warmup):
        ttnn.deallocate(run())
    ttnn.synchronize_device(run.device)
    timer.drain()
    timer.clear()
    for _ in range(iters):
        ttnn.deallocate(run())
    ttnn.synchronize_device(run.device)
    timer.drain()
    durations = timer.durations_us()
    assert len(durations) >= iters, f"expected >= {iters} program records, got {len(durations)}"
    # One program per call; extra records would mean the call dispatched more than the matmul.
    assert len(durations) == iters, f"expected exactly {iters} program records, got {len(durations)}"
    low, high = quartiles(durations)
    return statistics.median(durations), low, high, min(durations), max(durations)


class MatmulRun:
    """One measured matmul: fixed operands, one dispatched program per call."""

    def __init__(self, device, activation, weight, program_config, memory_config, compute_config):
        self.device = device
        self._activation = activation
        self._weight = weight
        self._program_config = program_config
        self._memory_config = memory_config
        self._compute_config = compute_config

    def __call__(self):
        return ttnn.linear(
            self._activation,
            self._weight,
            program_config=self._program_config,
            memory_config=self._memory_config,
            compute_kernel_config=self._compute_config,
        )


def weight_bytes(k, n):
    """Padded BF4_B bytes of a [k, n] weight, the one-read model of the per-op timelines."""
    return (roundup(k, TILE) // TILE) * (roundup(n, TILE) // TILE) * BF4_TILE_BYTES


def run_sweep(mesh_device):
    """Sweep both decode weight layouts over the served shapes; returns one row per cell."""
    iters = max(MIN_ITERS, int(os.environ.get("QWEN38_SWEEP_ITERS", "40")))
    wanted = os.environ.get("QWEN38_SWEEP_SHAPES")
    selected = set(wanted.split(",")) if wanted else None
    out_dir = os.environ.get("QWEN38_SWEEP_OUT")

    timer = ProgramTimer()
    timer.register()
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    grid_w = mesh_device.compute_with_storage_grid_size().x
    rows = []

    for label, k, n, cores_1d, served in SERVED_SHAPES:
        if selected is not None and label not in selected:
            continue
        torch_weight = torch.randn(k, n, dtype=torch.bfloat16)
        torch_activation = torch.randn(1, 1, TILE, k, dtype=torch.bfloat16)
        bytes_read = weight_bytes(k, n)

        # ---- 1D arm: interleaved weight, activation interleaved in L1 ----
        weight_il = ttnn.as_tensor(
            torch_weight,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # The LM head is the one served projection with no program config, and the model hands it
        # a DRAM activation and takes a DRAM result; every other shape runs out of and into L1.
        control_memcfg = ttnn.DRAM_MEMORY_CONFIG if cores_1d is None else ttnn.L1_MEMORY_CONFIG
        activation_il = ttnn.as_tensor(
            torch_activation,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=control_memcfg,
        )
        if cores_1d is None:
            program_config = None
            arm_label = "auto"
        else:
            program_config = tpc.create_matmul_1d_decode_progcfg(TILE, k, n, num_cores=cores_1d, grid_w=grid_w)
            arm_label = "1d"
        run = MatmulRun(mesh_device, activation_il, weight_il, program_config, control_memcfg, compute_config)
        median, low, high, fastest, slowest = measure(timer, run, iters)
        rows.append(
            dict(
                shape=label,
                served=served,
                k=k,
                n=n,
                arm=arm_label,
                cores=cores_1d if cores_1d is not None else "",
                workers="",
                in0_block_w=program_config.in0_block_w if program_config is not None else "",
                iters=iters,
                median_us=median,
                iqr_low_us=low,
                iqr_high_us=high,
                min_us=fastest,
                max_us=slowest,
                weight_bytes=bytes_read,
                gb_s=bytes_read / median / 1000,
            )
        )
        logger.info(f"{label} {arm_label}: {median:.3f} us  {bytes_read / median / 1000:.1f} GB/s")
        ttnn.deallocate(weight_il)
        ttnn.deallocate(activation_il)

        # ---- DRAM-sharded arm: width-sharded weight, width-sharded activation ----
        n_padded = roundup(n, TILE * DRAM_BANKS)
        weight_memcfg = tpc.create_dram_sharded_mem_config(k, n)
        padded_weight = torch_weight
        if n_padded != n:
            padded_weight = torch.nn.functional.pad(torch_weight, (0, n_padded - n))
        weight_sh = ttnn.as_tensor(
            padded_weight,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=weight_memcfg,
        )
        shard_width_tiles = n_padded // (TILE * DRAM_BANKS)
        for num_cores in sharded_core_counts(k):
            activation_sh = ttnn.as_tensor(
                torch_activation,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=activation_shard_config(k, num_cores),
            )
            per_core_n = math.ceil(n_padded / (TILE * num_cores))
            for workers in WORKERS:
                if shard_width_tiles % workers:
                    logger.info(
                        f"{label} dram_sharded c={num_cores} workers={workers}: skipped, "
                        f"{shard_width_tiles} tiles per bank"
                    )
                    continue
                for block_w in legal_in0_block_w(k, num_cores):
                    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                        in0_block_w=block_w,
                        per_core_M=1,
                        per_core_N=per_core_n,
                        fused_activation=None,
                        num_workers_per_dram_bank=workers,
                    )
                    run = MatmulRun(
                        mesh_device,
                        activation_sh,
                        weight_sh,
                        program_config,
                        ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                        compute_config,
                    )
                    try:
                        median, low, high, fastest, slowest = measure(timer, run, iters)
                    except RuntimeError as error:
                        logger.info(
                            f"{label} dram_sharded c={num_cores} w={workers} bw={block_w}: "
                            f"rejected ({str(error).splitlines()[0]})"
                        )
                        continue
                    rows.append(
                        dict(
                            shape=label,
                            served=served,
                            k=k,
                            n=n,
                            arm="dram_sharded",
                            cores=num_cores,
                            workers=workers,
                            in0_block_w=block_w,
                            iters=iters,
                            median_us=median,
                            iqr_low_us=low,
                            iqr_high_us=high,
                            min_us=fastest,
                            max_us=slowest,
                            weight_bytes=bytes_read,
                            gb_s=bytes_read / median / 1000,
                        )
                    )
                    logger.info(
                        f"{label} dram_sharded c={num_cores} w={workers} bw={block_w}: {median:.3f} us "
                        f"{bytes_read / median / 1000:.1f} GB/s"
                    )
            ttnn.deallocate(activation_sh)
        ttnn.deallocate(weight_sh)

    best = {}
    for row in rows:
        key = row["shape"]
        if key not in best or row["gb_s"] > best[key]["gb_s"]:
            best[key] = row
    for label, row in best.items():
        logger.info(f"best {label}: {row['arm']} w={row['workers']} bw={row['in0_block_w']} {row['gb_s']:.1f} GB/s")

    if out_dir:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "decode-matmul-sweep.csv"
        with target.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"wrote {target}")

    assert rows, "sweep produced no measurements"
    return rows


if pytest is not None:

    @pytest.mark.parametrize(
        "mesh_device",
        [{"P150": (1, 1)}.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
        indirect=True,
    )
    @pytest.mark.parametrize("device_params", [{"trace_region_size": 0, "l1_small_size": 24576}], indirect=True)
    def test_decode_matmul_layout_sweep(mesh_device):
        """Both layouts measure on every selected shape."""
        run_sweep(mesh_device)


if __name__ == "__main__":
    # Standalone entry so the sweep also runs against an installed wheel, where the repository's
    # pytest fixtures are not available.
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0)
    try:
        run_sweep(device)
    finally:
        ttnn.close_mesh_device(device)
