# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Greedy tokens from vocab-sharded LM-head logits without gathering the logits."""

from pathlib import Path

import ttnn
from models.tt_transformers.tt.ccl import tt_all_gather

_KERNELS = Path(__file__).with_name("greedy_kernels")


def _scratch(cores: ttnn.CoreRangeSet, index: int, size: int) -> ttnn.CBDescriptor:
    return ttnn.CBDescriptor(
        total_size=size,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.uint32, page_size=size)],
    )


def _kernel(source: str, cores, compile_args: list, tensors: tuple, config) -> ttnn.KernelDescriptor:
    compile_args = list(compile_args)
    for tensor in tensors:
        compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    return ttnn.KernelDescriptor(
        kernel_source=str(_KERNELS / source),
        core_ranges=cores,
        compile_time_args=compile_args,
        common_runtime_args=[tensor.buffer_address() for tensor in tensors],
        config=config,
    )


def greedy_tokens(logits, mesh, tt_ccl, topology) -> ttnn.Tensor:
    """Per row, the vocabulary index of the maximum logit across every device's shard.

    requires: BF16 tiled interleaved logits [1, 1, T, V / devices] on each device, device d
        holding vocabulary columns [d * V / devices, (d + 1) * V / devices); 1 <= T <= 8.
    ensures: returns FP32 tiled [1, 1, T, 1] in L1, replicated, whose row t is the first index of
        the maximum of row t over the whole vocabulary, as argmax over the gathered logits (-0 and
        +0 tie); nothing else is allocated past the call.
    """
    rows, width = logits.shape[-2], logits.shape[-1]
    assert 1 <= rows <= 8 and width % 32 == 0
    assert logits.dtype == ttnn.bfloat16 and logits.layout == ttnn.TILE_LAYOUT and not logits.is_sharded()
    devices = mesh.get_num_devices()
    grid = mesh.compute_with_storage_grid_size()
    tiles = width // 32
    per = -(-tiles // (grid.x * grid.y))
    cores = -(-tiles // per)
    count = 2 * cores  # one partial per data-movement core
    tile_rows = -(-count // 32)

    # Each core's two data-movement processors scan the lower and upper halves of its tiles.
    partials = ttnn.empty(
        [1, 1, 32 * tile_rows, 32], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scan = ttnn.num_cores_to_corerangeset(cores, grid, True)
    scratch = max(-(-per // 2) * 2 * rows * 32, 64)
    tensors = (logits, partials)
    ttnn.generic_op(
        [*tensors],
        ttnn.ProgramDescriptor(
            kernels=[
                _kernel("partial.cpp", scan, [rows, tiles, per, grid.x, 0], tensors, ttnn.ReaderConfigDescriptor()),
                _kernel("partial.cpp", scan, [rows, tiles, per, grid.x, 1], tensors, ttnn.WriterConfigDescriptor()),
            ],
            cbs=[_scratch(scan, 0, scratch), _scratch(scan, 1, scratch)],
            semaphores=[],
        ),
    )
    gathered = partials
    if devices > 1:
        gathered = tt_all_gather(
            partials, mesh, tt_ccl, cluster_axis=None, dim=3, topology=topology,
            memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.float32,
        )
        if partials.is_allocated():
            ttnn.deallocate(partials)

    # Core t reduces row t.
    tokens = ttnn.empty(
        [1, 1, rows, 1], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    reduce = ttnn.num_cores_to_corerangeset(rows, grid, True)
    tensors = (gathered, tokens)
    ttnn.generic_op(
        [*tensors],
        ttnn.ProgramDescriptor(
            kernels=[
                _kernel(
                    "final.cpp", reduce, [rows, count, devices, width, grid.x], tensors, ttnn.ReaderConfigDescriptor()
                )
            ],
            cbs=[_scratch(reduce, 0, (devices * tile_rows + 1) * 4096)],
            semaphores=[],
        ),
    )
    ttnn.deallocate(gathered)
    return tokens
