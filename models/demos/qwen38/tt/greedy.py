# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Greedy tokens and top-k candidates from vocab-sharded LM-head logits, without gathering them."""

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


def _split(logits, mesh) -> tuple[int, int, int, int]:
    """Column tiles, tiles per core, scan cores and partials (two per core) for a shard."""
    grid = mesh.compute_with_storage_grid_size()
    tiles = logits.shape[-1] // 32
    per = -(-tiles // (grid.x * grid.y))
    cores = -(-tiles // per)
    return tiles, per, cores, 2 * cores


def _chunked(logits, run):
    """Apply an at-most-8-row kernel `run` to each 8-row slice of logits [1, 1, T, V]; concat
    each of its results along rows. The kernels keep a row's partials in half a face row."""
    rows, width = logits.shape[-2], logits.shape[-1]
    parts = []
    for first in range(0, rows, 8):
        piece = ttnn.slice(logits, (0, 0, first, 0), (1, 1, min(first + 8, rows), width))
        parts.append(run(piece))
        ttnn.deallocate(piece)
    results = []
    for outputs in zip(*parts, strict=True):
        results.append(ttnn.concat(list(outputs), dim=2, memory_config=ttnn.L1_MEMORY_CONFIG))
        for output in outputs:
            ttnn.deallocate(output)
    return results

def greedy_tokens(logits, mesh, tt_ccl, topology) -> ttnn.Tensor:
    """Per row, the vocabulary index of the maximum logit across every device's shard.

    requires: BF16 tiled interleaved logits [1, 1, T, V / devices] on each device, device d
        holding vocabulary columns [d * V / devices, (d + 1) * V / devices); 1 <= T <= 32 (rows
        past 8 run in 8-row slices).
    ensures: returns FP32 tiled [1, 1, T, 1] in L1, replicated, whose row t is the first index of
        the maximum of row t over the whole vocabulary, as argmax over the gathered logits (-0 and
        +0 tie); nothing else is allocated past the call.
    """
    rows, width = logits.shape[-2], logits.shape[-1]
    if rows > 8:
        return _chunked(logits, lambda piece: (greedy_tokens(piece, mesh, tt_ccl, topology),))[0]
    assert 1 <= rows <= 8 and width % 32 == 0
    assert logits.dtype == ttnn.bfloat16 and logits.layout == ttnn.TILE_LAYOUT and not logits.is_sharded()
    devices = mesh.get_num_devices()
    grid = mesh.compute_with_storage_grid_size()
    tiles, per, cores, count = _split(logits, mesh)
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


def shard_candidates(logits, count: int, width: int = 32) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Per row, the `count` largest logits of each device's vocabulary shard and their columns.

    requires: BF16 tiled interleaved logits [1, 1, T, V / devices] per device; 1 <= T <= 32 (rows
        past 8 run in 8-row slices);
        1 <= count <= width == 32.
    ensures: returns FP32 tiled [1, 1, T, width] values and shard-local columns per device (not
        gathered): row t holds the `count` largest values of row t, ties to the lower column, in
        no particular order, then -inf values with column 0.
    """
    rows = logits.shape[-2]
    if rows > 8:
        return tuple(_chunked(logits, lambda piece: shard_candidates(piece, count, width)))
    assert 1 <= rows <= 8 and 1 <= count <= width == 32 and logits.shape[-1] % 32 == 0
    assert logits.dtype == ttnn.bfloat16 and logits.layout == ttnn.TILE_LAYOUT and not logits.is_sharded()
    mesh = logits.device()
    grid = mesh.compute_with_storage_grid_size()
    tiles, per, cores, count_partials = _split(logits, mesh)
    page = count * 8
    lists = ttnn.empty(
        [count_partials * rows, 2 * count], dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    assert lists.buffer_aligned_page_size() == page
    slices = max(-(-per // 2) * 2 * rows * 32, rows * page)
    scan = ttnn.num_cores_to_corerangeset(cores, grid, True)
    tensors = (logits, lists)
    compile_args = [rows, tiles, per, grid.x]
    ttnn.generic_op(
        [*tensors],
        ttnn.ProgramDescriptor(
            kernels=[
                _kernel("top_partial.cpp", scan, [*compile_args, 0, count], tensors, ttnn.ReaderConfigDescriptor()),
                _kernel("top_partial.cpp", scan, [*compile_args, 1, count], tensors, ttnn.WriterConfigDescriptor()),
            ],
            cbs=[_scratch(scan, 0, slices), _scratch(scan, 1, slices)],
            semaphores=[],
        ),
    )
    values, columns = (
        ttnn.empty(
            [1, 1, rows, width], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        for _ in range(2)
    )
    reduce = ttnn.num_cores_to_corerangeset(rows, grid, True)
    tensors = (lists, values, columns)
    ttnn.generic_op(
        [*tensors],
        ttnn.ProgramDescriptor(
            kernels=[
                _kernel(
                    "top_final.cpp", reduce, [rows, count_partials, count, grid.x], tensors, ttnn.ReaderConfigDescriptor()
                )
            ],
            cbs=[_scratch(reduce, 0, count_partials * page + 256)],
            semaphores=[],
        ),
    )
    ttnn.deallocate(lists)
    return values, columns
