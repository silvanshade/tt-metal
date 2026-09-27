# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verify-block convolution, sequential token recurrence in one device program, and accepted-prefix
selection.

Convolution keeps every block row on its own tile row, so verification reads whole tiles.
Verification advances every value-column block of every head in parallel and writes the state after
each row. Folding an accepted prefix then selects one snapshot and one window of convolution rows,
so its cost does not depend on how many rows were accepted.
"""

import struct
from pathlib import Path

import torch

import ttnn

_TILE = ttnn.Tile([32, 32])
_ANY = 0xFFFFFFFF  # read the accepted count from its device tensor


def _cores(device, count: int) -> list[ttnn.CoreCoord]:
    grid = device.compute_with_storage_grid_size()
    assert count <= grid.x * grid.y
    return [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(count)]


def _core_set(device, count: int) -> ttnn.CoreRangeSet:
    """The first `count` cores of the row-major grid, as whole rows plus one partial row: dispatch
    multicasts binaries and configuration per rectangle, not per core."""
    return ttnn.num_cores_to_corerangeset(count, device.compute_with_storage_grid_size(), True)


def _cbs(cores: ttnn.CoreRangeSet, pages: dict[int, int], dtype: ttnn.DataType, page_size: int) -> list:
    return [
        ttnn.CBDescriptor(
            total_size=count * page_size,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=index, data_format=dtype, page_size=page_size, tile=ttnn.TileDescriptor(_TILE)
                )
            ],
        )
        for index, count in pages.items()
    ]


def _split(device, heads: int) -> int:
    """Cores per head: the most value-column blocks the grid holds for every head at once."""
    grid = device.compute_with_storage_grid_size()
    return next(split for split in (4, 2, 1) if heads * split <= grid.x * grid.y)


def conv_constants(taps: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Selectors [9, 32, 32] and tap diagonals [4, 32, W] for `verify_conv`, from four [W] taps.

    Selector m < 4 places tap state m on row m, selector 4 moves token t to row 4 + t, selector
    4 + s lifts row t + s to row t. Tile (j, c) of the diagonals is diag(taps[j] over channel tile
    c), so sharding W on a tile boundary keeps every tile whole.
    """
    assert len(taps) == 4
    selectors = torch.zeros(9, 32, 32)
    for m in range(4):
        selectors[m, m, 0] = 1
    selectors[4, torch.arange(4, 32), torch.arange(28)] = 1
    for s in range(1, 5):
        selectors[4 + s, torch.arange(32 - s), torch.arange(s, 32)] = 1
    width = taps[0].numel()
    channels = torch.arange(width)
    diagonals = torch.zeros(4, 32, width)
    for j, tap in enumerate(taps):
        diagonals[j, channels % 32, channels] = tap.float()
    return selectors, diagonals


def verify_conv(qkv, states, diagonals, selectors, history):
    """Causal convolution and SiLU of a verify block; write the block's tap history.

    requires: BF16 tiled qkv [1, T, W] and four BF16 tiled tap states [1, 1, W]; BF16 tiled
        diagonals [4, 32, W] and selectors [9, 32, 32] from `conv_constants`; BF16 tiled history
        [1, 4 + T, W]; 4 + T <= 32; interleaved tensors.
    ensures: history rows are the tap states then the qkv rows; returns FP32 tiled [1, T, W] whose
        row t is silu(sum_j taps[j] * history[t + j + 1]). Padding rows are finite, outside the result.
    """
    rows, width = qkv.shape[-2], qkv.shape[-1]
    assert len(states) == 4 and 4 + rows <= 32 and width % 32 == 0
    assert history.shape[-2] == 4 + rows and history.shape[-1] == width
    device = qkv.device()
    tensors = (selectors, qkv, *states, diagonals)
    assert all(t.dtype == ttnn.bfloat16 and t.layout == ttnn.TILE_LAYOUT for t in (*tensors, history))
    activated = ttnn.empty(
        qkv.shape, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    grid = device.compute_with_storage_grid_size()
    tiles = width // 32
    per = -(-tiles // (grid.x * grid.y))
    cores = _core_set(device, -(-tiles // per))
    shape = [tiles, per, grid.x]
    reader_compile = list(shape)
    for tensor in tensors:
        reader_compile.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    writer_compile = list(shape)
    for tensor in (history, activated):
        writer_compile.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    kernels = Path(__file__).with_name("conv_kernels")
    bf16 = _cbs(cores, {0: 9, 1: 10, 2: 8, 3: 1, 4: 2, 5: 4}, ttnn.bfloat16, 2048)
    descriptor = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=str(kernels / "reader.cpp"),
                core_ranges=cores,
                compile_time_args=reader_compile,
                common_runtime_args=[t.buffer_address() for t in tensors],
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(kernels / "writer.cpp"),
                core_ranges=cores,
                compile_time_args=writer_compile,
                common_runtime_args=[history.buffer_address(), activated.buffer_address()],
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(kernels / "compute.cpp"),
                core_ranges=cores,
                compile_time_args=shape,
                # Selector and diagonal products are exact only when every mantissa phase runs.
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
                ),
            ),
        ],
        cbs=bf16 + _cbs(cores, {6: 2}, ttnn.float32, 4096),
        semaphores=[],
    )
    ttnn.generic_op([*tensors, history, activated], descriptor)
    return activated


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def verify_recurrence(activated, g, beta, state, snapshots, key_heads: int, scale: float, eps: float = 1e-6):
    """Return every row's output; write the state after each row into `snapshots`.

    requires: FP32 activated [1, T, W] from `verify_conv`, W = (2 * key_heads + H) * 128 holding
        q, k (key heads) then v (value heads); FP32 g/beta [T,1,H]; FP32 state [1,H,128,128]; FP32
        snapshots [>=T,H,128,128]; value head h reads key head h // (H / key_heads); interleaved tiles.
    ensures: q and k are L2-normalized (q times `scale`) per row; each head advances rows in order;
        `state` is not written; snapshot r holds the state after rows 0..r. No state or token crosses
        the host boundary. Padding is not part of the result.
    """
    rows, width = activated.shape[-2], activated.shape[-1]
    heads = state.shape[1]
    assert width == (2 * key_heads + heads) * 128 and heads % key_heads == 0
    assert 1 <= rows <= 32 and snapshots.shape[0] >= rows and snapshots.shape[1] == heads
    device = activated.device()
    tensors = (activated, g, beta, state)
    assert all(
        t.dtype == ttnn.float32 and t.layout == ttnn.TILE_LAYOUT and not t.is_sharded() for t in (*tensors, snapshots)
    )
    output = ttnn.empty(
        [rows, 1, heads, 128],
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    split = _split(device, heads)
    columns = 4 // split
    cores = _core_set(device, heads * split)
    # Core i of the row-major grid owns value-column block i; kernels derive it from their
    # coordinates, so every argument is common and dispatch writes it once per kernel.
    grid_x = device.compute_with_storage_grid_size().x
    n = columns
    pages = {0: 4, 1: 4, 2: n, 3: 2, 4: 2, 5: 4 * n, 6: 4 * n, 7: 4 * n, 8: n, 9: n, 10: 4, 11: 4, 12: 4, 13: 4}
    pages |= {14: 2 * n, 15: 1, 16: 8 * n, 17: 1, 18: 1, 19: 1}
    head_tiles = (heads + 31) // 32
    reader_compile = [head_tiles, columns, grid_x, heads // key_heads, key_heads * 4]
    for tensor in tensors:
        reader_compile.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    writer_compile = [head_tiles, columns, heads, grid_x]
    for tensor in (output, snapshots):
        writer_compile.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    kernels = Path(__file__).with_name("verify_kernels")
    descriptor = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=str(kernels / "reader.cpp"),
                core_ranges=cores,
                compile_time_args=reader_compile,
                common_runtime_args=[rows, *(t.buffer_address() for t in tensors)],
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(kernels / "writer.cpp"),
                core_ranges=cores,
                compile_time_args=writer_compile,
                common_runtime_args=[rows, output.buffer_address(), snapshots.buffer_address()],
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(kernels / "compute.cpp"),
                core_ranges=cores,
                compile_time_args=[columns, _bits(eps), _bits(scale)],
                common_runtime_args=[rows],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
                ),
            ),
        ],
        cbs=_cbs(cores, pages, ttnn.float32, 4096),
        semaphores=[],
    )
    ttnn.generic_op([*tensors, snapshots, output], descriptor)
    return output


def _accepted_args(accepted: int | ttnn.Tensor) -> tuple[int, ttnn.Tensor | None]:
    if isinstance(accepted, ttnn.Tensor):
        assert accepted.dtype == ttnn.float32 and accepted.layout == ttnn.TILE_LAYOUT
        return _ANY, accepted
    assert accepted >= 0
    return int(accepted), None


def select_state(snapshots, checkpoint, accepted: int | ttnn.Tensor, state, slot: int) -> None:
    """Write the state after `accepted` verified rows into `state[slot]`.

    requires: FP32 tiled snapshots [T,H,128,128] from verify_recurrence, checkpoint [1,H,128,128] it
        started from, state [B,H,128,128]; accepted is an int or FP32 tiled scalar in [0, T].
    ensures: slot `slot` holds the checkpoint for zero, else snapshot accepted - 1; other slots unchanged.
    """
    pages = checkpoint.shape[1] * 16
    assert tuple(snapshots.shape)[1:] == tuple(checkpoint.shape)[1:] == tuple(state.shape)[1:]
    assert 0 <= slot < state.shape[0]
    fixed, count = _accepted_args(accepted)
    count = count if count is not None else checkpoint  # unread placeholder keeps the accessor valid
    device = state.device()
    # Latency-bound: spread the pages over the grid so each core copies one batch.
    grid = device.compute_with_storage_grid_size()
    per = -(-pages // (grid.x * grid.y))
    cores = -(-pages // per)
    coordinates = _cores(device, cores)
    args = ttnn.RuntimeArgs()
    tensors = (count, snapshots, checkpoint, state)
    for index, core in enumerate(coordinates):
        begin, end = min(index * per, pages), min((index + 1) * per, pages)
        args[core.x][core.y] = [fixed, slot, begin, end, *(t.buffer_address() for t in tensors)]
    batch = 8
    compile_args = [pages, batch]
    for tensor in tensors:
        compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    core_set = _core_set(device, cores)
    descriptor = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=str(Path(__file__).with_name("fold_kernels") / "select_state.cpp"),
                core_ranges=core_set,
                compile_time_args=compile_args,
                runtime_args=args,
                config=ttnn.ReaderConfigDescriptor(),
            )
        ],
        cbs=_cbs(core_set, {0: batch}, ttnn.float32, 4096),
        semaphores=[],
    )
    ttnn.generic_op([*tensors], descriptor)


def select_rows(history, accepted: int | ttnn.Tensor, taps: list, slot: int) -> None:
    """Write convolution taps after `accepted` verified rows into row `slot` of each tap state.

    requires: BF16 tiled history [1, K + T, W] (checkpoint taps then verified inputs, K + T <= 32);
        four BF16 tiled tap states [1, B, W]; accepted an int or FP32 tiled scalar in [0, T].
    ensures: tap m's row `slot` equals history row accepted + m; other rows unchanged.
    """
    assert len(taps) == 4 and history.shape[-2] <= 32
    assert history.dtype == ttnn.bfloat16 and all(t.dtype == ttnn.bfloat16 and t.shape[-1] == history.shape[-1] for t in taps)
    fixed, count = _accepted_args(accepted)
    count = count if count is not None else history
    tiles = history.shape[-1] // 32
    device = history.device()
    # Latency-bound: spread the tiles over the grid; each core's rows share one scratch page.
    grid = device.compute_with_storage_grid_size()
    per = -(-tiles // (grid.x * grid.y))
    assert per <= 8
    cores = -(-tiles // per)
    coordinates = _cores(device, cores)
    args = ttnn.RuntimeArgs()
    tensors = (count, history, *taps)
    for index, core in enumerate(coordinates):
        begin, end = min(index * per, tiles), min((index + 1) * per, tiles)
        args[core.x][core.y] = [fixed, slot, begin, end, *(t.buffer_address() for t in tensors)]
    compile_args = []
    for tensor in tensors:
        compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    core_set = _core_set(device, cores)
    descriptor = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=str(Path(__file__).with_name("fold_kernels") / "select_rows.cpp"),
                core_ranges=core_set,
                compile_time_args=compile_args,
                runtime_args=args,
                config=ttnn.ReaderConfigDescriptor(),
            )
        ],
        cbs=_cbs(core_set, {0: 1}, ttnn.float32, 4096),
        semaphores=[],
    )
    ttnn.generic_op([*tensors], descriptor)
