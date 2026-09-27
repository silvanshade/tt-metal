# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sequential token recurrence in one device program, and accepted-prefix selection.

Verification advances every value-column block of every head in parallel and writes the state after
each row. Folding an accepted prefix then selects one snapshot and one window of convolution rows,
so its cost does not depend on how many rows were accepted.
"""

from pathlib import Path

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


def verify_recurrence(q, k, v, g, beta, state, snapshots):
    """Return every row's output; write the state after each row into `snapshots`.

    requires: normalized/scaled FP32 q, normalized FP32 k, FP32 v [T,1,H,128]; FP32 g/beta [T,1,H];
        FP32 state [1,H,128,128]; FP32 snapshots [>=T,H,128,128]; interleaved tiled tensors.
    ensures: each head advances rows in order; `state` is not written; snapshot r holds the state
        after rows 0..r. No state or token crosses the host boundary. Padding is not part of the result.
    """
    rows, _, heads, width = q.shape
    assert width == 128 and 1 <= rows <= 32 and snapshots.shape[0] >= rows and snapshots.shape[1] == heads
    device = q.device()
    tensors = (q, k, v, g, beta, state)
    assert all(
        t.dtype == ttnn.float32 and t.layout == ttnn.TILE_LAYOUT and not t.is_sharded() for t in (*tensors, snapshots)
    )
    output = ttnn.empty(
        q.shape, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    split = _split(device, heads)
    columns = 4 // split
    cores = _core_set(device, heads * split)
    # Core i of the row-major grid owns value-column block i; kernels derive it from their
    # coordinates, so every argument is common and dispatch writes it once per kernel.
    grid_x = device.compute_with_storage_grid_size().x
    n = columns
    pages = {0: 4, 1: 4, 2: n, 3: 2, 4: 2, 5: 4 * n, 6: 4 * n, 7: 4 * n, 8: n, 9: n, 10: 4, 14: 2 * n, 15: 1}
    pages |= {16: 8 * n, 18: 1}
    head_tiles = (heads + 31) // 32
    reader_compile = [head_tiles, columns, grid_x]
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
                compile_time_args=[columns],
                common_runtime_args=[rows],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False
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


def select_state(snapshots, checkpoint, accepted: int | ttnn.Tensor, state, slot: int, cores: int = 32) -> None:
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
    coordinates = _cores(device, cores)
    per = (pages + cores - 1) // cores
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


def select_rows(history, accepted: int | ttnn.Tensor, taps: list, slot: int, cores: int = 20) -> None:
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
    coordinates = _cores(device, cores)
    per = (tiles + cores - 1) // cores
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
