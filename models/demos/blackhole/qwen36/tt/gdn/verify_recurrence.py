# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sequential token recurrence in one device program, parallel across heads."""

from pathlib import Path

import ttnn


def verify_recurrence(q, k, v, g, beta, state):
    """Return every row's output/delta and final state without host synchronization.

    requires: normalized/scaled FP32 q, normalized FP32 k, FP32 v [T,1,H,128];
        FP32 g/beta [T,1,H]; FP32 state [1,H,128,128]; interleaved tiled tensors.
    ensures: each head advances rows in order; input state is retained; no state
        or token crosses the host boundary. Padding is not part of the result.
    """
    rows, _, heads, width = q.shape
    assert width == 128 and 1 <= rows <= 32
    device = q.device()
    tensors = (q, k, v, g, beta, state)
    assert all(t.dtype == ttnn.float32 and t.layout == ttnn.TILE_LAYOUT and not t.is_sharded() for t in tensors)
    output, delta = [
        ttnn.empty(
            q.shape, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        for _ in range(2)
    ]
    final = ttnn.empty(
        state.shape, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=state.memory_config()
    )
    grid = device.compute_with_storage_grid_size()
    assert heads <= grid.x * grid.y
    coordinates = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(heads)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coordinates])
    reader_args, writer_args, compute_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for head, core in enumerate(coordinates):
        reader_args[core.x][core.y] = [head, rows, *(t.buffer_address() for t in tensors)]
        writer_args[core.x][core.y] = [
            head,
            rows,
            output.buffer_address(),
            delta.buffer_address(),
            final.buffer_address(),
        ]
        compute_args[core.x][core.y] = [rows]
    tile = ttnn.Tile([32, 32])
    pages = {
        0: 4,
        1: 4,
        2: 4,
        3: 1,
        4: 1,
        5: 16,
        6: 16,
        7: 16,
        8: 4,
        9: 4,
        10: 4,
        11: 16,
        12: 16,
        14: 4,
        15: 1,
        16: 16,
        17: 4,
        18: 1,
    }
    cbs = [
        ttnn.CBDescriptor(
            total_size=count * 4096,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=index, data_format=ttnn.float32, page_size=4096, tile=ttnn.TileDescriptor(tile)
                )
            ],
        )
        for index, count in pages.items()
    ]
    reader_compile, writer_compile = [(heads + 31) // 32], [(heads + 31) // 32]
    for tensor in tensors:
        reader_compile.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    for tensor in (output, delta, final):
        writer_compile.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    kernels = Path(__file__).with_name("verify_kernels")
    descriptor = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=str(kernels / "reader.cpp"),
                core_ranges=cores,
                compile_time_args=reader_compile,
                runtime_args=reader_args,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(kernels / "writer.cpp"),
                core_ranges=cores,
                compile_time_args=writer_compile,
                runtime_args=writer_args,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(kernels / "compute.cpp"),
                core_ranges=cores,
                runtime_args=compute_args,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False
                ),
            ),
        ],
        cbs=cbs,
        semaphores=[],
    )
    ttnn.generic_op([*tensors, output, delta, final], descriptor)
    return output, delta, final
