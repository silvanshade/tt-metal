# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Write a verify block's K and V rows into the paged cache in one device program."""

from pathlib import Path

import ttnn

# Per 16-value face row: value bytes, and the exponent section leading a tile of that dtype.
_ROW = {ttnn.bfloat16: (32, 0), ttnn.bfloat8_b: (16, 64), ttnn.bfloat4_b: (8, 64)}


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def write_kv_block(k, v, keys, values, positions, page_table) -> None:
    """Write row r of `k` and `v` at position positions[r] of the paged caches.

    requires: k and v tiled [1, T, H, D] in the cache dtype (BF16 or block float), interleaved;
        caches [pages, H, block, D] in that dtype; positions int32 row-major [T]; page_table int32
        row-major [T, P] whose row 0 names the sequence's pages; rows of a cache tile touched
        by the block are consecutive in r.
    ensures: every touched cache row holds the input row's exact bytes; every other row is
        unchanged. One core owns each (cache, head, column tile), so writes never race.
    """
    rows, heads, width = k.shape[1], k.shape[2], k.shape[3]
    block = keys.shape[2]
    assert tuple(v.shape) == tuple(k.shape) and tuple(values.shape) == tuple(keys.shape)
    assert keys.shape[1] == heads and keys.shape[3] == width and block % 32 == 0 and width % 32 == 0
    assert k.dtype == v.dtype == keys.dtype == values.dtype and keys.dtype in _ROW
    assert positions.dtype == page_table.dtype == ttnn.int32
    assert positions.layout == page_table.layout == ttnn.ROW_MAJOR_LAYOUT and positions.shape[-1] >= rows
    columns = width // 32
    row_bytes, exponents = _ROW[keys.dtype]
    tile_bytes = exponents + 64 * row_bytes
    device = keys.device()
    grid = device.compute_with_storage_grid_size()
    cores = ttnn.num_cores_to_corerangeset(2 * heads * columns, grid, True)
    position_page, table_page = positions.buffer_aligned_page_size(), page_table.buffer_aligned_page_size()
    position_slot, table_slot = _round_up(position_page, 64), _round_up(table_page, 64)
    compile_args = [rows, heads, columns, block, tile_bytes, row_bytes, grid.x]
    compile_args += [position_page, table_page, position_slot, table_slot, exponents]
    tensors = (k, v, keys, values, positions, page_table)
    for tensor in tensors:
        compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    scratch = (rows + 1) * tile_bytes + position_slot + table_slot
    descriptor = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=str(Path(__file__).with_name("kv_kernels") / "writer.cpp"),
                core_ranges=cores,
                compile_time_args=compile_args,
                common_runtime_args=[tensor.buffer_address() for tensor in tensors],
                config=ttnn.ReaderConfigDescriptor(),
            )
        ],
        cbs=[
            ttnn.CBDescriptor(
                total_size=scratch,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=scratch)
                ],
            )
        ],
        semaphores=[],
    )
    ttnn.generic_op([*tensors], descriptor)
