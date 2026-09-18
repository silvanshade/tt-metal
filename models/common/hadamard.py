# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Matrix-free orthogonal rotation for queries and quantized attention keys."""

from math import prod
from pathlib import Path

import ttnn


def hadamard_h128(tensor):
    """Rotate tiled BF16 rows of width 128 into normalized BFP8 on Blackhole.

    # Specification
    - requires: nonempty interleaved device tensor, standard tiles, final width 128.
    - ensures: input is retained; normalized rows return as BFP8, with shape and memory preserved.
    - fails: ValueError for non-Blackhole, non-tiled/BF16, nonstandard tile/width or sharded input.
    - panics: none.
    - intension: four MVMULs per row; no persistent matrix or host tensor copy.

    # Adequacy
    Host float64 basis/sign/exponent/full-mantissa witnesses distinguish sign,
    normalization and row/face-addressing faults; poisoned trace replays detect
    stale output. Finite witnesses do not establish a universal error bound.
    """
    device = tensor.device()
    if device.arch() != ttnn.device.Arch.BLACKHOLE:
        raise ValueError("H128 requires Blackhole")
    if tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("H128 requires tiled BF16 input")
    if tensor.shape[-1] != 128 or tuple(tensor.tile.tile_shape) != (32, 32):
        raise ValueError("H128 requires width 128 with standard 32x32 tiles")
    if tensor.is_sharded():
        raise ValueError("H128 requires interleaved memory")
    output = ttnn.empty(
        tensor.shape, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=tensor.memory_config()
    )
    blocks = prod(tensor.padded_shape) // (32 * 128)
    grid = device.compute_with_storage_grid_size()
    workers = min(blocks, grid.x * grid.y)
    coordinates = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coordinates])
    read_args, write_args, compute_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    first = 0
    for index, core in enumerate(coordinates):
        count = blocks // workers + (index < blocks % workers)
        read_args[core.x][core.y] = [tensor.buffer_address(), first, count]
        write_args[core.x][core.y] = [output.buffer_address(), first, count]
        compute_args[core.x][core.y] = [count * 32]
        first += count
    face, tile = ttnn.Tile([16, 16]), ttnn.Tile([32, 32])
    cbs = []
    for index, dtype, shape, pages in [
        (0, ttnn.bfloat16, face, 2),
        (1, ttnn.bfloat16, face, 1),
        (2, ttnn.bfloat16, tile, 4),
        (16, ttnn.bfloat8_b, face, 2),
        (17, ttnn.bfloat8_b, tile, 4),
    ]:
        size = shape.get_tile_size(dtype)
        cbs.append(
            ttnn.CBDescriptor(
                total_size=pages * size,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=index, data_format=dtype, page_size=size, tile=ttnn.TileDescriptor(shape)
                    )
                ],
            )
        )
    kernels = Path(__file__).with_name("hadamard_kernels")
    descriptor = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=str(kernels / "reader.cpp"),
                core_ranges=cores,
                compile_time_args=ttnn.TensorAccessorArgs(tensor).get_compile_time_args(),
                runtime_args=read_args,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(kernels / "writer.cpp"),
                core_ranges=cores,
                compile_time_args=ttnn.TensorAccessorArgs(output).get_compile_time_args(),
                runtime_args=write_args,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(kernels / "compute.cpp"),
                core_ranges=cores,
                runtime_args=compute_args,
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, dst_full_sync_en=True),
            ),
        ],
        cbs=cbs,
        semaphores=[],
    )
    return ttnn.generic_op([tensor, output], descriptor)


class HadamardRotation:
    """Apply a normalized Sylvester rotation without a persistent device matrix.

    Apply after RoPE and before key quantization; values stay in their original
    basis. H128 intermediates are BFP8, not exact real-valued rotations. H256
    combines them in the caller dtype; this widening cannot recover lost bits.
    Prefill casts K to cache dtype; decode cache updates receive caller BF16.

    # Specification
    - requires: Blackhole and head dimension 128 or 256.
    - ensures: normalized Sylvester rotation consumes its input and preserves shape, dtype and memory.
    - panics: none.

    # Adequacy
    H256 head/padding witnesses compare against independent float64 arithmetic;
    model prefill and decode exercise temporary lifetimes and cache writes.
    """

    def __init__(self, mesh_device, head_dim):
        """Select the supported rotation width.

        # Specification
        - ensures: stores the selected head dimension without allocating a matrix.
        - fails: ValueError for non-Blackhole or unsupported head dimensions.
        - panics: none.

        # Adequacy
        Device witnesses cover admitted widths; guards delimit the primitive domain.
        """
        if mesh_device.arch() != ttnn.device.Arch.BLACKHOLE:
            raise ValueError("Hadamard rotation requires Blackhole")
        if head_dim not in (128, 256):
            raise ValueError("Hadamard rotation supports head dimensions 128 and 256")
        self.head_dim = head_dim

    def __call__(self, tensor):
        """Consume an unrotated tensor; preserve its shape, dtype and memory.

        # Specification
        - requires: nonempty interleaved tiled tensor on Blackhole, with the selected width.
        - ensures: returns normalized H128 or Sylvester-composed H256 in caller dtype.
        - fails: ValueError for width mismatch or rejected H128 input; runtime allocation/dispatch errors propagate.
        - panics: none.

        # Adequacy
        Float64 head/padding witnesses detect half-order, sign and normalization faults.
        Full-model prefill additionally detects L1 lifetime interference with attention.
        """
        if tensor.shape[-1] != self.head_dim:
            raise ValueError("Hadamard input width must match the head dimension")
        dtype, memory = tensor.dtype, tensor.memory_config()
        source = tensor if dtype == ttnn.bfloat16 else ttnn.typecast(tensor, ttnn.bfloat16)
        if self.head_dim == 128:
            rotated = hadamard_h128(source)
            result = ttnn.typecast(rotated, dtype)
            if dtype != rotated.dtype:
                ttnn.deallocate(rotated)
        else:
            start = [0] * len(source.shape)
            end = list(source.shape)
            end[-1] = 128
            left = ttnn.slice(source, start, end)
            start[-1], end[-1] = 128, 256
            right = ttnn.slice(source, start, end)
            if source is not tensor:
                ttnn.deallocate(source)
            ttnn.deallocate(tensor)
            a = hadamard_h128(left)
            ttnn.deallocate(left)
            b = hadamard_h128(right)
            ttnn.deallocate(right)
            # H256 = [[H128,H128],[H128,-H128]] / sqrt(2),
            # with each H128 already normalized. Widen only the combination
            # to the caller dtype; the primitive result remains BFP8.
            summed = ttnn.add(a, b, dtype=dtype, memory_config=memory)
            difference = ttnn.subtract(a, b, dtype=dtype, memory_config=memory)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            plus = ttnn.multiply(summed, 2**-0.5, memory_config=memory)
            minus = ttnn.multiply(difference, 2**-0.5, memory_config=memory)
            ttnn.deallocate(summed)
            ttnn.deallocate(difference)
            result = ttnn.concat([plus, minus], dim=-1, memory_config=memory)
            ttnn.deallocate(plus)
            ttnn.deallocate(minus)
        if self.head_dim == 128:
            if source is not tensor:
                ttnn.deallocate(source)
            ttnn.deallocate(tensor)
        return result
