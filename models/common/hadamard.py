# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Orthogonal rotation for queries and quantized attention keys."""

import os
import struct
from math import prod
from pathlib import Path

import torch

import ttnn

TILE = 32
"""Side of the device tile, and the width of the stage the rotation factors through."""


def _stage_mop():
    """Whether the compute kernel takes the two-phase Hadamard stage primitive.

    The stage operand is a signed power of two, so only two of the four fidelity
    phases contribute; the primitive issues exactly those two where the stock
    matmul issues HiFi3's three. `QWEN_HADAMARD_STAGE_MOP=0` selects the stock
    path, so the two can be measured against each other without a rebuild.
    """
    return os.environ.get("QWEN_HADAMARD_STAGE_MOP", "1") == "1"


def _sylvester(order):
    """Return the unnormalized Sylvester matrix of the given power-of-two order.

    # Specification
    - requires: order is a positive power of two.
    - ensures: returns the float32 symmetric matrix of the Sylvester recursion.
    - panics: none.
    """
    matrix = torch.ones(1, 1, dtype=torch.float32)
    while matrix.shape[0] < order:
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    return matrix


def _scale_folds(width):
    """Whether 1/sqrt(width) is a power of two, so the stage tile can carry it exactly."""
    return (width.bit_length() - 1) % 2 == 0


def _check_width(width):
    """Reject a width the H_m x H_32 factorization does not cover."""
    if width < TILE or width > TILE * TILE or width & (width - 1):
        raise ValueError("Hadamard width must be a power of two between 32 and 1024")


def stage_tiles(device, width, mesh_mapper=None):
    """Build the signed stage pair the rotation program multiplies by.

    H_w factors as H_(w/32) (x) H_32, so the whole rotation is expressed by one
    32x32 tile and its negation: 4 KB of constant against the 128 KB of a dense
    H_256 matrix. Page 0 is +H_32, page 1 is -H_32, and the normalization rides
    on them whenever 1/sqrt(width) is a power of two and therefore exact.

    # Specification
    - requires: an open device and a width the factorization covers.
    - ensures: returns one resident BF16 [2, 32, 32] tensor; the caller owns its lifetime.
    - fails: ValueError for an unsupported width.
    - panics: none.

    # Adequacy
    Host float64 witnesses over both stages distinguish a sign, order or
    normalization fault in the stage pair from one in the program that uses it.
    """
    _check_width(width)
    stage = _sylvester(TILE)
    if _scale_folds(width):
        stage = stage * width**-0.5
    return ttnn.from_torch(
        torch.stack((stage, -stage)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )


def _residual_scale(width):
    """Return the fp32 bit pattern of the scale the stage tiles could not carry."""
    if _scale_folds(width):
        return 0
    return struct.unpack("<I", struct.pack("<f", width**-0.5))[0]


def _partition(blocks, tiles, cores):
    """Split the work into groups of output tiles, widest split the row blocks leave.

    A row block is 32 rows and `tiles` tiles wide, and every output tile of a
    block reads all of its input tiles. Row blocks therefore parallelize for
    free and column groups parallelize at the cost of re-reading the block, so
    the split grows only while row blocks alone leave cores idle.
    """
    split = 1 << (min(tiles, max(1, cores // blocks)).bit_length() - 1)
    return split, tiles // split, min(blocks * split, cores)


def hadamard_rotate(tensor, stage, dtype=None):
    """Rotate every row of a tiled activation by the normalized Sylvester matrix.

    One program: each worker accumulates one output tile per group as
    sum_k X_k * (H_m[j, k] H_32), reading its row block's tiles whole. Nothing
    is exchanged between workers and nothing is gathered inside a tile, so the
    kernel issues only whole-tile reads and the rotation costs one dispatch.

    # Specification
    - requires: nonempty tiled BF16 device tensor on Blackhole, standard tiles,
      interleaved memory at power-of-two widths 32 through 1024 or L1 height-sharded
      memory with whole 256-wide rows, and a stage pair built for that width.
    - ensures: input is retained; the normalized rotation returns in `dtype` with the
      input's shape and memory configuration.
    - fails: ValueError for non-Blackhole, non-tiled/BF16, nonstandard tile, unsupported
      width or shard shape.
    - panics: none.

    # Adequacy
    Host float64 basis, sign, exponent and full-mantissa witnesses distinguish
    sign, tile-order and normalization faults; multi-block and partial-row cases
    distinguish page-addressing faults; poisoned trace replays detect stale output.
    Finite witnesses do not establish a universal error bound.
    """
    device = tensor.device()
    if device.arch() != ttnn.device.Arch.BLACKHOLE:
        raise ValueError("Hadamard rotation requires Blackhole")
    if tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("Hadamard rotation requires tiled BF16 input")
    if tuple(tensor.tile.tile_shape) != (TILE, TILE):
        raise ValueError("Hadamard rotation requires standard 32x32 tiles")
    if tensor.is_sharded():
        memory = tensor.memory_config()
        shard = memory.shard_spec
        if (
            memory.memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            or memory.buffer_type != ttnn.BufferType.L1
            or int(tensor.shape[-1]) != 256
            or shard.shape[1] != 256
            or shard.shape[0] % TILE
        ):
            raise ValueError("Hadamard sharded input requires L1 height-sharded whole 256-wide rows")
    width = int(tensor.shape[-1])
    _check_width(width)
    tiles = width // TILE
    output = ttnn.empty(
        tensor.shape,
        dtype=dtype or tensor.dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=tensor.memory_config(),
    )
    blocks = prod(tensor.padded_shape) // (TILE * width)
    grid = device.compute_with_storage_grid_size()
    split, group_tiles, workers = _partition(blocks, tiles, grid.x * grid.y)
    coordinates = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
    # Whole rows plus one partial row: dispatch multicasts per rectangle, not per core.
    cores = ttnn.num_cores_to_corerangeset(workers, grid, True)
    read_args, write_args, compute_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    groups, first = blocks * split, 0
    for index, core in enumerate(coordinates):
        count = groups // workers + (index < groups % workers)
        read_args[core.x][core.y] = [tensor.buffer_address(), stage.buffer_address(), first, count]
        write_args[core.x][core.y] = [output.buffer_address(), first, count]
        compute_args[core.x][core.y] = [first, count]
        first += count
    tile = ttnn.Tile([TILE, TILE])
    cbs = []
    for index, page_dtype, pages in [
        (0, ttnn.bfloat16, 2 * tiles),
        (1, ttnn.bfloat16, 2),
        (16, output.dtype, 2 * group_tiles),
    ]:
        size = tile.get_tile_size(page_dtype)
        cbs.append(
            ttnn.CBDescriptor(
                total_size=pages * size,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=index, data_format=page_dtype, page_size=size, tile=ttnn.TileDescriptor(tile)
                    )
                ],
            )
        )
    shape_args = [tiles, group_tiles, split]
    read_compile = [*shape_args, *ttnn.TensorAccessorArgs(tensor).get_compile_time_args()]
    read_compile.extend(ttnn.TensorAccessorArgs(stage).get_compile_time_args())
    write_compile = [*shape_args, *ttnn.TensorAccessorArgs(output).get_compile_time_args()]
    kernels = Path(__file__).with_name("hadamard_kernels")
    descriptor = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=str(kernels / "reader.cpp"),
                core_ranges=cores,
                compile_time_args=read_compile,
                runtime_args=read_args,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1,
                    noc=ttnn.NOC.RISCV_1_default,
                    noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(kernels / "writer.cpp"),
                core_ranges=cores,
                compile_time_args=write_compile,
                runtime_args=write_args,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0,
                    noc=ttnn.NOC.RISCV_0_default,
                    noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(kernels / "compute.cpp"),
                core_ranges=cores,
                compile_time_args=[*shape_args, _residual_scale(width), int(_stage_mop())],
                runtime_args=compute_args,
                # A stage entry is a signed power of two, so its low mantissa half is
                # zero and the fidelity phases that read it contribute nothing: HiFi3
                # is bit-identical to HiFi4 here and runs one pass fewer.
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False, fp32_dest_acc_en=True
                ),
            ),
        ],
        cbs=cbs,
        semaphores=[],
    )
    ttnn.generic_op([tensor, stage, output], descriptor)
    return output


class HadamardRotation:
    """Apply a normalized Sylvester rotation to queries and keys.

    Apply after RoPE and before key quantization; values stay in their original
    basis. The rotation is one program over the tiles it is given, holding one
    32x32 stage pair rather than a dense width x width matrix.

    # Specification
    - requires: Blackhole and a head dimension the factorization covers.
    - ensures: the rotation consumes its input and preserves shape, dtype and memory.
    - panics: none.

    # Adequacy
    Float64 head and padding witnesses compare against independent arithmetic;
    model prefill and decode exercise temporary lifetimes and cache writes.
    """

    def __init__(self, mesh_device: ttnn.MeshDevice, head_dim: int) -> None:
        """Allocate the persistent stage pair for this head dimension.

        # Specification
        - ensures: one 4 KB stage pair is shared by every call.
        - fails: ValueError for non-Blackhole or an unsupported head dimension; runtime
          allocation errors propagate.
        - panics: none.

        # Adequacy
        Device witnesses cover the admitted widths; guards delimit the domain.
        """
        if mesh_device.arch() != ttnn.device.Arch.BLACKHOLE:
            raise ValueError("Hadamard rotation requires Blackhole")
        if head_dim not in (128, 256):
            raise ValueError("Hadamard rotation supports head dimensions 128 and 256")
        self.head_dim = head_dim
        self.stage = stage_tiles(mesh_device, head_dim, ttnn.ReplicateTensorToMesh(mesh_device))

    def __call__(self, tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Consume an unrotated tensor; preserve its shape, dtype and memory.

        # Specification
        - requires: nonempty tiled tensor on Blackhole, with the selected width and
          an interleaved or H256 L1 height-sharded layout admitted by `hadamard_rotate`.
        - ensures: returns the normalized rotation in the caller's dtype.
        - fails: ValueError for width mismatch or rejected input; runtime allocation and
          dispatch errors propagate.
        - panics: none.

        # Adequacy
        Float64 head and padding witnesses detect half-order, sign and normalization
        faults. Full-model prefill additionally detects L1 lifetime interference with
        attention.
        """
        if tensor.shape[-1] != self.head_dim:
            raise ValueError("Hadamard input width must match the head dimension")
        dtype = tensor.dtype
        source = tensor if dtype == ttnn.bfloat16 else ttnn.typecast(tensor, ttnn.bfloat16)
        result = hadamard_rotate(source, self.stage, dtype)
        if source is not tensor:
            ttnn.deallocate(source)
        ttnn.deallocate(tensor)
        return result
