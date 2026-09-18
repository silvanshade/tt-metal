# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "k,n,tp_dim",
    [
        (5120, 17408, "n"),
        (17408, 5120, "k"),
        (5120, 14336, "n"),
        (5120, 16480, "n"),
        (6144, 5120, "k"),
    ],
    ids=["mlp_gate_up", "mlp_down", "attn_qkv", "gdn_qkvzab", "out_proj"],
)
@pytest.mark.parametrize("tp", [1, 4])
@pytest.mark.parametrize("cols,split_width", [(11, False), (7, True)])
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_matmul_dram_width_sharded_prefill(
    mesh_device: ttnn.MeshDevice, k: int, n: int, tp_dim: str, tp: int, cols: int, split_width: bool
) -> None:
    """Prefill reads partial and crossing bank shards independently of worker width."""
    grid = mesh_device.compute_with_storage_grid_size()
    if grid.x < cols or grid.y < 8:
        pytest.skip(f"Requires a {cols}x8 worker grid")

    # Model-parallel dimensions are local to each device; the reader never receives TP.
    if tp_dim == "n":
        n //= tp
    else:
        k //= tp
    banks = mesh_device.dram_grid_size().x
    shard_width = math.ceil(n / (32 * banks)) * 32
    # Match bank-padded weights used for decode and consumed unchanged during prefill.
    padded_n = shard_width * banks
    per_core_n = math.ceil(padded_n / (32 * cols))
    if split_width:
        per_core_n = math.ceil(per_core_n / 2) * 2
    out_block_w = per_core_n // 2 if split_width else per_core_n
    assert per_core_n % (shard_width // 32) != 0

    torch.manual_seed(17)
    weight = torch.randn(k, n, dtype=torch.bfloat16) * 0.02
    activation = torch.randn(1, 1, 2048, k, dtype=torch.bfloat16)
    expected = activation.float() @ weight.float()
    weight = torch.nn.functional.pad(weight, (0, padded_n - n))
    weight_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
            [k, shard_width],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    weight_tt = ttnn.from_torch(
        weight,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=weight_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    activation_tt = ttnn.from_torch(
        activation,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(cols, 8),
        in0_block_w=1 if n > 5120 else 4,
        out_subblock_h=1,
        out_subblock_w=1,
        # Multiple M blocks exercise restarting K without slicing/copying activations.
        out_block_h=2,
        out_block_w=out_block_w,
        per_core_M=8,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fuse_batch=False,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    output_tt = ttnn.matmul(
        activation_tt,
        weight_tt,
        program_config=program_config,
        compute_kernel_config=compute_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    for output in ttnn.get_device_tensors(output_tt):
        actual = ttnn.to_torch(output)[..., :n]
        assert torch.isfinite(actual).all(), "Prefill produced non-finite output from sharded weights"
        assert_with_pcc(expected, actual, 0.99)
