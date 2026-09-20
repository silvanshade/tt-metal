# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Both-NoC in1 weight reads are bit-identical to the single-NoC path.

The 1D `mcast_in0` matmul reads its in1 weights one tile at a time from interleaved DRAM, and under
NOC_MODE::DM_DYNAMIC_NOC those requests alternate across both NoCs to clear the Blackhole
per-endpoint request-rate limit (tenstorrent/tt-metal#55898). A read carries no torus direction, so
which NoC it rides must not change one output byte. PCC would hide a misrouted, duplicated or
dropped weight tile behind the reduction over K, so the assertion here is exact equality.
"""

import pytest
import torch
import ttnn

# Shaped after the served vocabulary projection: in0 one row of activations from interleaved DRAM,
# in1 a bfloat4_b weight block from interleaved DRAM, in0 multicast and in1 read per core. N is cut
# to keep the weight under 50 MB; the read loop and its addressing are the served ones.
K = 5120
N = 16384
GRID = ttnn.CoreCoord(8, 4)


def _run_matmul(device, *, both_noc, monkeypatch, in0, in1, bias, in1_dtype):
    """Runs one arm and returns its raw device output.

    The env override is read per program build, so the program cache has to be cleared between the
    arms or the second one would replay the first one's kernels and the comparison would be vacuous.
    """
    monkeypatch.setenv("TT_MATMUL_IN1_BOTH_NOC", "1" if both_noc else "0")
    device.disable_and_clear_program_cache()

    num_cores = GRID.x * GRID.y
    in0_t = ttnn.from_torch(
        in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    in1_t = ttnn.from_torch(
        in1, dtype=in1_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    bias_t = (
        None
        if bias is None
        else ttnn.from_torch(
            bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=GRID,
        in0_block_w=K // 32 // num_cores,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=N // 32 // num_cores,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    out_t = ttnn.linear(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
    )
    out = ttnn.to_torch(out_t).clone()
    ttnn.deallocate(out_t)
    return out


@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat4_b, ttnn.bfloat8_b], ids=["in1_bfp4", "in1_bfp8"])
@pytest.mark.parametrize("has_bias", [False, True], ids=["no_bias", "bias"])
@pytest.mark.timeout(300)
def test_matmul_in1_both_noc_bit_identity(device, in1_dtype, has_bias, monkeypatch):
    if not ttnn.device.is_blackhole(device):
        pytest.skip("Both-NoC in1 reads are gated on Blackhole")

    torch.manual_seed(1234)
    in0 = torch.randn([1, 1, 32, K]).bfloat16().float()
    in1 = torch.randn([1, 1, K, N]).bfloat16().float()
    # The bias block is read by the same loop, on its own barrier, so it gets its own arm.
    bias = torch.randn([1, 1, 32, N]).bfloat16().float() if has_bias else None

    arm = dict(monkeypatch=monkeypatch, in0=in0, in1=in1, bias=bias, in1_dtype=in1_dtype)
    try:
        single_noc = _run_matmul(device, both_noc=False, **arm)
        both_noc = _run_matmul(device, both_noc=True, **arm)
    finally:
        # The arms run with the cache off; leaving it off would slow every later test on this device.
        device.enable_program_cache()

    # Both arms must have produced real output, not a zeroed or skipped program: a universal
    # equality over two empty results would pass while proving nothing.
    assert torch.count_nonzero(single_noc) > 0, "single-NoC arm produced an all-zero output"
    assert torch.equal(single_noc, both_noc), (
        f"both-NoC output differs from single-NoC: "
        f"{torch.count_nonzero(single_noc != both_noc).item()} of {single_noc.numel()} elements, "
        f"max abs delta {(single_noc - both_noc).abs().max().item()}"
    )
