# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Both-NoC paged decode reads are bit-identical to the single-NoC path.

The paged decode reader alternates its per-tile K/V DRAM requests across both NoCs under
NOC_MODE::DM_DYNAMIC_NOC, to clear the Blackhole per-endpoint request-rate limit
(tenstorrent/tt-metal#55898). A read carries no torus direction, so which NoC it rides must not
change one output byte. PCC would hide a misrouted, duplicated or dropped tile behind attention's
own averaging, so the assertion here is exact equality.
"""

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import (
    fa_rand,
    get_chunk_size,
    nearest_n,
    nearest_pow_2,
)

# Served Qwen3.8-27B full-attention shape: 4 KV heads, 24 query heads, head_dim 256, block 64.
SERVED_NKV = 4
SERVED_NH = 24
SERVED_D = 256
SERVED_BLOCK_SIZE = 64


def _to_paged_cache(cache, batch, num_kv, max_num_blocks_per_seq, block_size, head_dim):
    return (
        cache.reshape(batch, num_kv, max_num_blocks_per_seq, block_size, head_dim)
        .transpose(1, 2)
        .reshape(batch * max_num_blocks_per_seq, num_kv, block_size, head_dim)
    )


def _run_paged_decode(device, *, both_noc, monkeypatch, k, v, q, page_table, kv_dtype, grid_size, cur_pos, s):
    """Runs one paged decode arm and returns its raw device output as a torch tensor.

    `both_noc` selects the arm through TT_SDPA_DECODE_BOTH_NOC. The program cache is keyed on the
    operation attributes and not on the kernel defines, so it is cleared here: a cache hit would
    silently rerun the first arm's program and make the comparison vacuous.
    """
    monkeypatch.setenv("TT_SDPA_DECODE_BOTH_NOC", "1" if both_noc else "0")
    device.disable_and_clear_program_cache()

    b, nkv = k.shape[0], k.shape[1]
    nh, d = q.shape[2], q.shape[3]
    max_num_blocks_per_seq = s // SERVED_BLOCK_SIZE

    paged_k = _to_paged_cache(k, b, nkv, max_num_blocks_per_seq, SERVED_BLOCK_SIZE, d)
    paged_v = _to_paged_cache(v, b, nkv, max_num_blocks_per_seq, SERVED_BLOCK_SIZE, d)

    tt_k = ttnn.as_tensor(
        paged_k, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_v = ttnn.as_tensor(
        paged_v, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_q = ttnn.as_tensor(
        q, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_page_table = ttnn.Tensor(page_table, ttnn.int32).to(device)
    tt_cur_pos = ttnn.Tensor(torch.tensor([cur_pos] * b), ttnn.int32).to(device)

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=padded_num_heads,
        k_chunk_size=get_chunk_size(cur_pos + 1, s),
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    tt_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt_q,
        tt_k,
        tt_v,
        tt_page_table,
        cur_pos_tensor=tt_cur_pos,
        scale=d**-0.5,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.to_torch(tt_out)[:, :, :nh, :].clone()
    ttnn.deallocate(tt_out)
    return out


@pytest.mark.parametrize("kv_dtype", [ttnn.bfloat4_b, ttnn.bfloat8_b], ids=["kv_bfp4", "kv_bfp8"])
@pytest.mark.parametrize("b, s, grid_size", [(1, 1024, ttnn.CoreCoord(8, 4))], ids=["b1_s1024"])
@pytest.mark.timeout(180)
def test_sdpa_decode_both_noc_bit_identity(device, kv_dtype, b, s, grid_size, monkeypatch):
    if not ttnn.device.is_blackhole(device):
        pytest.skip("Both-NoC decode reads are gated on Blackhole")

    torch.manual_seed(1234)
    nkv, nh, d = SERVED_NKV, SERVED_NH, SERVED_D
    cur_pos = s // 2 - 1

    k = fa_rand(b, nkv, s, d)
    v = fa_rand(b, nkv, s, d)
    q = fa_rand(1, b, nh, d)

    max_num_blocks = b * s // SERVED_BLOCK_SIZE
    # Shuffled page table: consecutive virtual tiles land on unrelated physical pages, so a NoC
    # that resolved the wrong page would not accidentally read the neighbouring correct one.
    permutation = torch.randperm(max_num_blocks)
    page_table = torch.argsort(permutation).reshape(b, s // SERVED_BLOCK_SIZE)

    arm = dict(monkeypatch=monkeypatch, k=k, v=v, q=q, page_table=page_table, kv_dtype=kv_dtype, s=s)
    try:
        single_noc = _run_paged_decode(device, both_noc=False, grid_size=grid_size, cur_pos=cur_pos, **arm)
        both_noc = _run_paged_decode(device, both_noc=True, grid_size=grid_size, cur_pos=cur_pos, **arm)
    finally:
        # The arms run with the cache off; leaving it off would slow every later test on this device.
        device.enable_program_cache()

    # Both arms must have produced real attention output, not a zeroed or skipped program: a
    # universal equality over two empty results would pass while proving nothing.
    assert torch.count_nonzero(single_noc) > 0, "single-NoC arm produced an all-zero output"
    assert torch.equal(single_noc, both_noc), (
        f"both-NoC output differs from single-NoC: "
        f"{torch.count_nonzero(single_noc != both_noc).item()} of {single_noc.numel()} elements, "
        f"max abs delta {(single_noc - both_noc).abs().max().item()}"
    )
