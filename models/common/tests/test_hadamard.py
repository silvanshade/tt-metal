# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Blackhole H128 packing, normalization, padding and trace witnesses.

The host oracle is an independent float64 Sylvester matrix. Boundary vectors
mirror the LLK sign/exponent/full-mantissa cases; tiled padding and H256 exercise
the model adapter. These observations target row/face addressing, sign/order,
normalization, stale replay output and subsequent SFPU initialization faults.
"""

import pytest
import torch

import ttnn
from models.common.hadamard import HadamardRotation, hadamard_h128

pytestmark = [
    pytest.mark.usefixtures("require_blackhole_mesh_device"),
    pytest.mark.parametrize("ttnn_mesh_device", [{"mesh_shape": (1, 1), "trace_region_size": 1048576}], indirect=True),
]


def _reference(values):
    """Compute the normalized transform independently in float64.

    # Specification
    - requires: the final dimension is a positive power of two.
    - ensures: returns values multiplied by the normalized Sylvester matrix.
    - panics: none.

    # Adequacy
    Basis vectors expose ordering/sign faults; scale families expose missing
    normalization. This oracle does not model intermediate device rounding.
    """
    width = values.shape[-1]
    matrix = torch.ones(1, 1, dtype=torch.float64)
    while matrix.shape[0] < width:
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    return values.double() @ (matrix / width**0.5)


def test_h128_signs_exponents_and_basis(ttnn_mesh_device):
    generator = torch.Generator().manual_seed(303)
    signs = (torch.randint(0, 2, (5, 128), generator=generator) * 2 - 1).float()
    scales = torch.tensor([2.0**e for e in (-8, -1, 0, 1, 8)])[:, None]
    values = torch.cat((torch.eye(128), signs * scales), 0).bfloat16().reshape(1, 1, 133, 128)
    source = ttnn.from_torch(values, device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    result = hadamard_h128(source)
    try:
        assert result.dtype == ttnn.bfloat8_b
        actual = ttnn.to_torch(result).double()
        expected = _reference(values)
        bound = expected.square().mean(-1, keepdim=True).sqrt() * 0.04
        assert torch.all((actual - expected).abs() <= bound)
    finally:
        ttnn.deallocate(result)
        ttnn.deallocate(source)


def test_h128_full_mantissa_and_batch_padding(ttnn_mesh_device):
    values = torch.randn((1, 3, 65, 128), generator=torch.Generator().manual_seed(606)).bfloat16()
    source = ttnn.from_torch(values, device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    result = hadamard_h128(source)
    try:
        actual = ttnn.to_torch(result).double()
        expected = _reference(values)
        pcc = torch.corrcoef(torch.stack((actual.flatten(), expected.flatten())))[0, 1]
        assert pcc > 0.9999
        bound = expected.square().mean(-1, keepdim=True).sqrt() * 0.06
        assert torch.all((actual - expected).abs() <= bound)
    finally:
        ttnn.deallocate(result)
        ttnn.deallocate(source)


def test_h128_trace_replay_and_following_sfpu(ttnn_mesh_device):
    device = ttnn_mesh_device
    values = torch.randn((1, 1, 37, 128), generator=torch.Generator().manual_seed(404)).bfloat16()
    source = ttnn.from_torch(values, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    poison = ttnn.from_torch(torch.zeros_like(values), device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    warm = hadamard_h128(source)
    expected = ttnn.to_torch(warm).clone()
    warm_sfpu = ttnn.silu(source)
    expected_sfpu = ttnn.to_torch(warm_sfpu).clone()
    # Compile the poison copy before parking the trace, too.
    ttnn.copy(poison, warm)
    ttnn.deallocate(warm)
    ttnn.deallocate(warm_sfpu)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    result = hadamard_h128(source)
    following = ttnn.silu(source)
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for _ in range(3):
            ttnn.copy(poison, result)
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            assert torch.equal(ttnn.to_torch(result), expected)
            actual_sfpu = ttnn.to_torch(following)
            assert torch.equal(actual_sfpu, expected_sfpu)
            torch.testing.assert_close(
                actual_sfpu.float(), torch.nn.functional.silu(values.float()), rtol=0.02, atol=0.02
            )
    finally:
        ttnn.release_trace(device, trace)
        for tensor in (source, poison, result, following):
            ttnn.deallocate(tensor)


@pytest.mark.parametrize("shape", [(1, 1, 32, 256), (1, 1, 4, 256), (1, 3, 65, 256)])
def test_h256_sylvester_composition(ttnn_mesh_device, shape):
    values = torch.randn(shape, generator=torch.Generator().manual_seed(352)).bfloat16()
    source = ttnn.from_torch(
        values,
        device=ttnn_mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    result = HadamardRotation(ttnn_mesh_device, 256)(source)
    try:
        assert result.dtype == ttnn.bfloat16
        assert result.memory_config() == ttnn.L1_MEMORY_CONFIG
        actual = ttnn.to_torch(result).double()
        expected = _reference(values)
        pcc = torch.corrcoef(torch.stack((actual.flatten(), expected.flatten())))[0, 1]
        assert pcc > 0.9999
        bound = expected.square().mean(-1, keepdim=True).sqrt() * 0.06
        assert torch.all((actual - expected).abs() <= bound)
    finally:
        ttnn.deallocate(result)
