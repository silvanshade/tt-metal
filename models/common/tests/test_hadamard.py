# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Blackhole Hadamard rotation: order, sign, normalization, addressing and replay.

The host oracle is an independent float64 Sylvester matrix; the device oracle is
the dense H256 matmul this program replaces. Boundary vectors separate sign,
column order and normalization faults; the served shapes and a many-block shape
separate the page addressing of the one-block split from the row-block split;
poisoned replays separate stale output from recomputation.
"""

import pytest
import torch

import ttnn
from models.common.hadamard import HadamardRotation, hadamard_rotate, stage_tiles

pytestmark = [
    pytest.mark.usefixtures("require_blackhole_mesh_device"),
    pytest.mark.parametrize("ttnn_mesh_device", [{"mesh_shape": (1, 1), "trace_region_size": 1048576}], indirect=True),
]

SERVED_SHAPES = [(1, 1, 24, 256), (1, 1, 4, 256), (1, 6, 24, 256), (1, 6, 4, 256)]
"""Q and K at decode and at the six-row MTP verify, the shapes the served matmul runs."""


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


def _rotate(device, values, dtype=ttnn.bfloat16, out_dtype=None):
    """Run the rotation program over host values and return its device result."""
    source = ttnn.from_torch(values, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    stage = stage_tiles(device, values.shape[-1])
    try:
        return source, stage, hadamard_rotate(source, stage, out_dtype)
    except Exception:
        ttnn.deallocate(stage)
        ttnn.deallocate(source)
        raise


@pytest.mark.parametrize("width", [128, 256])
def test_basis_signs_and_exponents(ttnn_mesh_device, width):
    """Each basis row must return its own Hadamard column, normalized, at every exponent."""
    generator = torch.Generator().manual_seed(303)
    signs = (torch.randint(0, 2, (5, width), generator=generator) * 2 - 1).float()
    scales = torch.tensor([2.0**e for e in (-8, -1, 0, 1, 8)])[:, None]
    values = torch.cat((torch.eye(width), signs * scales), 0).bfloat16().reshape(1, 1, width + 5, width)
    source, stage, result = _rotate(ttnn_mesh_device, values)
    try:
        actual = ttnn.to_torch(result).double()
        expected = _reference(values)
        bound = expected.square().mean(-1, keepdim=True).sqrt() * 0.04
        assert torch.all((actual - expected).abs() <= bound)
    finally:
        for tensor in (result, stage, source):
            ttnn.deallocate(tensor)


def test_many_blocks_and_partial_rows(ttnn_mesh_device):
    """A shape with more row blocks than cores must address every page it owns."""
    values = torch.randn((1, 3, 4001, 256), generator=torch.Generator().manual_seed(606)).bfloat16()
    source, stage, result = _rotate(ttnn_mesh_device, values)
    try:
        actual = ttnn.to_torch(result).double()
        expected = _reference(values)
        pcc = torch.corrcoef(torch.stack((actual.flatten(), expected.flatten())))[0, 1]
        assert pcc > 0.9999
        bound = expected.square().mean(-1, keepdim=True).sqrt() * 0.06
        assert torch.all((actual - expected).abs() <= bound)
    finally:
        for tensor in (result, stage, source):
            ttnn.deallocate(tensor)


def test_stage_primitive_matches_the_stock_matmul_path(ttnn_mesh_device, monkeypatch):
    """The two-phase stage primitive must return exactly what three phases return.

    Phase 1 multiplies the stage operand's absent low mantissa half, so skipping
    it is exact: a wrong phase step lands the second pass on phase 1 instead of
    phase 2, drops the activation's low half, and breaks this equality.
    """
    values = torch.randn((1, 1, 32, 256), generator=torch.Generator().manual_seed(909)).bfloat16()
    results = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("QWEN_HADAMARD_STAGE_MOP", flag)
        source, stage, result = _rotate(ttnn_mesh_device, values)
        try:
            results[flag] = ttnn.to_torch(result).clone()
        finally:
            for tensor in (result, stage, source):
                ttnn.deallocate(tensor)
    assert torch.equal(results["0"], results["1"])
    expected = _reference(values)
    bound = expected.square().mean(-1, keepdim=True).sqrt() * 0.04
    assert torch.all((results["1"].double() - expected).abs() <= bound)


@pytest.mark.parametrize("shape", SERVED_SHAPES)
def test_served_shapes_against_the_dense_matmul(ttnn_mesh_device, shape):
    """The program must agree with the dense H256 matmul it replaces, shape by shape."""
    values = torch.randn(shape, generator=torch.Generator().manual_seed(352)).bfloat16()
    matrix = torch.ones(1, 1, dtype=torch.float32)
    while matrix.shape[0] < shape[-1]:
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    dense = ttnn.from_torch(
        matrix * shape[-1] ** -0.5, device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    source, stage, result = _rotate(ttnn_mesh_device, values)
    served_input = ttnn.from_torch(values, device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    served = ttnn.matmul(
        served_input,
        dense,
        dtype=ttnn.bfloat16,
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            ttnn_mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    try:
        actual, matmul_result = ttnn.to_torch(result).double(), ttnn.to_torch(served).double()
        expected = _reference(values)
        for other in (matmul_result, expected):
            pcc = torch.corrcoef(torch.stack((actual.flatten(), other.flatten())))[0, 1]
            assert pcc > 0.9999
            bound = expected.square().mean(-1, keepdim=True).sqrt() * 0.06
            assert torch.all((actual - other).abs() <= bound)
    finally:
        for tensor in (served, served_input, dense, result, stage, source):
            ttnn.deallocate(tensor)


def test_trace_replay_and_following_sfpu(ttnn_mesh_device):
    """A replayed program must recompute its output, and leave the next SFPU op intact."""
    device = ttnn_mesh_device
    values = torch.randn((1, 1, 37, 256), generator=torch.Generator().manual_seed(404)).bfloat16()
    source = ttnn.from_torch(values, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    stage = stage_tiles(device, 256)
    poison = ttnn.from_torch(torch.zeros_like(values), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    warm = hadamard_rotate(source, stage)
    expected = ttnn.to_torch(warm).clone()
    warm_sfpu = ttnn.silu(source)
    expected_sfpu = ttnn.to_torch(warm_sfpu).clone()
    # Compile the poison copy before parking the trace, too.
    ttnn.copy(poison, warm)
    ttnn.deallocate(warm)
    ttnn.deallocate(warm_sfpu)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    result = hadamard_rotate(source, stage)
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
        for tensor in (source, stage, poison, result, following):
            ttnn.deallocate(tensor)


@pytest.mark.parametrize("head_dim", [128, 256])
def test_rotation_consumes_input_and_keeps_dtype(ttnn_mesh_device, head_dim):
    """The model adapter must consume its input and return the caller's dtype and memory."""
    values = torch.randn((1, 3, 65, head_dim), generator=torch.Generator().manual_seed(128)).bfloat16()
    rotation = HadamardRotation(ttnn_mesh_device, head_dim)
    for dtype in (ttnn.bfloat16, ttnn.bfloat8_b):
        source = ttnn.from_torch(
            values,
            device=ttnn_mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        expected = _reference(ttnn.to_torch(source))
        result = rotation(source)
        try:
            assert not source.is_allocated()
            assert result.dtype == dtype
            assert tuple(result.shape) == tuple(values.shape)
            assert result.memory_config() == ttnn.L1_MEMORY_CONFIG
            actual = ttnn.to_torch(result).double()
            pcc = torch.corrcoef(torch.stack((actual.flatten(), expected.flatten())))[0, 1]
            assert pcc > 0.9999
            bound = expected.square().mean(-1, keepdim=True).sqrt() * 0.06
            assert torch.all((actual - expected).abs() <= bound)
        finally:
            ttnn.deallocate(result)


def test_rejected_inputs(ttnn_mesh_device, expect_error):
    """Each rejected input class must be refused rather than silently rotated."""
    stage = stage_tiles(ttnn_mesh_device, 256)
    values = torch.randn((1, 1, 32, 256), generator=torch.Generator().manual_seed(77)).bfloat16()
    row_major = ttnn.from_torch(values, device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    quantized = ttnn.from_torch(values, device=ttnn_mesh_device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    unsupported = ttnn.from_torch(
        torch.randn((1, 1, 32, 96), generator=torch.Generator().manual_seed(78)).bfloat16(),
        device=ttnn_mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    try:
        for tensor, message in (
            (row_major, "tiled BF16"),
            (quantized, "tiled BF16"),
            (unsupported, "power of two"),
        ):
            with expect_error(ValueError, message):
                hadamard_rotate(tensor, stage)
        with expect_error(ValueError, "head dimensions"):
            HadamardRotation(ttnn_mesh_device, 64)
    finally:
        for tensor in (unsupported, quantized, row_major, stage):
            ttnn.deallocate(tensor)
