# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decode matmul layout selection in Qwen38ModelArgs — host only, no device.

QWEN38_DECODE_MATMUL picks the arm for every decode projection; QWEN38_QKVZAB_LAYOUT overrides it
for the GDN [dim, qkvzab] in-projection alone. The shape's padded width is 520 tiles, so 520/8 banks
is odd and the sharded builder drops it to one reader per bank: measured on p150a at 23K the 48
sharded in-projections read at 165.1 GB/s against 274 GB/s on the 1D arm, so the override defaults
to 1d while keeping the sharded arm reachable for a later two-reader measurement.

Runs on a stub mesh: _init_tp_config reads the device only for its shape, device count and worker
grid width, and every config it builds is a host-side object.
"""
import pytest

import ttnn
from models.demos.qwen38.tt.model_config import Qwen38ModelArgs

# Other decode projections, each with the readers-per-bank its even per-bank width allows.
SHARDED_SHAPES = [
    ("mlp_w1", 2),
    ("mlp_w3", 2),
    ("mlp_w2", 2),
    ("attn_qkv_fused", 2),
    ("gdn_out", 2),
]


class _StubGrid:
    """Worker grid of a P150 (11 columns); only .x is read."""

    x = 11
    y = 10


class _StubMesh:
    """Single-device mesh stand-in for _init_tp_config."""

    shape = (1, 1)

    def get_num_devices(self):
        return 1

    def compute_with_storage_grid_size(self):
        return _StubGrid()


@pytest.fixture(scope="module")
def args():
    """One Qwen3.8-27B args object; each case re-runs _init_tp_config under its own environment."""
    return Qwen38ModelArgs(mesh_device=None)


def configure(args, monkeypatch, decode_matmul=None, qkvzab=None):
    """Rebuild the TP config with the two layout variables set (None = unset)."""
    for name, value in (("QWEN38_DECODE_MATMUL", decode_matmul), ("QWEN38_QKVZAB_LAYOUT", qkvzab)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    args._init_tp_config(_StubMesh())
    return args


def test_qkvzab_opts_out_of_the_sharded_default(args, monkeypatch):
    """Unset: the arm is dram_sharded and qkvzab alone takes the 1D interleaved path."""
    configure(args, monkeypatch)
    assert args.decode_matmul_layout == "dram_sharded"
    assert args.gdn_qkvzab_layout == "1d"
    assert args.gdn_qkvzab_1d_decode is True


def test_override_restores_the_sharded_arm_for_qkvzab(args, monkeypatch):
    """dram_sharded: the shape returns to the WIDTH_SHARDED weight and its one-reader-per-bank kernel."""
    configure(args, monkeypatch, qkvzab="dram_sharded")
    assert args.gdn_qkvzab_layout == "dram_sharded"
    assert args.gdn_qkvzab_1d_decode is False
    assert args.gdn_qkvzab_weight_memcfg.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    assert args.gdn_qkvzab_progcfg.num_workers_per_dram_bank == 1


@pytest.mark.parametrize("qkvzab", [None, "1d", "dram_sharded"])
def test_no_other_shape_moves_with_the_override(args, monkeypatch, qkvzab):
    """The override is per shape: every other decode projection keeps its sharded weight and readers."""
    configure(args, monkeypatch, qkvzab=qkvzab)
    assert args.proj_1d_decode is False
    assert args.mlp_1d_decode is False
    for name, workers in SHARDED_SHAPES:
        assert getattr(args, f"{name}_weight_memcfg").memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
        assert getattr(args, f"{name}_progcfg").num_workers_per_dram_bank == workers


def test_the_1d_arm_keeps_qkvzab_interleaved(args, monkeypatch):
    """QWEN38_DECODE_MATMUL=1d builds no sharded weights, so the override cannot shard this shape."""
    configure(args, monkeypatch, decode_matmul="1d", qkvzab="dram_sharded")
    assert args.gdn_qkvzab_layout == "1d"
    assert args.gdn_qkvzab_1d_decode is True
    assert args.proj_1d_decode is True


def test_an_unknown_override_value_is_rejected(args, monkeypatch, expect_error):
    """A typo selects no arm silently."""
    with expect_error(ValueError, "QWEN38_QKVZAB_LAYOUT"):
        configure(args, monkeypatch, qkvzab="sharded")
