# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistent orthogonal rotation for queries and quantized attention keys."""

import torch

import ttnn


class HadamardRotation:
    """Share one normalized matrix across attention layers and trace replays.

    Apply after RoPE and before key quantization. Queries and keys must use the
    same rotation; values remain in their original basis.
    """

    def __init__(self, mesh_device, head_dim):
        if head_dim < 1 or head_dim & (head_dim - 1):
            raise ValueError("Hadamard head dimension must be a positive power of two")
        matrix = torch.ones(1, 1, dtype=torch.float32)
        while matrix.shape[0] < head_dim:
            matrix = torch.cat((torch.cat((matrix, matrix), dim=1), torch.cat((matrix, -matrix), dim=1)), dim=0)
        matrix *= head_dim**-0.5
        self.matrix = ttnn.from_torch(
            matrix,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def __call__(self, tensor):
        """Consume an unrotated tensor and return it in the shared basis."""
        rotated = ttnn.matmul(
            tensor,
            self.matrix,
            dtype=tensor.dtype,
            memory_config=tensor.memory_config(),
            compute_kernel_config=self.compute_config,
        )
        ttnn.deallocate(tensor)
        return rotated
