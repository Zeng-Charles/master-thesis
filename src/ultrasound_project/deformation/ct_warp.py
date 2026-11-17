"""Warping CT and label volumes by an FEM deformation.

An FEM step leaves node positions before (X) and after (x). A CT of the deformed body
needs the opposite direction: for every voxel x, the material point X that now sits
there, so that the deformed image is CT(X(x)). That pull-back is evaluated exactly inside
the deformed tetrahedral mesh, by barycentric coordinates
(:class:`~ultrasound_project.deformation.field.TetrahedralInverseMapper`), and
extended outside it with a decaying copy of the nearest interior value.

The extension is what makes a probe's indentation read as air. Without it a voxel the
skin has been pushed away from keeps X = x, i.e. the tissue that used to be there; with
it the voxel maps back to just above the original skin.

Everything is in CT voxel units -- FEM nodes are moved into the voxel frame once -- so a
displacement of 1.0 is one voxel, 1.5 mm for SonoGym's patients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage

from ultrasound_project.deformation.field import TetrahedralInverseMapper
from ultrasound_project.sensors.sonogym_us import sample_volume


def submesh(elements: np.ndarray, rest: np.ndarray, lower: np.ndarray, upper: np.ndarray):
    """The tetrahedra with at least one rest node inside ``[lower, upper]``.

    Returns the kept node ids, the kept tetrahedra re-indexed into them, and the kept
    tetrahedra's ids in ``elements``. Pick the box with a margin of a couple of elements
    around the region you will query: a query point is only found inside the mesh if its
    whole tetrahedron was kept.
    """
    near = np.all((rest >= lower) & (rest <= upper), axis=1)
    kept = np.nonzero(near[elements].any(axis=1))[0]
    nodes, local = np.unique(elements[kept], return_inverse=True)
    return nodes, local.reshape(-1, 4), kept


def pullback(
    points: torch.Tensor,
    spacing_mm: tuple[float, ...],
    mapper: TetrahedralInverseMapper,
    rest: torch.Tensor,
    current: torch.Tensor,
    decay_mm: float = 15.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Material coordinates of a regular grid of deformed points.

    Args:
        points: Deformed positions in voxel coordinates, shaped (*grid, 3) for a 2D or 3D
            grid -- a CT box or an image plane.
        spacing_mm: Physical spacing of the grid axes, used for the distance to the mesh.
        mapper: Topology of the FEM mesh.
        rest / current: FEM node positions in voxel coordinates, (N, 3).
        decay_mm: Length over which the displacement fades outside the deformed mesh. It
            has to exceed the indentation depth for the indentation to map onto air.

    Returns:
        Material coordinates (*grid, 3), and the mask of points inside the deformed mesh.
    """
    grid = points.shape[:-1]
    material, inside = mapper.material_coords(points.reshape(-1, 3), rest, current)
    displacement = (points.reshape(-1, 3) - material).reshape(*grid, 3)
    inside = inside.reshape(grid)
    if bool(inside.any()) and not bool(inside.all()):
        distance, nearest = ndimage.distance_transform_edt(
            (~inside).cpu().numpy(), sampling=spacing_mm, return_indices=True
        )
        nearest = torch.as_tensor(nearest, device=points.device)
        fade = torch.exp(-((torch.as_tensor(distance, device=points.device) / decay_mm) ** 2))
        extended = displacement[tuple(nearest)] * fade[..., None].to(displacement.dtype)
        displacement = torch.where(inside[..., None], displacement, extended)
    return points - displacement, inside


@dataclass
class VoxelBox:
    """The integer box ``[start, start + shape)`` of a voxel grid."""

    start: np.ndarray
    shape: np.ndarray

    @classmethod
    def around(cls, points: np.ndarray, margin: float, volume_shape) -> "VoxelBox":
        lower = np.maximum(np.floor(points.min(axis=0) - margin), 0).astype(int)
        upper = np.minimum(np.ceil(points.max(axis=0) + margin) + 1, np.asarray(volume_shape)).astype(int)
        return cls(start=lower, shape=upper - lower)

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return tuple(slice(int(s), int(s + n)) for s, n in zip(self.start, self.shape))

    def coords(self, device) -> torch.Tensor:
        axes = [torch.arange(int(s), int(s + n), device=device, dtype=torch.float32) for s, n in zip(self.start, self.shape)]
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)


@dataclass
class PullbackField:
    """x -> X over a voxel box, stored as ``displacement = x - X`` in voxels.

    Outside the box the deformation is taken to be zero; the box is chosen so that the
    FEM displacement has died out at its faces.
    """

    box: VoxelBox
    displacement: torch.Tensor
    inside: torch.Tensor

    @classmethod
    def build(
        cls,
        box: VoxelBox,
        mapper: TetrahedralInverseMapper,
        rest: torch.Tensor,
        current: torch.Tensor,
        voxel_mm: float,
        decay_mm: float = 15.0,
    ) -> "PullbackField":
        coords = box.coords(rest.device)
        material, inside = pullback(coords, (voxel_mm,) * 3, mapper, rest, current, decay_mm)
        return cls(box=box, displacement=coords - material, inside=inside)

    def material(self, voxels: torch.Tensor) -> torch.Tensor:
        """Material coordinates of arbitrary deformed points (..., 3), by trilinear lookup."""
        local = voxels - torch.as_tensor(self.box.start, device=voxels.device, dtype=voxels.dtype)
        shape = torch.as_tensor(self.box.shape, device=voxels.device, dtype=voxels.dtype)
        inside_box = ((local >= 0) & (local <= shape - 1)).all(dim=-1, keepdim=True)
        flat = local.reshape(-1, 3)
        components = [sample_volume(self.displacement[..., c], flat) for c in range(3)]
        displacement = torch.stack(components, dim=-1).reshape(voxels.shape)
        return voxels - torch.where(inside_box, displacement, torch.zeros_like(displacement))

    def warp(self, volume: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
        """The deformed volume: ``volume(X(x))`` inside the box, unchanged outside."""
        out = volume.clone()
        material = self.box.coords(volume.device) - self.displacement
        out[self.box.slices] = sample_volume(volume, material, mode=mode).to(volume.dtype)
        return out

    def jacobian_determinant(self) -> torch.Tensor:
        """det(dX/dx) per voxel; its inverse is the local volume ratio of the deformation."""
        rows = []
        for component in range(3):
            gradients = torch.gradient(self.displacement[..., component], dim=(0, 1, 2))
            rows.append(torch.stack(gradients, dim=-1))
        jacobian = torch.eye(3, device=self.displacement.device) - torch.stack(rows, dim=-2)
        return torch.linalg.det(jacobian)
