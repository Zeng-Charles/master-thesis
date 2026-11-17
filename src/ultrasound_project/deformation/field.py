"""Turning an FEM soft body's node motion into a dense displacement field.

A PhysX deformable gives displacement at its simulation nodes: a few thousand scattered
points, ~17 mm apart at the resolutions we run. What the ultrasound side needs is a
displacement it can evaluate at arbitrary points -- the ~30k sample points of one image
plane, or every CT voxel along a beam. This module bridges the two: scatter the node
displacements onto a regular grid once per step, then trilinearly interpolate it.

The grid is stored in the same metric frame as the nodes, so mapping it onto CT voxel
indices later is a division by the voxel spacing and nothing else.

Two framings, both useful and easy to mix up:

- ``keyed_by="rest"`` indexes the field by material (undeformed) position. This is the CT's
  own frame, and it is the field you would use to warp a CT volume forward.
- ``keyed_by="deformed"`` indexes it by current position, which is what you want when you
  hold a point in the deformed body and ask which material it is made of.

:meth:`DeformationField.material_coords` answers that second question either way -- in one
lookup for a deformed-keyed field, by fixed-point iteration for a rest-keyed one.

Nothing here imports Isaac Sim, so it can be tested on synthetic fields.

For a live 2D plane with access to the FEM tetrahedral topology,
:class:`TetrahedralInverseMapper` is preferable to a scattered displacement field: it
uses the containing tetrahedron's barycentric coordinates, stays continuous across
conforming element faces, and never needs an arbitrary interpolation radius.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# grid points this far apart get their neighbours searched in one chunk; purely a
# memory/speed knob, the result does not depend on it
_CHUNK = 4096


class TetrahedralInverseMapper:
    """Map current points into a tetrahedral mesh's reference configuration.

    The topology is fixed, so the node-to-tetrahedron adjacency is built once.  For
    every query point we search tetrahedra incident to its nearest simulation nodes,
    find the current tetrahedron containing the point, and reuse its barycentric
    coordinates in the reference tetrahedron.  Points outside the current volume are
    returned unchanged, which keeps deformation from leaking into air.
    """

    def __init__(
        self,
        elements: torch.Tensor,
        node_count: int,
        *,
        nearest_nodes: int = 8,
        inside_tolerance: float = 1.0e-4,
    ):
        elements = elements.to(dtype=torch.long)
        valid = ((elements >= 0) & (elements < node_count)).all(dim=-1)
        elements = elements[valid]
        if elements.ndim != 2 or elements.shape[1] != 4 or elements.shape[0] == 0:
            raise ValueError("elements must contain at least one valid tetrahedron shaped (T, 4)")

        # node -> incident tetrahedra, padded with -1: sort the (node, tetrahedron) pairs by
        # node and give each pair its rank among the node's pairs as the column
        nodes = elements.reshape(-1)
        owners = torch.arange(elements.shape[0], device=elements.device).repeat_interleave(4)
        order = torch.argsort(nodes, stable=True)
        counts = torch.bincount(nodes, minlength=node_count)
        first = torch.cumsum(counts, dim=0) - counts
        rank = torch.arange(nodes.numel(), device=elements.device) - first[nodes[order]]
        adjacency = torch.full(
            (node_count, int(counts.max())), -1, dtype=torch.long, device=elements.device
        )
        adjacency[nodes[order], rank] = owners[order]

        self.elements = elements
        self.node_tetrahedra = adjacency
        self.nearest_nodes = min(max(nearest_nodes, 1), node_count)
        self.inside_tolerance = inside_tolerance

    @property
    def tetrahedron_count(self) -> int:
        return self.elements.shape[0]

    def _candidates(
        self, chunk: torch.Tensor, pos: torch.Tensor, usable: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tetrahedra around each point, its barycentric coordinates in them, and which are usable."""
        nearest = torch.cdist(chunk, pos).topk(self.nearest_nodes, largest=False).indices
        candidates = self.node_tetrahedra[nearest].reshape(chunk.shape[0], -1)
        valid = candidates >= 0
        safe = candidates.clamp_min(0)
        if usable is not None:
            valid &= usable[safe]
        corners = pos[self.elements[safe]]

        # x = x0 + [x1-x0, x2-x0, x3-x0] * [l1,l2,l3]
        edges = (corners[..., 1:, :] - corners[..., :1, :]).transpose(-1, -2)
        rhs = chunk[:, None, :] - corners[..., 0, :]
        tail, info = torch.linalg.solve_ex(edges, rhs.unsqueeze(-1), check_errors=False)
        tail = tail.squeeze(-1)
        barycentric = torch.cat((1.0 - tail.sum(dim=-1, keepdim=True), tail), dim=-1)
        return safe, barycentric, valid & (info == 0)

    def locate(
        self, points: torch.Tensor, pos: torch.Tensor, usable: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Containing tetrahedron of each point in the mesh posed at ``pos``.

        Returns the tetrahedron index (-1 where none contains the point), the barycentric
        coordinates in it, and the mask of points that were found. Tetrahedra where the
        optional per-tetrahedron mask ``usable`` is False are never returned.
        """
        if pos.shape != (self.node_tetrahedra.shape[0], 3):
            raise ValueError("node arrays do not match the mapper topology")
        tetrahedron = torch.full((points.shape[0],), -1, dtype=torch.long, device=points.device)
        weights = torch.zeros((points.shape[0], 4), dtype=points.dtype, device=points.device)
        for start in range(0, points.shape[0], _CHUNK):
            chunk = points[start : start + _CHUNK]
            safe_candidates, barycentric, valid_candidate = self._candidates(chunk, pos, usable)
            tolerance = self.inside_tolerance
            inside = (
                valid_candidate
                & (barycentric.amin(dim=-1) >= -tolerance)
                & (barycentric.amax(dim=-1) <= 1.0 + tolerance)
            )

            # Shared faces can match more than one tetrahedron. Choosing the candidate
            # with the largest minimum barycentric coordinate selects the most interior
            # match; both choices give the same position on a conforming shared face.
            score = torch.where(
                inside,
                barycentric.amin(dim=-1),
                torch.full_like(barycentric[..., 0], -torch.inf),
            )
            best_column = score.argmax(dim=-1)
            has_match = inside.any(dim=-1)
            row = torch.arange(chunk.shape[0], device=points.device)
            stop = start + chunk.shape[0]
            tetrahedron[start:stop] = torch.where(has_match, safe_candidates[row, best_column], -1)
            weights[start:stop] = barycentric[row, best_column]
        return tetrahedron, weights, tetrahedron >= 0

    def count_containing(
        self,
        points: torch.Tensor,
        pos: torch.Tensor,
        usable: torch.Tensor | None = None,
        interior: float = 1.0e-3,
    ) -> torch.Tensor:
        """How many tetrahedra hold each point with all barycentric coordinates above ``interior``.

        In a mesh that partitions space this is at most 1 -- points on shared faces are
        excluded by ``interior``. More than one means the mesh overlaps itself there and the
        point has more than one material origin.
        """
        counts = torch.zeros(points.shape[0], dtype=torch.long, device=points.device)
        for start in range(0, points.shape[0], _CHUNK):
            chunk = points[start : start + _CHUNK]
            tetrahedra, barycentric, valid = self._candidates(chunk, pos, usable)
            # a tetrahedron is a candidate once per nearest node it touches; count it once
            keys, order = torch.sort(torch.where(valid, tetrahedra, -1), dim=1)
            inside = keys >= 0
            inside[:, 1:] &= keys[:, 1:] != keys[:, :-1]
            inside &= barycentric.amin(dim=-1).gather(1, order) > interior
            counts[start : start + chunk.shape[0]] = inside.sum(dim=1)
        return counts

    def material_coords(
        self,
        points: torch.Tensor,
        rest_pos: torch.Tensor,
        cur_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reference coordinates and a mask of points inside the current mesh.

        Tetrahedra the deformation has turned inside out are skipped: the map has no
        inverse there, so a point only they cover counts as outside the mesh.
        """
        if rest_pos.shape != cur_pos.shape:
            raise ValueError(f"rest_pos {tuple(rest_pos.shape)} and cur_pos {tuple(cur_pos.shape)} must match")
        upright = tetrahedron_volume_ratios(self.elements, rest_pos, cur_pos) > 0
        tetrahedron, weights, found = self.locate(points, cur_pos, upright)
        corners = rest_pos[self.elements[tetrahedron.clamp_min(0)]]
        reference_point = (weights[..., None] * corners).sum(dim=-2)
        return torch.where(found[:, None], reference_point, points), found

    def spatial_coords(
        self,
        points: torch.Tensor,
        rest_pos: torch.Tensor,
        cur_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The forward map: where material points (rest configuration) are now.

        Points outside the rest mesh are returned unchanged, with a False mask.
        """
        if rest_pos.shape != cur_pos.shape:
            raise ValueError(f"rest_pos {tuple(rest_pos.shape)} and cur_pos {tuple(cur_pos.shape)} must match")
        tetrahedron, weights, found = self.locate(points, rest_pos)
        corners = cur_pos[self.elements[tetrahedron.clamp_min(0)]]
        moved = (weights[..., None] * corners).sum(dim=-2)
        return torch.where(found[:, None], moved, points), found


@dataclass
class DeformationField:
    """A displacement field sampled on a regular grid.

    Attributes:
        u: Displacement in m, shape ``(nx, ny, nz, 3)``.
        origin: Metric position of voxel ``(0, 0, 0)``, shape ``(3,)``.
        spacing: Grid spacing in m.
        keyed_by: ``"rest"`` or ``"deformed"`` -- which configuration indexes the grid.
        meta: Free-form provenance (force, indentation, ...), carried through save/load.
    """

    u: torch.Tensor
    origin: torch.Tensor
    spacing: float
    keyed_by: str = "rest"
    meta: dict | None = None

    @property
    def dims(self) -> torch.Tensor:
        return torch.tensor(self.u.shape[:3], device=self.u.device)

    def sample(self, points: torch.Tensor) -> torch.Tensor:
        """Interpolates the field at ``points`` (N, 3), zero outside the grid.

        Zero outside is the physically right answer here: the grid is built around the
        body with a margin, so anything beyond it is air or tissue the body never reached.
        """
        # continuous voxel index, then to grid_sample's [-1, 1]
        idx = (points - self.origin.to(points)) / self.spacing
        norm = 2.0 * idx / (self.dims.to(points) - 1.0) - 1.0
        # grid_sample reads the last axis as (W, H, D), i.e. our (z, y, x) -- hence flip
        grid = norm.flip(-1).view(1, 1, 1, -1, 3)
        volume = self.u.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, nx, ny, nz)
        out = F.grid_sample(volume, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return out.view(3, -1).T

    def material_coords(self, points: torch.Tensor, iters: int = 3) -> torch.Tensor:
        """Maps deformed positions back to the material positions they came from.

        Solves ``X = x - u(X)``. With a deformed-keyed field that is one lookup; with a
        rest-keyed one it is a fixed-point iteration, which contracts as long as the
        deformation gradient stays below 1 -- true for any non-degenerate FEM state.
        """
        if self.keyed_by == "deformed":
            return points - self.sample(points)
        material = points.clone()
        for _ in range(max(iters, 1)):
            material = points - self.sample(material)
        return material

    def save(self, path: str):
        np.savez_compressed(
            path,
            u=self.u.detach().cpu().numpy().astype(np.float32),
            origin=self.origin.detach().cpu().numpy().astype(np.float32),
            spacing=np.float32(self.spacing),
            keyed_by=self.keyed_by,
            meta=json.dumps(self.meta or {}),
        )

    @classmethod
    def load(cls, path: str, device: str | torch.device = "cpu") -> "DeformationField":
        raw = np.load(path, allow_pickle=False)
        return cls(
            u=torch.as_tensor(raw["u"], device=device),
            origin=torch.as_tensor(raw["origin"], device=device),
            spacing=float(raw["spacing"]),
            keyed_by=str(raw["keyed_by"]),
            meta=json.loads(str(raw["meta"])),
        )


def node_spacing(positions: torch.Tensor, sample_size: int = 512) -> float:
    """Mean nearest-neighbour distance of a node cloud, from a random subsample.

    Used to size the splatting kernel: the grid can be as fine as we like, but the
    information in it is only as dense as the FEM mesh.
    """
    count = positions.shape[0]
    if count < 2:
        return 0.0
    if count > sample_size:
        pick = torch.randperm(count, device=positions.device)[:sample_size]
        subset = positions[pick]
    else:
        subset = positions
    dist = torch.cdist(subset, positions)
    # column 0 of each row is the point itself
    return dist.topk(2, largest=False).values[:, 1].mean().item()


def build_field(
    rest_pos: torch.Tensor,
    cur_pos: torch.Tensor,
    *,
    spacing: float = 0.005,
    margin: float = 0.02,
    radius: float | None = None,
    neighbours: int = 8,
    keyed_by: str = "rest",
    meta: dict | None = None,
) -> DeformationField:
    """Scatters per-node displacement onto a regular grid.

    Each grid point takes a weighted average of its ``neighbours`` nearest nodes, ignoring
    any beyond ``radius``, and stays at zero if none are in range. That last rule is what
    keeps the field from bleeding out into the air around the body.

    The weights are modified Shepard, ``((R - d) / (R d))^2``, not a Gaussian: they go to
    infinity as a grid point approaches a node, so the field reproduces the node
    displacements exactly where the nodes are instead of smoothing the dent's peak away.
    In between it is only as good as the node spacing allows.

    Args:
        rest_pos: Node positions in the undeformed body, ``(N, 3)`` in m.
        cur_pos: The same nodes now, ``(N, 3)`` in m.
        spacing: Grid spacing in m. Finer than the FEM mesh buys smoothness, not detail.
        margin: How far past the node bounding box the grid extends, in m.
        radius: Splatting cutoff in m. Defaults to twice the mean node spacing.
        neighbours: Nodes averaged per grid point.
        keyed_by: ``"rest"`` to index the grid by material position, ``"deformed"`` by
            current position.
        meta: Provenance to carry along.
    """
    if rest_pos.shape != cur_pos.shape:
        raise ValueError(f"rest_pos {tuple(rest_pos.shape)} and cur_pos {tuple(cur_pos.shape)} must match")
    if keyed_by not in ("rest", "deformed"):
        raise ValueError(f"keyed_by must be 'rest' or 'deformed', got {keyed_by!r}")

    device = rest_pos.device
    disp = cur_pos - rest_pos
    key_pos = rest_pos if keyed_by == "rest" else cur_pos

    if radius is None:
        radius = 2.0 * node_spacing(key_pos)

    lower = key_pos.amin(dim=0) - margin
    upper = key_pos.amax(dim=0) + margin
    dims = torch.ceil((upper - lower) / spacing).long() + 1
    axes = [lower[i] + spacing * torch.arange(int(dims[i]), device=device) for i in range(3)]
    grid_pos = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)

    u_flat = torch.zeros_like(grid_pos)
    k = min(neighbours, key_pos.shape[0])
    for start in range(0, grid_pos.shape[0], _CHUNK):
        chunk = grid_pos[start : start + _CHUNK]
        dist, idx = torch.cdist(chunk, key_pos).topk(k, largest=False)
        # clamped so a grid point sitting exactly on a node gets a huge but finite weight
        weight = ((radius - dist).clamp_min(0.0) / (radius * dist.clamp_min(1e-6))) ** 2
        total = weight.sum(dim=1, keepdim=True)
        weighted = (weight.unsqueeze(-1) * disp[idx]).sum(dim=1)
        u_flat[start : start + chunk.shape[0]] = torch.where(total > 0, weighted / total.clamp_min(1e-12), 0.0)

    return DeformationField(
        u=u_flat.reshape(int(dims[0]), int(dims[1]), int(dims[2]), 3),
        origin=lower,
        spacing=spacing,
        keyed_by=keyed_by,
        meta=meta,
    )


def tetrahedron_volume_ratios(elements: torch.Tensor, rest_pos: torch.Tensor, cur_pos: torch.Tensor) -> torch.Tensor:
    """det F of every tetrahedron: its current over its rest volume, negative if inverted."""
    def signed_volume(pos: torch.Tensor) -> torch.Tensor:
        corners = pos[elements.long()]
        return torch.linalg.det(corners[:, 1:] - corners[:, :1])

    return signed_volume(cur_pos) / signed_volume(rest_pos)


# outward faces of a positively oriented tetrahedron (v0, v1, v2, v3)
_TET_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


def boundary_faces(elements: np.ndarray, rest_pos: np.ndarray) -> np.ndarray:
    """Surface triangles of a tetrahedral mesh, wound to face outward."""
    elements = np.asarray(elements, dtype=np.int64)
    corners = rest_pos[elements]
    volume = np.linalg.det(corners[:, 1:] - corners[:, :1])
    elements = elements[volume != 0]
    flipped = volume[volume != 0] < 0
    elements[flipped] = elements[flipped][:, [0, 2, 1, 3]]
    faces = np.concatenate([elements[:, list(face)] for face in _TET_FACES])
    _, index, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    return faces[np.sort(index[counts == 1])]


_TET_EDGES = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])


def slice_tetrahedra(
    elements: np.ndarray, pos: np.ndarray, origin: np.ndarray, u_axis: np.ndarray, v_axis: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray]:
    """Cross-section of a tetrahedral mesh with the plane ``origin + s u + t v``.

    Every tetrahedron the plane cuts leaves a triangle or a quadrilateral. Returns those
    polygons in plane coordinates (s, t), corners in order around the polygon, and the
    index of the tetrahedron each came from.
    """
    normal = np.cross(u_axis, v_axis)
    normal = normal / np.linalg.norm(normal)
    corners = pos[elements]  # (T, 4, 3)
    side = (corners - origin) @ normal  # (T, 4)
    cut = (side.min(axis=1) < 0) & (side.max(axis=1) > 0)
    index = np.nonzero(cut)[0]
    corners, side = corners[index], side[index]
    a, b = _TET_EDGES[:, 0], _TET_EDGES[:, 1]
    crosses = (side[:, a] < 0) != (side[:, b] < 0)  # (T', 6)
    # edges the plane does not cross divide by zero here; they are masked out below
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = side[:, a] / (side[:, a] - side[:, b])
        points = corners[:, a] + weight[..., None] * (corners[:, b] - corners[:, a])  # (T', 6, 3)
        planar = np.stack([(points - origin) @ u_axis, (points - origin) @ v_axis], axis=-1)
    polygons = []
    for row, mask in zip(planar, crosses):
        corner = row[mask]
        centre = corner.mean(axis=0)
        order = np.argsort(np.arctan2(corner[:, 1] - centre[1], corner[:, 0] - centre[0]))
        polygons.append(corner[order])
    return polygons, index


def round_trip_error(field: DeformationField, rest_pos: torch.Tensor, cur_pos: torch.Tensor) -> tuple[float, float]:
    """Mean and max error of the field against the node displacements it was built from.

    The only honest check of whether the grid kept what the FEM mesh knew. Errors well
    below the voxel size mean the field is a faithful stand-in for the node data.
    """
    truth = cur_pos - rest_pos
    query = rest_pos if field.keyed_by == "rest" else cur_pos
    err = torch.linalg.norm(field.sample(query) - truth, dim=-1)
    return err.mean().item(), err.max().item()
