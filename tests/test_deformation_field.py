"""Checks the displacement grid against an analytic deformation it should reproduce."""

import numpy as np
import pytest
import torch

from ultrasound_project.deformation.field import (
    DeformationField,
    TetrahedralInverseMapper,
    build_field,
    round_trip_error,
)

# a slab of "tissue" sampled like an FEM mesh: 15 mm nodes over 0.3 x 0.2 x 0.1 m
NODE_SPACING = 0.015
INDENT = 0.012  # m, peak displacement of the analytic press
SIGMA = 0.03  # m, lateral width of the press
DECAY = 0.04  # m, how fast it dies out with depth


def analytic_displacement(pos: torch.Tensor) -> torch.Tensor:
    """A probe press: Gaussian in x/y, exponential decay downwards from the top face."""
    radial = torch.linalg.norm(pos[:, :2], dim=-1)
    depth = pos[:, 2].max() - pos[:, 2]
    drop = INDENT * torch.exp(-0.5 * (radial / SIGMA) ** 2) * torch.exp(-depth / DECAY)
    return torch.stack([torch.zeros_like(drop), torch.zeros_like(drop), -drop], dim=-1)


@pytest.fixture
def nodes() -> tuple[torch.Tensor, torch.Tensor]:
    # lattice chosen so the probe axis and the top face land on nodes, not between them
    axes = [
        torch.arange(-0.150, 0.1501, NODE_SPACING),
        torch.arange(-0.105, 0.1051, NODE_SPACING),
        torch.arange(-0.105, 0.0001, NODE_SPACING),
    ]
    rest = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    return rest, rest + analytic_displacement(rest)


def test_field_reproduces_node_displacements(nodes):
    rest, cur = nodes
    field = build_field(rest, cur, spacing=0.005)
    mean_err, max_err = round_trip_error(field, rest, cur)
    # the grid must not lose what the nodes knew: sub-millimetre at the nodes themselves
    assert mean_err < 2e-4
    assert max_err < 1e-3


def test_field_is_zero_away_from_the_body(nodes):
    rest, cur = nodes
    field = build_field(rest, cur, spacing=0.005, margin=0.02)
    far = torch.tensor([[0.0, 0.0, 0.5], [1.0, 1.0, 1.0], [-0.4, 0.0, 0.0]])
    assert torch.allclose(field.sample(far), torch.zeros(3, 3), atol=1e-9)


def test_peak_displacement_matches_the_press(nodes):
    rest, cur = nodes
    field = build_field(rest, cur, spacing=0.005)
    # deepest point of the dent is under the probe axis, on the top face
    peak = field.sample(torch.tensor([[0.0, 0.0, 0.0]]))
    assert peak[0, 2] == pytest.approx(-INDENT, abs=5e-4)


def test_material_coords_inverts_the_deformation(nodes):
    rest, cur = nodes
    field = build_field(rest, cur, spacing=0.005)
    # points well inside the body, so the interpolation is not fighting the boundary
    inside = rest[(torch.linalg.norm(rest[:, :2], dim=-1) < 0.08) & (rest[:, 2] > -0.06)]
    deformed = inside + analytic_displacement(inside)
    recovered = field.material_coords(deformed)
    assert torch.linalg.norm(recovered - inside, dim=-1).max() < 5e-4


def test_deformed_keyed_field_inverts_in_one_lookup(nodes):
    rest, cur = nodes
    field = build_field(rest, cur, spacing=0.005, keyed_by="deformed")
    inside = rest[(torch.linalg.norm(rest[:, :2], dim=-1) < 0.08) & (rest[:, 2] > -0.06)]
    deformed = inside + analytic_displacement(inside)
    recovered = field.material_coords(deformed, iters=1)
    assert torch.linalg.norm(recovered - inside, dim=-1).max() < 5e-4


def test_save_and_load_round_trip(nodes, tmp_path):
    rest, cur = nodes
    field = build_field(rest, cur, spacing=0.005, meta={"force_n": 20.0})
    path = tmp_path / "field.npz"
    field.save(str(path))
    loaded = DeformationField.load(str(path))

    assert loaded.keyed_by == field.keyed_by
    assert loaded.spacing == pytest.approx(field.spacing)
    assert loaded.meta["force_n"] == 20.0
    assert np.allclose(loaded.u.numpy(), field.u.numpy(), atol=1e-6)
    assert np.allclose(loaded.origin.numpy(), field.origin.numpy(), atol=1e-6)


def test_tetrahedral_inverse_mapping_recovers_material_coordinates():
    rest = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    affine = torch.tensor(
        [
            [1.1, 0.1, 0.0],
            [0.0, 0.9, 0.05],
            [0.0, 0.0, 0.8],
        ]
    )
    translation = torch.tensor([0.2, -0.1, 0.05])
    current = rest @ affine.T + translation
    elements = torch.tensor([[0, 1, 2, 3]])
    mapper = TetrahedralInverseMapper(elements, node_count=4)

    barycentric = torch.tensor(
        [
            [0.4, 0.2, 0.3, 0.1],
            [0.1, 0.3, 0.2, 0.4],
        ]
    )
    expected_material = barycentric @ rest
    deformed_points = barycentric @ current
    recovered, inside = mapper.material_coords(deformed_points, rest, current)

    assert inside.all()
    assert torch.allclose(recovered, expected_material, atol=1e-6)


def test_tetrahedral_inverse_mapping_leaves_air_unchanged():
    rest = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    current = rest + torch.tensor([0.0, 0.0, -0.1])
    mapper = TetrahedralInverseMapper(torch.tensor([[0, 1, 2, 3]]), node_count=4)
    air = torch.tensor([[2.0, 2.0, 2.0]])

    recovered, inside = mapper.material_coords(air, rest, current)

    assert not inside.any()
    assert torch.equal(recovered, air)
