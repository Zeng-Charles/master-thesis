"""Checks the CT side of the soft-body pipeline on meshes small enough to reason about.

The pull-back and CT warp (:mod:`ultrasound_project.deformation.ct_warp`), the tetrahedral mesh
helpers they rest on, speckle tracking (:mod:`ultrasound_project.utils.image_metrics`) and
the SonoGym geometry the images are sliced with. Coordinates are voxels unless a test says
otherwise.
"""

import itertools

import numpy as np
import pytest
import torch
from scipy import ndimage

from ultrasound_project.deformation.ct_warp import PullbackField, VoxelBox, pullback, submesh
from ultrasound_project.deformation.field import (
    TetrahedralInverseMapper,
    boundary_faces,
    slice_tetrahedra,
    tetrahedron_volume_ratios,
)
from ultrasound_project.sensors.sonogym_us import UltrasoundPlane, sample_volume
from ultrasound_project.utils.image_metrics import BlockMatch, accumulate, block_match, compare, parabolic_offset
from ultrasound_project.utils.sonogym_patient import (
    BONE_LABEL,
    LABEL_RES,
    SENSOR_CFG_DIR,
    ProbePose,
    SceneFrame,
    sonogym_label_lut,
)

CUBE = 20.0  # side of the test cube, in voxels


def box_mesh(cells: int, size: float) -> tuple[np.ndarray, np.ndarray]:
    """The cube [0, size]^3 cut into cells^3 cubes of six tetrahedra each.

    Every cube is split the same way (Kuhn: the six monotone paths from its low to its high
    corner), so neighbouring cubes agree on their shared faces and the mesh is conforming.
    """
    axis = np.linspace(0.0, size, cells + 1)
    nodes = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    stride = np.array([(cells + 1) ** 2, cells + 1, 1])
    tetrahedra = []
    for cube in itertools.product(range(cells), repeat=3):
        for order in itertools.permutations(range(3)):
            path = [np.array(cube)]
            for axis_id in order:
                path.append(path[-1] + np.eye(3, dtype=int)[axis_id])
            tetrahedra.append(np.array(path) @ stride)
    return nodes, np.array(tetrahedra)


@pytest.fixture
def cube() -> tuple[torch.Tensor, np.ndarray, TetrahedralInverseMapper]:
    nodes, tetrahedra = box_mesh(cells=4, size=CUBE)
    mapper = TetrahedralInverseMapper(torch.as_tensor(tetrahedra), node_count=len(nodes), nearest_nodes=12)
    return torch.as_tensor(nodes), tetrahedra, mapper


def two_tetrahedra() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two tetrahedra sharing the face z = 0, and a pose that pushes the lower one through it."""
    rest = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], dtype=torch.float64)
    elements = torch.tensor([[0, 1, 2, 3], [0, 2, 1, 4]])
    current = rest.clone()
    current[4] = torch.tensor([0.25, 0.25, 0.6], dtype=torch.float64)
    return rest, elements, current


# ------------------------------------------------------------------------------ pull-back


def test_pullback_undoes_a_translation(cube):
    rest, _, mapper = cube
    shift = torch.tensor([0.4, -0.7, 1.5], dtype=torch.float64)
    axis = torch.arange(4.0, 16.0, dtype=torch.float64)
    points = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)

    material, inside = pullback(points, (1.0, 1.0, 1.0), mapper, rest, rest + shift)

    assert inside.all()
    assert torch.allclose(material, points - shift, atol=1e-9)


def test_pullback_maps_a_pressed_in_surface_onto_air(cube):
    """Voxels the skin has been pushed away from must read what was above the skin."""
    rest, _, mapper = cube
    drop = 2.0
    s = torch.arange(2.0, 18.0, 0.5, dtype=torch.float64)
    z = torch.arange(10.0, 30.0, 0.5, dtype=torch.float64)
    plane = torch.stack(torch.meshgrid(s, z, indexing="ij"), dim=-1)
    points = torch.stack([plane[..., 0], torch.full_like(plane[..., 0], CUBE / 2), plane[..., 1]], dim=-1)
    current = rest - torch.tensor([0.0, 0.0, drop], dtype=torch.float64)

    material, inside = pullback(points, (0.5, 0.5), mapper, rest, current, decay_mm=15.0)

    top = CUBE - drop
    assert torch.equal(inside, points[..., 2] <= top)
    vacated = ~inside & (points[..., 2] < CUBE)
    assert vacated.any()
    # without the extension these would keep X = x: the tissue that used to be there
    assert (material[..., 2][vacated] > CUBE).all()
    # outside, the nearest inside displacement fades with distance to the deformed surface
    fade = torch.exp(-(((points[..., 2] - top) / 15.0) ** 2))
    expected = torch.where(inside, points[..., 2] + drop, points[..., 2] + drop * fade)
    assert torch.allclose(material[..., 2], expected, atol=1e-9)
    assert torch.allclose(material[..., :2], points[..., :2], atol=1e-9)


def test_warp_squeezes_a_ct_like_the_mesh(cube):
    rest, _, mapper = cube
    rest = rest.float()
    current = rest * torch.tensor([1.0, 1.0, 0.9])
    volume = torch.arange(24, dtype=torch.float32).expand(24, 24, 24).clone()  # value = k
    box = VoxelBox(start=np.array([4, 4, 2]), shape=np.array([12, 12, 12]))

    field = PullbackField.build(box, mapper, rest, current, voxel_mm=1.0)
    warped = field.warp(volume)

    k = torch.arange(2, 14, dtype=torch.float32)
    assert field.inside.all()
    assert torch.allclose(warped[box.slices], (k / 0.9).expand(12, 12, 12), atol=1e-4)
    assert torch.equal(warped[:4], volume[:4])
    assert torch.allclose(field.jacobian_determinant(), torch.full((12, 12, 12), 1 / 0.9), atol=1e-4)
    probe = torch.tensor([[6.5, 7.25, 9.0]])
    assert torch.allclose(field.material(probe), torch.tensor([[6.5, 7.25, 10.0]]), atol=1e-4)


# ------------------------------------------------------------------------------ mesh helpers


def test_forward_and_inverse_maps_agree(cube):
    rest, _, mapper = cube
    x, y, z = rest.unbind(-1)
    bump = torch.exp(-((x - 10) ** 2 + (y - 10) ** 2) / 50)
    current = rest + torch.stack([0.3 * torch.sin(z / 6), 0.2 * torch.cos(x / 7), -0.04 * z * bump], dim=-1)
    material = 1.0 + 18.0 * torch.rand((500, 3), generator=torch.Generator().manual_seed(0), dtype=torch.float64)

    moved, found = mapper.spatial_coords(material, rest, current)
    recovered, inside = mapper.material_coords(moved, rest, current)

    assert found.all() and inside.all()
    assert torch.allclose(recovered, material, atol=1e-9)


def test_material_coords_skips_inverted_tetrahedra():
    rest, elements, current = two_tetrahedra()
    mapper = TetrahedralInverseMapper(elements, node_count=5)
    ratios = tetrahedron_volume_ratios(elements, rest, current)
    assert ratios[0] == pytest.approx(1.0) and ratios[1] < 0
    # centroid of the inverted element, which lies inside the upright one as well
    query = current[elements[1]].mean(dim=0, keepdim=True)
    tetrahedron, _, _ = mapper.locate(query, current)
    assert tetrahedron.item() == 1  # left alone, the search prefers the inverted element here

    material, inside = mapper.material_coords(query, rest, current)

    assert inside.all()
    assert torch.allclose(material, query)  # the upright element has not moved


def test_count_containing_finds_where_the_mesh_overlaps_itself():
    rest, elements, current = two_tetrahedra()
    mapper = TetrahedralInverseMapper(elements, node_count=5)
    points = torch.tensor(
        [[0.3125, 0.3125, 0.15], [0.1, 0.1, 0.7], [0.2, 0.2, -0.3], [2.0, 2.0, 2.0]], dtype=torch.float64
    )
    upright = tetrahedron_volume_ratios(elements, rest, current) > 0

    assert mapper.count_containing(points, rest).tolist() == [1, 1, 1, 0]
    assert mapper.count_containing(points, current).tolist() == [2, 1, 0, 0]
    assert mapper.count_containing(points, current, upright).tolist() == [1, 1, 0, 0]


def test_volume_ratios_of_a_uniform_scaling(cube):
    rest, tetrahedra, _ = cube
    ratios = tetrahedron_volume_ratios(torch.as_tensor(tetrahedra), rest, rest * 1.1)
    assert torch.allclose(ratios, torch.full_like(ratios, 1.1**3))


def test_boundary_faces_wrap_the_cube_outward(cube):
    rest, tetrahedra, _ = cube
    nodes = rest.numpy()

    faces = boundary_faces(tetrahedra, nodes)

    assert len(faces) == 6 * 4 * 4 * 2  # six sides of 4 x 4 squares, two triangles each
    corners = nodes[faces]
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    assert (np.einsum("ij,ij->i", normal, corners.mean(axis=1) - CUBE / 2) > 0).all()
    assert 0.5 * np.linalg.norm(normal, axis=1).sum() == pytest.approx(6 * CUBE**2)


def test_slice_covers_the_cross_section(cube):
    rest, tetrahedra, _ = cube
    nodes = rest.numpy()
    height = 7.3

    polygons, index = slice_tetrahedra(
        tetrahedra, nodes, np.array([0.0, 0.0, height]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    )

    assert all(len(p) in (3, 4) for p in polygons)
    area = sum(0.5 * abs(p[:, 0] @ np.roll(p[:, 1], -1) - p[:, 1] @ np.roll(p[:, 0], -1)) for p in polygons)
    assert area == pytest.approx(CUBE**2)
    z = nodes[tetrahedra[index]][..., 2]
    assert ((z.min(axis=1) < height) & (z.max(axis=1) > height)).all()


def test_submesh_keeps_every_element_touching_the_box(cube):
    rest, tetrahedra, _ = cube
    nodes = rest.numpy()
    lower, upper = np.array([4.0, 4.0, 4.0]), np.array([11.0, 9.0, 16.0])

    kept_nodes, local, kept = submesh(tetrahedra, nodes, lower, upper)

    assert np.array_equal(kept_nodes[local], tetrahedra[kept])
    touching = np.all((nodes >= lower) & (nodes <= upper), axis=1)[tetrahedra].any(axis=1)
    assert np.array_equal(np.sort(kept), np.nonzero(touching)[0])


# ------------------------------------------------------------------------------ speckle tracking


def speckle(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    return ndimage.gaussian_filter(np.random.default_rng(seed).standard_normal(shape), 1.5)


def uniform_match(rows: np.ndarray, cols: np.ndarray, motion: tuple[float, float]) -> BlockMatch:
    displacement = np.broadcast_to(np.asarray(motion, dtype=np.float64), (len(rows), len(cols), 2)).copy()
    return BlockMatch(rows=rows, cols=cols, displacement=displacement, correlation=np.ones((len(rows), len(cols))))


def test_block_match_recovers_a_subpixel_shift():
    reference = speckle((160, 120))
    image = ndimage.shift(reference, (2.3, -1.4), order=3, mode="nearest")

    match = block_match(reference, image)

    assert (match.correlation > 0.95).all()
    error = match.displacement - np.array([2.3, -1.4])
    assert np.abs(np.median(error, axis=(0, 1))).max() < 0.05
    assert np.abs(error).max() < 0.2


def test_parabolic_offset_finds_the_vertex():
    scores = -((np.arange(7.0) - 3.3) ** 2)
    assert parabolic_offset(scores, 3) == pytest.approx(0.3)
    assert parabolic_offset(scores, 0) == 0.0  # no neighbour on one side
    assert parabolic_offset(np.ones(5), 2) == 0.0  # flat: nothing to fit


def test_accumulate_reads_each_step_where_the_block_has_moved():
    rows, cols = np.arange(0, 80, 8), np.arange(0, 64, 8)
    first = uniform_match(rows, cols, (4.0, 1.0))
    second = uniform_match(rows, cols, (0.0, 0.0))
    second.displacement[..., 0] = 0.05 * rows[:, None]  # stretches with depth

    total = accumulate([first, second])

    # block at row 16 sits at row 20 after the first step, where the second moves it 1.0
    assert total[2, 3] == pytest.approx([5.0, 1.0])
    # past the last block row there is nothing to read the second step from
    assert np.isnan(total[-1]).all()


def test_accumulate_drops_only_paths_through_unreliable_blocks():
    rows, cols = np.arange(0, 80, 8), np.arange(0, 64, 8)
    first = uniform_match(rows, cols, (2.0, 0.0))
    second = uniform_match(rows, cols, (0.0, 0.0))
    second.correlation[4, 4] = 0.3  # the block at row 32, column 32

    total = accumulate([first, second])

    # rows 26 and 34 are read between row 32 and a neighbour, so they lean on the dropped block
    assert np.isnan(total[3, 4]).all() and np.isnan(total[4, 4]).all()
    assert total[2, 4] == pytest.approx([2.0, 0.0])
    # column 40 is read exactly on its node, where the dropped column-32 block has zero weight
    assert total[4, 5] == pytest.approx([2.0, 0.0])


def test_compare_scores_an_image_against_itself():
    image = speckle((64, 48))
    scores = compare(image, image, data_range=float(np.ptp(image)))
    assert scores["ssim"] == pytest.approx(1.0)
    assert scores["ncc"] == pytest.approx(1.0)
    assert scores["mae"] == 0.0
    assert scores["psnr_db"] == float("inf")


# ------------------------------------------------------------------------------ SonoGym geometry


@pytest.mark.skipif(not (SENSOR_CFG_DIR / "label_conversion.yaml").exists(), reason="SonoGym submodule not checked out")
def test_label_lut_chains_sonogym_replacements():
    lut = sonogym_label_lut(120)
    # 2 -> 6 -> 12 -> 2: the yaml read as a plain mapping would leave raw 2 at 6
    expected = {
        2: list(range(1, 15)),
        6: [23, 24],
        8: [51, *range(80, 90)],
        10: [79],
        12: [*range(15, 23), 120],
        BONE_LABEL: [*range(25, 51), *range(69, 79), *range(92, 118)],
        15: list(range(52, 69)),
    }
    for target, sources in expected.items():
        assert (lut[sources] == target).all(), target
    assert lut[0] == 0


def test_ultrasound_plane_follows_the_probe():
    # lateral = y, elevation = z, normal = x: a right-handed probe frame
    pose = ProbePose(skin=np.array([0.1, 0.2, 0.3]), axes=np.eye(3)[:, [1, 2, 0]], surface_xz=(0, 0), x_angle=0.0)
    plane = UltrasoundPlane()

    points = plane.points(pose, advance=0.002)

    assert points.shape == (150, 200, 3)
    # the image starts 3 mm below the probe face, which has moved in by the advance
    assert np.allclose(points[75, 0], pose.skin + 0.005 * pose.normal)
    assert np.allclose(points[1:] - points[:-1], 0.0005 * pose.lateral)
    assert np.allclose(points[:, 1:] - points[:, :-1], 0.0005 * pose.normal)
    assert plane.extent_mm() == pytest.approx([-37.5, 37.5, 100.0, 0.0])


def test_sample_volume_is_trilinear_and_clamps_to_the_border():
    i, j, k = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) for n in (5, 6, 7)), indexing="ij")
    volume = i + 2 * j + 3 * k
    points = torch.tensor([[1.25, 2.5, 3.75], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0], [-3.0, 2.0, 9.0]], dtype=torch.float64)

    values = sample_volume(volume, points)

    assert torch.allclose(values, torch.tensor([17.5, 0.0, 32.0, 22.0], dtype=torch.float64))


def test_scene_frame_lays_the_patient_prone():
    bounds = np.array([[10, 20, 30], [110, 220, 330]])
    frame = SceneFrame.prone(bounds)
    points = np.random.default_rng(0).uniform(0.0, 0.5, size=(10, 3))

    assert np.allclose(frame.to_human(frame.to_scene(points)), points)
    assert np.allclose(frame.to_scene(torch.as_tensor(points)).numpy(), frame.to_scene(points))
    # SonoGym's j grows from the back into the body, i.e. downwards in the prone scene
    assert np.allclose(frame.vector_to_scene(np.array([0.0, 1.0, 0.0])), [0.0, 0.0, -1.0])
    front, back = frame.to_scene(np.array([[60, 220, 180], [60, 20, 180]]) * LABEL_RES)
    assert front == pytest.approx([0.0, 0.0, 0.0])
    assert back == pytest.approx([0.0, 0.0, 200 * LABEL_RES])
