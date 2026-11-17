"""SonoGym's patient assets, and the frames needed to deform one.

SonoGym keeps a patient in its CT's voxel grid: the CT, the label map and every STL share
one (i, j, k) index space, and the tasks' "human frame" is that grid scaled by
``label_res`` (1.5 mm). Probe poses come from two caches over (i, k): the skin voxel
``j = body_lowest_y_array[i, k]`` and the surface normal there, which points into the
body. For the bundled patients the low-j side is the back, which is where SonoGym scans
the spine.

A soft body also needs a world. :class:`SceneFrame` lays the patient prone on z = 0, back
up: (i, j, k) -> (x, y, z) = (i, k, -j) up to a translation. The FEM press and the CT
warp both take it from here, so the simulation and the CT side cannot drift apart.

Nothing here imports Isaac Sim.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from ruamel.yaml import YAML

from ultrasound_project import PATIENTS_DIR, PROJECT_DIR

SONOGYM_ROOT = Path(os.environ.get("SONOGYM_ROOT", os.path.join(PROJECT_DIR, "sonogym-custom")))
SONOGYM_PACKAGE_DIR = SONOGYM_ROOT / "source" / "spinal_surgery" / "spinal_surgery"
HUMAN_MODELS_DIR = SONOGYM_PACKAGE_DIR / "assets" / "data" / "HumanModels"
SENSOR_CFG_DIR = SONOGYM_PACKAGE_DIR / "lab" / "sensors" / "cfgs"

LABEL_RES = 0.0015
"""Metres per voxel; ``patient.label_res`` in every SonoGym task config."""

BONE_LABEL = 13
"""Bone after label conversion, ``surface_to_label_map['bone']`` in ``us_cfg.yaml``."""

# USSlicer's defaults, which the tasks never override: the end effector sits HEIGHT above
# the skin point and the image starts HEIGHT_IMG below the end effector, i.e. 3 mm deep
PROBE_HEIGHT = 0.13
IMAGE_HEIGHT = 0.133


def load_yaml(path: str | os.PathLike):
    with open(path) as stream:
        return YAML().load(stream)


def sonogym_label_lut(max_label: int = 255) -> np.ndarray:
    """The label conversion USSlicer applies, as a lookup table.

    ``USSlicer`` converts TotalSegmentator labels with one in-place replacement per entry
    of ``label_conversion.yaml``, in file order. The replacements chain -- a voxel turned
    from 2 into 6 is turned into 12 by the later ``6: 12`` entry, and back into 2 by
    ``12: 2`` -- so the table is built by running that same loop over every label value
    rather than by reading the yaml as a plain mapping.
    """
    lut = np.arange(max_label + 1, dtype=np.int64)
    for source, target in load_yaml(SENSOR_CFG_DIR / "label_conversion.yaml").items():
        lut[lut == int(source)] = int(target)
    return lut


@dataclass(frozen=True)
class SonoGymPatient:
    """File access for one SonoGym patient.

    Patients added under ``assets/patients/<id>/{raw,stl}`` take precedence over the ones
    that came with SonoGym's asset download, as in :mod:`ultrasound_project.utils.assets`
    -- which cannot be used here because it imports ``spinal_surgery`` and with it Isaac Lab.
    """

    patient_id: str = "s0010"

    @property
    def raw_dir(self) -> Path:
        local = Path(PATIENTS_DIR) / self.patient_id / "raw"
        return local if local.is_dir() else HUMAN_MODELS_DIR / "selected_dataset" / self.patient_id

    @property
    def stl_dir(self) -> Path:
        local = Path(PATIENTS_DIR) / self.patient_id / "stl"
        return local if local.is_dir() else HUMAN_MODELS_DIR / "selected_dataset_stl" / self.patient_id

    @property
    def ct_path(self) -> Path:
        return self.raw_dir / "ct.nii.gz"

    @property
    def label_path(self) -> Path:
        return self.stl_dir / "combined_label_map.nii.gz"

    @property
    def surface_mesh_path(self) -> Path:
        """Watertight outer surface in voxel coordinates; SonoGym's rigid patient uses it too."""
        return self.stl_dir / "combined_wrapwrap.stl"

    def load_ct(self) -> tuple[np.ndarray, np.ndarray]:
        """CT in HU as float32, and its affine."""
        image = nib.load(self.ct_path)
        return np.asarray(image.dataobj, dtype=np.float32), image.affine

    def load_labels(self, convert: bool = True) -> np.ndarray:
        """Label map, by default converted to the tissue classes SonoGym's US model uses."""
        labels = np.asarray(nib.load(self.label_path).dataobj).astype(np.int64)
        return sonogym_label_lut(int(labels.max()))[labels] if convert else labels

    @cached_property
    def surface_map(self) -> np.ndarray:
        """Skin voxel j for every (i, k); -1 where the column misses the body."""
        surface = torch.load(self.stl_dir / "body_lowest_y_array.pt", map_location="cpu", weights_only=True)
        return surface.numpy()

    @cached_property
    def normal_map(self) -> np.ndarray:
        """Unit inward surface normal for every (i, k), in the voxel frame."""
        normals = torch.load(self.stl_dir / "body_surface_normal_array.pt", map_location="cpu", weights_only=True)
        normals = normals.numpy().astype(np.float64)
        return normals / np.linalg.norm(normals, axis=-1, keepdims=True).clip(min=1e-12)

    def anatomy_centroid(self, name: str) -> np.ndarray:
        """Vertex centroid of one anatomy STL, in voxel coordinates.

        This is the point ``VertebraViewer`` takes as the guidance goal when
        ``use_vertebra_goal`` is set.
        """
        import trimesh

        return np.asarray(trimesh.load_mesh(self.stl_dir / f"{name}.stl").vertices).mean(axis=0)


@dataclass(frozen=True, eq=False)
class ProbePose:
    """A SonoGym probe pose on the skin, in the human frame (m).

    ``axes`` holds the probe frame as columns: lateral (the image width), elevation, and
    the inward normal (the image depth). The probe face touches ``skin``.
    """

    skin: np.ndarray
    axes: np.ndarray
    surface_xz: tuple[int, int]
    x_angle: float

    @property
    def lateral(self) -> np.ndarray:
        return self.axes[:, 0]

    @property
    def elevation(self) -> np.ndarray:
        return self.axes[:, 1]

    @property
    def normal(self) -> np.ndarray:
        return self.axes[:, 2]

    def to_dict(self) -> dict:
        return {
            "skin": self.skin.tolist(),
            "axes": self.axes.tolist(),
            "surface_xz": list(self.surface_xz),
            "x_angle": self.x_angle,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProbePose":
        return cls(
            skin=np.asarray(data["skin"], dtype=np.float64),
            axes=np.asarray(data["axes"], dtype=np.float64),
            surface_xz=tuple(data["surface_xz"]),
            x_angle=float(data["x_angle"]),
        )


def sonogym_probe_pose(patient: SonoGymPatient, x: int, z: int, x_angle: float = np.pi) -> ProbePose:
    """The probe pose SonoGym commands for surface cell (x, z) and in-plane angle ``x_angle``.

    Mirrors ``SurfaceMotionPlanner.compute_world_ee_pose_from_cmd``: depth along the
    surface normal, the image width along the projection of (cos a, 0, sin a). SonoGym
    leaves the two cross products unnormalised and lets the robot's quaternion absorb
    the difference; they are normalised here.
    """
    surface_j = int(patient.surface_map[x, z])
    if surface_j < 0:
        raise ValueError(f"surface cell ({x}, {z}) of {patient.patient_id} does not hit the body")
    normal = patient.normal_map[x, z]
    if normal[1] < 0:  # LabelImgSlicer flips the normal to point into the body (+j)
        normal = -normal
    lateral_hint = np.array([np.cos(x_angle), 0.0, np.sin(x_angle)])
    elevation = np.cross(normal, lateral_hint)
    elevation /= np.linalg.norm(elevation)
    lateral = np.cross(elevation, normal)
    return ProbePose(
        skin=np.array([x, surface_j, z], dtype=np.float64) * LABEL_RES,
        axes=np.stack([lateral, elevation, normal], axis=1),
        surface_xz=(int(x), int(z)),
        x_angle=float(x_angle),
    )


@dataclass(eq=False)
class SceneFrame:
    """Rigid map from SonoGym's human frame (m) to a prone PhysX scene (m).

    ``scene = rotation @ human + translation``. Works on numpy arrays and torch tensors
    shaped (..., 3).
    """

    rotation: np.ndarray
    translation: np.ndarray
    _torch_cache: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def prone(cls, voxel_bounds: np.ndarray) -> "SceneFrame":
        """Prone on z = 0, back up, centred in x/y on the body's bounding box.

        ``voxel_bounds`` is the (2, 3) min/max of the body surface in voxel coordinates.
        """
        bounds = np.asarray(voxel_bounds, dtype=np.float64) * LABEL_RES
        rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        translation = np.array(
            [-(bounds[0, 0] + bounds[1, 0]) / 2.0, -(bounds[0, 2] + bounds[1, 2]) / 2.0, bounds[1, 1]]
        )
        return cls(rotation=rotation, translation=translation)

    def _as(self, like):
        if isinstance(like, torch.Tensor):
            key = (like.device, like.dtype)
            if key not in self._torch_cache:
                self._torch_cache[key] = (
                    torch.as_tensor(self.rotation, device=like.device, dtype=like.dtype),
                    torch.as_tensor(self.translation, device=like.device, dtype=like.dtype),
                )
            return self._torch_cache[key]
        return self.rotation, self.translation

    def to_scene(self, points_human):
        rotation, translation = self._as(points_human)
        return points_human @ rotation.T + translation

    def to_human(self, points_scene):
        rotation, translation = self._as(points_scene)
        return (points_scene - translation) @ rotation

    def vector_to_scene(self, vectors_human):
        rotation, _ = self._as(vectors_human)
        return vectors_human @ rotation.T

    def vector_to_human(self, vectors_scene):
        rotation, _ = self._as(vectors_scene)
        return vectors_scene @ rotation

    def to_dict(self) -> dict:
        return {"rotation": self.rotation.tolist(), "translation": self.translation.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "SceneFrame":
        return cls(
            rotation=np.asarray(data["rotation"], dtype=np.float64),
            translation=np.asarray(data["translation"], dtype=np.float64),
        )
