"""SonoGym's two ultrasound simulators, driven outside Isaac Sim.

``import spinal_surgery`` registers SonoGym's tasks and with them all of Isaac Lab, so the
simulator modules are loaded by file path with only ``PACKAGE_DIR``/``PROJECT_DIR``
stubbed, the same trick ``tools/test_us_network_standalone.py`` uses. The simulators are
SonoGym's own classes with SonoGym's configs and weights; what this module adds is the
part of ``USSlicer`` that turns a probe pose into their input:

- the plane: 150 x 200 px at 0.5 mm, width along the probe's lateral axis, depth along
  the inward normal, starting ``IMAGE_HEIGHT - PROBE_HEIGHT`` = 3 mm below the skin point;
- CT preprocessing: HU clipped to the guidance task's ``ct_range`` [-300, 3000] and
  scaled to 0..255 before slicing; the network then clips that to its own
  ``CT.range`` [0, 150];
- the network's input layout, (lateral, depth). That is what the env feeds it; a
  transposed input produces streaks instead of an image;
- the model-based simulator as the guidance task runs it: ``if_use_ct=True``, impedance
  from CT and the other acoustic parameters from the converted label map, no added noise.

One thing differs on purpose. SonoGym truncates voxel coordinates to int when slicing,
which quantises the plane to 1.5 mm; a sub-voxel tissue displacement would disappear in
that rounding. CT is sampled trilinearly here and labels by nearest neighbour.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ultrasound_project.utils.sonogym_patient import (
    IMAGE_HEIGHT,
    PROBE_HEIGHT,
    SENSOR_CFG_DIR,
    SONOGYM_PACKAGE_DIR,
    SONOGYM_ROOT,
    ProbePose,
    load_yaml,
)

CT_RANGE = (-300.0, 3000.0)
"""``sim.ct_range`` of the guidance task: HU window mapped to 0..255 before slicing."""


def load_sonogym_module(name: str):
    """Import ``lab/sensors/ultrasound/<name>.py`` from the submodule without Isaac Lab."""
    if "spinal_surgery" not in sys.modules:
        stub = types.ModuleType("spinal_surgery")
        stub.PACKAGE_DIR = str(SONOGYM_PACKAGE_DIR)
        stub.PROJECT_DIR = str(SONOGYM_ROOT)
        sys.modules["spinal_surgery"] = stub
    path = SONOGYM_PACKAGE_DIR / "lab" / "sensors" / "ultrasound" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"sonogym_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ct_to_sonogym_scale(hu):
    """HU -> the 0..255 scale SonoGym slices and feeds to both simulators."""
    low, high = CT_RANGE
    if isinstance(hu, torch.Tensor):
        return (hu.clamp(low, high) - low) / (high - low) * 255.0
    return (np.clip(hu, low, high) - low) / (high - low) * 255.0


def sample_volume(volume: torch.Tensor, voxels: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Sample a (D0, D1, D2) volume at continuous voxel coordinates (..., 3).

    Coordinates outside the volume read the border, which is what SonoGym's clamping of
    slice coordinates amounts to. ``mode="bilinear"`` is trilinear on a volume.
    """
    shape = voxels.shape[:-1]
    dims = torch.tensor(volume.shape, device=voxels.device, dtype=voxels.dtype)
    grid = (2.0 * voxels / (dims - 1.0) - 1.0).flip(-1).reshape(1, 1, 1, -1, 3)
    out = F.grid_sample(
        volume[None, None].to(grid.dtype), grid, mode=mode, padding_mode="border", align_corners=True
    )
    return out.reshape(shape)


@dataclass(frozen=True)
class UltrasoundPlane:
    """USSlicer's image geometry: ``width`` x ``depth`` pixels of ``resolution`` m."""

    width: int = 150
    depth: int = 200
    resolution: float = 0.0005
    offset: float = IMAGE_HEIGHT - PROBE_HEIGHT

    @classmethod
    def from_sonogym(cls) -> "UltrasoundPlane":
        cfg = load_yaml(SENSOR_CFG_DIR / "us_cfg.yaml")
        width, depth = cfg["image_size"]
        return cls(width=int(width), depth=int(depth), resolution=float(cfg["resolution"]))

    def points(self, pose: ProbePose, advance: float = 0.0) -> np.ndarray:
        """Human-frame position of every pixel, shaped (width, depth, 3) like USSlicer.

        ``advance`` moves the probe along its normal, e.g. by the indentation it presses
        into the skin; the image is rigidly attached to the probe and moves with it.
        """
        lateral = (np.arange(self.width) - self.width // 2) * self.resolution
        depth = np.arange(self.depth) * self.resolution
        origin = pose.skin + (self.offset + advance) * pose.normal
        return origin + lateral[:, None, None] * pose.lateral + depth[None, :, None] * pose.normal

    def extent_mm(self) -> list[float]:
        """matplotlib ``extent`` (left, right, bottom, top) of a depth-down image, in mm."""
        half = self.width // 2 * self.resolution * 1e3
        return [-half, (self.width - self.width // 2) * self.resolution * 1e3, self.depth * self.resolution * 1e3, 0.0]


class MaterialSpeckle:
    """SonoGym's T0/T1 random maps, attached to the tissue instead of to the image.

    The model-based simulator draws its scatterers from two Gaussian random volumes that
    ``USSlicer`` stores in the human frame at the image resolution and reads by nearest
    neighbour. For a rigid patient the human frame is the material frame, so sampling
    them at material coordinates is the same thing -- and for a deformed one it makes the
    speckle move with the tissue, which is what real speckle does.
    """

    def __init__(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        resolution: float,
        device: str | torch.device,
        seed: int = 0,
    ):
        self.resolution = resolution
        self.origin = torch.as_tensor(lower, dtype=torch.float32, device=device)
        size = np.ceil((np.asarray(upper) - np.asarray(lower)) / resolution).astype(int) + 1
        generator = torch.Generator(device=device).manual_seed(seed)
        self.maps = torch.randn((2, *size.tolist()), generator=generator, device=device)
        self.upper_index = torch.tensor(size - 1, device=device)

    @classmethod
    def around_plane(cls, plane_points: np.ndarray, margin: float, resolution: float, device, seed: int = 0):
        """Covers the human-frame bounding box of a plane plus ``margin`` on every side."""
        flat = plane_points.reshape(-1, 3)
        return cls(flat.min(axis=0) - margin, flat.max(axis=0) + margin, resolution, device, seed)

    def sample(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(T0, T1) at human-frame points (..., 3)."""
        index = torch.round((points - self.origin) / self.resolution).long()
        index = torch.minimum(index.clamp(min=0), self.upper_index)
        values = self.maps[:, index[..., 0], index[..., 1], index[..., 2]]
        return values[0], values[1]


class SonoGymUltrasound:
    """Renders B-mode images with SonoGym's pix2pix network and its model-based simulator.

    Images go in and come out depth-down, ``(batch, depth, width)``.

    The network matches the intensity histogram of its input to its training data, and
    SonoGym computes that lookup table once, on the first image it ever sees
    (``reset_hist_interval`` is 10^7). :meth:`render_net` keeps that behaviour but makes
    the first image explicit: pass ``calibrate=True`` with the reference image, and every
    later call reuses its table, so two renders differ only where their inputs do.
    """

    def __init__(self, device: str | torch.device, seed: int = 0):
        self.device = torch.device(device)
        self.us_cfg = load_yaml(SENSOR_CFG_DIR / "us_cfg.yaml")
        self.net_cfg = load_yaml(SENSOR_CFG_DIR / "us_generative_cfg.yaml")
        self.net = load_sonogym_module("simulate_US_network").USSimulatorNetwork(self.net_cfg, self.device)
        # with several checkpoints configured SonoGym swaps between them at random every
        # model_change_interval images; keep whichever the calibration image picked
        self.net.model_k = 10**12
        self.conv = load_sonogym_module("simulate_US_conv").USSimulatorConv(self.us_cfg, self.device)
        size = int(self.us_cfg["large_scale_resolution"])
        generator = torch.Generator(device=self.device).manual_seed(seed + 1)
        # the large-scale speckle is a 10 x 10 gain map over 15 mm cells of the image; it
        # is kept fixed rather than tissue-anchored, which is far below what it can resolve
        self.large_scale = torch.randn((1, size, size), generator=generator, device=self.device)
        self._calibrated = False

    @torch.no_grad()
    def render_net(self, ct_scaled: torch.Tensor, calibrate: bool = False) -> torch.Tensor:
        """``ct_scaled``: (batch, depth, width) CT on SonoGym's 0..255 scale."""
        if calibrate:
            self.net.step = 0
        elif not self._calibrated:
            raise RuntimeError("render the reference image with calibrate=True first")
        elif self.net.step % self.net.k == 0:
            self.net.step += 1
        image = self.net.simulate_US_image(ct_scaled.transpose(1, 2)[:, None].float())
        self._calibrated = True
        return image[:, 0].transpose(1, 2)

    @torch.no_grad()
    def render_conv(
        self, labels: torch.Tensor, ct_scaled: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor
    ) -> torch.Tensor:
        """All inputs (batch, depth, width): converted labels, CT 0..255, speckle maps."""
        batch = labels.shape[0]
        return self.conv.simulate_US_image_given_rand_map(
            labels.to(torch.uint8),
            t0.float(),
            t1.float(),
            self.large_scale.expand(batch, -1, -1),
            False,
            True,
            ct_scaled.float(),
        )
