"""Comparing ultrasound images, and measuring tissue motion from them.

The similarity scores are the usual ones; :func:`block_match` is speckle tracking -- the
way ultrasound elastography measures displacement -- and gives an estimate of the tissue
motion between two images that knows nothing about the FEM it came from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from skimage.feature import match_template
from skimage.metrics import structural_similarity


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-normalised cross-correlation of two images."""
    a = a.astype(np.float64).ravel() - a.mean()
    b = b.astype(np.float64).ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def psnr(reference: np.ndarray, image: np.ndarray, data_range: float) -> float:
    mse = np.mean((reference.astype(np.float64) - image) ** 2)
    return float(10.0 * np.log10(data_range**2 / mse)) if mse > 0 else float("inf")


def ssim(reference: np.ndarray, image: np.ndarray, data_range: float) -> float:
    return float(structural_similarity(reference, image, data_range=data_range))


def compare(reference: np.ndarray, image: np.ndarray, data_range: float) -> dict[str, float]:
    """SSIM, PSNR, NCC and mean absolute error of ``image`` against ``reference``."""
    return {
        "ssim": ssim(reference, image, data_range),
        "psnr_db": psnr(reference, image, data_range),
        "ncc": ncc(reference, image),
        "mae": float(np.mean(np.abs(reference.astype(np.float64) - image))),
    }


@dataclass
class BlockMatch:
    """Block-matching result on a grid of block centres."""

    rows: np.ndarray  # (R,) pixel row (depth) of each block centre
    cols: np.ndarray  # (C,) pixel column (lateral)
    displacement: np.ndarray  # (R, C, 2) motion of each block in pixels, (depth, lateral)
    correlation: np.ndarray  # (R, C) peak normalised cross-correlation


def parabolic_offset(scores: np.ndarray, peak: int) -> float:
    """Sub-sample offset of a 1D peak from the parabola through it and its neighbours."""
    if peak <= 0 or peak >= len(scores) - 1:
        return 0.0
    left, centre, right = scores[peak - 1], scores[peak], scores[peak + 1]
    denominator = left - 2.0 * centre + right
    return float(0.5 * (left - right) / denominator) if denominator < 0 else 0.0


def block_match(
    reference: np.ndarray,
    image: np.ndarray,
    block: int = 25,
    step: int = 8,
    search: tuple[int, int] = (20, 6),
) -> BlockMatch:
    """Where each ``block`` x ``block`` patch of ``reference`` went in ``image``.

    Patches are taken every ``step`` pixels wherever the search window, ``search`` pixels
    along depth and lateral on each side, fits in the image. The match is the peak of the
    normalised cross-correlation, refined to sub-pixel by a parabola along each axis.
    """
    half = block // 2
    sr, sc = search
    rows = np.arange(half + sr, reference.shape[0] - half - sr, step)
    cols = np.arange(half + sc, reference.shape[1] - half - sc, step)
    displacement = np.zeros((len(rows), len(cols), 2))
    correlation = np.zeros((len(rows), len(cols)))
    reference = reference.astype(np.float64)
    image = image.astype(np.float64)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            template = reference[r - half : r + half + 1, c - half : c + half + 1]
            window = image[r - half - sr : r + half + sr + 1, c - half - sc : c + half + sc + 1]
            if template.std() < 1e-9 or window.std() < 1e-9:
                correlation[i, j] = np.nan
                displacement[i, j] = np.nan
                continue
            scores = match_template(window, template)
            pr, pc = np.unravel_index(np.argmax(scores), scores.shape)
            displacement[i, j] = (
                pr - sr + parabolic_offset(scores[:, pc], pr),
                pc - sc + parabolic_offset(scores[pr, :], pc),
            )
            correlation[i, j] = scores[pr, pc]
    return BlockMatch(rows=rows, cols=cols, displacement=displacement, correlation=correlation)


def accumulate(matches: list[BlockMatch], min_correlation: float = 0.6) -> np.ndarray:
    """Chain block matches between consecutive images into the motion from the first to the last.

    All matches must share one block grid. Every block centre of the first image is moved
    by the first match, the second match is interpolated where the centre then is, and so
    on -- how elastography builds a large deformation out of small, well-correlated steps
    that a single match across the whole of it would lose to decorrelation. A path that
    meets a block below ``min_correlation`` comes out NaN. Returns (R, C, 2) pixels.
    """
    rows, cols = matches[0].rows, matches[0].cols
    start = np.stack(np.meshgrid(rows, cols, indexing="ij"), axis=-1).astype(np.float64)
    position = start.copy()

    def sample(values: np.ndarray) -> np.ndarray:
        return RegularGridInterpolator((rows, cols), values, bounds_error=False, fill_value=np.nan)(position)

    for match in matches:
        good = match.correlation >= min_correlation
        # interpolating NaN directly would also poison neighbours it has zero weight on
        displacement = np.where(good[..., None], match.displacement, 0.0)
        step = np.stack([sample(displacement[..., k]) for k in range(2)], axis=-1)
        reliable = sample(good.astype(np.float64)) > 1.0 - 1e-9
        position = position + np.where(reliable[..., None], step, np.nan)
    return position - start
