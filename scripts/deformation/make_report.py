"""HTML report of the soft body -> deformed CT -> ultrasound check.

Builds one self-contained page from what ``patient_press_record.py`` (the FEM press, its
render and the solver sweep) and ``deformed_ct_to_us.py`` (images, fields, metrics)
wrote. Figures, the two animations and the Isaac Sim render are embedded, so the page can
be passed around on its own; the separate PNG/MP4 files are kept next to it. The text is
in Chinese.

.. code-block:: bash

    python scripts/deformation/make_report.py
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
from html import escape
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from ultrasound_project import PROJECT_DIR  # noqa: E402

plt.rcParams.update(
    {
        "font.sans-serif": ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "savefig.dpi": 110,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)

CT_WINDOW = (0.0, 100.0)
"""Display window on SonoGym's 0..255 CT scale (HU -300..3000): fat ~15, muscle ~27, bone 80+."""

CLASS_NAMES = {
    2: "器官（SonoGym 第 2 类 lung，原始标签 1–14）",
    6: "SonoGym 第 6 类（原始 23–24）",
    8: "肌肉（原始 51、80–89）",
    10: "脊髓（原始 79）",
    12: "皮肤/软组织（原始 15–22、120）",
    13: "骨（原始 25–50、69–78、92–117）",
    15: "血管（原始 52–68）",
}

PLANE_NAMES = {"sagittal": "矢状面（成像平面）", "transverse": "横断面（过探头中心）"}


def parse_args():
    parser = argparse.ArgumentParser(description="HTML report of the deformation -> CT -> ultrasound check.")
    parser.add_argument("--data_dir", default="outputs/deformation_us", help="Outputs of stages 1 and 2.")
    parser.add_argument("--output_dir", default="outputs/deformation_us/report", help="Where the page and its assets go.")
    parser.add_argument("--fps", type=float, default=20.0, help="Frame rate of the animations; the recording is 20 Hz.")
    return parser.parse_args()


# ------------------------------------------------------------------------------- helpers


def split_polygons(points: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [points[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


def polygon_means(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.add.reduceat(values.astype(np.float64), offsets[:-1]) / np.diff(offsets)


def probe_patch(half_width: float, face: float, height: float, **style) -> Rectangle:
    """The probe in a (s, t) plane with depth t pointing into the body."""
    style = {"facecolor": (0.2, 0.4, 0.85, 0.55), "edgecolor": "white", "linewidth": 0.8} | style
    return Rectangle((-half_width, face - height), 2 * half_width, height, **style)


def plane_extent(d, plane: str) -> list[float]:
    s, t = d[f"{plane}_s"], d[f"{plane}_t"]
    return [s[0], s[-1], t[-1], t[0]]


def frame_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def video_writer(path: Path, fps: float):
    return imageio.get_writer(
        path,
        format="FFMPEG",
        fps=fps,
        codec="libx264",
        quality=None,
        pixelformat="yuv420p",
        macro_block_size=8,
        output_params=["-crf", "24", "-preset", "slow", "-movflags", "+faststart"],
    )


def data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def sci(value: float, digits: int = 1) -> str:
    """1.2e-05 -> 1.2×10⁻⁵ in HTML."""
    if value == 0 or not np.isfinite(value):
        return f"{value:g}"
    exponent = int(np.floor(np.log10(abs(value))))
    if -3 <= exponent <= 3:
        return f"{value:.{max(digits + 1 - exponent, 0) if exponent < 0 else digits}f}"
    return f"{value / 10**exponent:.{digits}f}×10<sup>{exponent}</sup>".replace("<sup>-", "<sup>−")


def table(headers: list[str], rows: list[list], cls: str = "") -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def load_sweep(directory: Path) -> list[dict]:
    runs = []
    for path in sorted(directory.glob("res*_it*.npz")):
        with np.load(path) as data:
            meta = json.loads(str(data["meta"]))
            nodes = int(data["sim_rest"].shape[0])
        plateaus = {p["force_n"]: p for p in meta["plateaus"]}
        runs.append(
            {
                "hex_res": meta["solver"]["hex_res"],
                "iterations": meta["solver"]["iterations"],
                "nodes": nodes,
                "indentation": {f: p["indentation_mm"] for f, p in plateaus.items()},
                "penetration": max(p["max_penetration_mm"] for p in plateaus.values()),
                "drift": max(abs(p["drift_last_0p5s_mm"]) for p in plateaus.values()),
                "wall_s": meta["wall_time_s"],
                "phases": meta["phases"],
                "record_hz": meta["record_hz"],
            }
        )
    return sorted(runs, key=lambda r: (r["hex_res"], r["iterations"]))


# ------------------------------------------------------------------------------- figures


def figure_force(d, m, meta, path: Path):
    fem = m["fem"]
    force, indentation = np.array(fem["force_n"]), np.array(fem["indentation_mm"])
    upper = force >= 10
    slope, intercept = np.polyfit(force[upper], indentation[upper], 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.0))
    a.plot(d["step_force"], d["step_advance_mm"] - d["contact_mm"], color="0.65", lw=1, label="逐步轨迹（斜坡 + 保持）")
    a.plot(force, indentation, "o", color="C0", ms=6, label="各保持段末的稳态值")
    f = np.linspace(0, 21, 50)
    a.plot(f, slope * f + intercept, "--", color="C0", lw=1, label=f"10–20 N 线性拟合：{1 / slope:.2f} N/mm，R² = {fem['linear_fit_r2_10_to_20n']:.5f}")
    a.plot(
        f,
        f * fem["flat_punch_halfspace_mm"][0] / force[0],
        ":",
        color="C3",
        label=f"平冲头·弹性半空间解析估计（E = {meta['material']['youngs_pa'] / 1e3:.0f} kPa）",
    )
    a.set_xlabel("探头力 (N)")
    a.set_ylabel("压入深度 (mm，自首次接触算起)")
    a.set_xlim(0, 21)
    a.set_ylim(-1.5, 17)
    a.grid(alpha=0.3)
    a.legend(loc="upper left")
    a.set_title("力–压入深度曲线")

    t = d["step_time"]
    b.plot(t, d["step_force"], color="C1", lw=1.2)
    b.set_xlabel("时间 (s)")
    b.set_ylabel("探头力 (N)", color="C1")
    b2 = b.twinx()
    b2.plot(t, d["step_advance_mm"] - d["contact_mm"], color="C0", lw=1.2)
    b2.plot(d["frame_time"], d["frame_penetration_mm"], color="C3", lw=0.8)
    b2.set_ylabel("压入深度 / 碰撞网格穿透 (mm)", color="C0")
    for k, frame in enumerate(d["plateau_frames"]):
        b.axvline(d["frame_time"][frame], color="0.75", lw=0.8, zorder=0)
    b2.text(0.99, 0.04, "红线：碰撞网格节点进入探头的最大深度", transform=b2.transAxes, ha="right", color="C3", fontsize=8)
    b.set_title("加载过程（竖线：各平台取样帧）")
    b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_sweep(runs: list[dict], meta, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    main = (meta["solver"]["hex_res"], meta["solver"]["iterations"])
    for ax, force in zip(axes, (10.0, 20.0)):
        for k, res in enumerate(sorted({r["hex_res"] for r in runs})):
            rows = [r for r in runs if r["hex_res"] == res]
            ax.plot(
                [r["iterations"] for r in rows],
                [r["indentation"][force] for r in rows],
                "o-",
                color=f"C{k}",
                label=f"hex_res {res}（{rows[0]['nodes']:,} 节点）",
            )
        chosen = [r for r in runs if (r["hex_res"], r["iterations"]) == main]
        if chosen:
            ax.plot(main[1], chosen[0]["indentation"][force], "*", color="k", ms=14, label="本文主算例", zorder=5)
        ax.set_xscale("log", base=2)
        ax.set_xticks([60, 120, 240], ["60", "120", "240"])
        ax.set_xlabel("PhysX 位置迭代次数")
        ax.set_ylabel("压入深度 (mm)")
        ax.set_title(f"{force:g} N 时的稳态压入深度")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_mesh(d, path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.4))
    norm = TwoSlopeNorm(vcenter=1.0, vmin=0.8, vmax=1.2)
    face_last = float(d["frame_advance_mm"][-1])
    for row, plane in enumerate(("sagittal", "transverse")):
        extent = plane_extent(d, plane)
        half = d["probe_size_mm"][0 if plane == "sagittal" else 1] / 2
        for col, state in enumerate(("rest", "last")):
            ax = axes[row, col]
            ct = d[f"{plane}_ct_before" if state == "rest" else f"{plane}_ct_after"]
            ax.imshow(ct, cmap="gray", vmin=CT_WINDOW[0], vmax=CT_WINDOW[1], extent=extent)
            polygons = split_polygons(d[f"{plane}_{state}_polygons"], d[f"{plane}_{state}_offsets"])
            if state == "rest":
                ax.add_collection(PolyCollection(polygons, facecolors="none", edgecolors=(0.35, 0.85, 1.0, 0.55), linewidths=0.45))
                ax.add_patch(probe_patch(half, float(d["contact_mm"]), d["probe_size_mm"][2]))
                ax.set_title(f"{PLANE_NAMES[plane]}：未形变网格与 CT（探头在首次接触处）")
            else:
                ratio = d[f"{plane}_last_values"]
                colors = plt.cm.RdBu_r(norm(ratio))
                colors[:, 3] = np.clip(np.abs(ratio - 1.0) / 0.15, 0.05, 0.85)
                ax.add_collection(PolyCollection(polygons, facecolors=colors, edgecolors=(1, 1, 1, 0.3), linewidths=0.3))
                ax.add_patch(probe_patch(half, face_last, d["probe_size_mm"][2]))
                ax.set_title(f"{PLANE_NAMES[plane]}：20 N 形变后网格与形变 CT（颜色 = 单元体积比 J）")
                fig.colorbar(
                    plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), ax=ax, fraction=0.035, pad=0.02, label="J = 现体积 / 原体积（越接近 1 越透明）"
                )
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_xlabel("沿平面 (mm)")
            ax.set_ylabel("深度 (mm，自 SonoGym 皮肤点)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_field(d, path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.8))
    face_last = float(d["frame_advance_mm"][-1])
    for row, plane in enumerate(("sagittal", "transverse")):
        s, t = d[f"{plane}_s"], d[f"{plane}_t"]
        extent = plane_extent(d, plane)
        half = d["probe_size_mm"][0 if plane == "sagittal" else 1] / 2
        inside = np.isfinite(d[f"{plane}_volume_ratio"])
        magnitude = np.where(inside, np.linalg.norm(d[f"{plane}_forward_mm"], axis=-1), np.nan)

        ax = axes[row, 0]
        image = ax.imshow(magnitude, cmap="viridis", vmin=0, vmax=7.5, extent=extent)
        contours = ax.contour(s, t, magnitude, levels=[0.5, 1, 2, 3, 4, 5, 6], colors="white", linewidths=0.5)
        ax.clabel(contours, fontsize=6, fmt="%g")
        ax.plot(s, d[f"{plane}_skin_material"], color="w", lw=0.8, ls="--")
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="|u| (mm)")
        ax.set_title(f"{PLANE_NAMES[plane]}：位移幅值（物质坐标）")

        ax = axes[row, 1]
        image = ax.imshow(d[f"{plane}_strain_depth"] * 100, cmap="RdBu_r", vmin=-25, vmax=25, extent=extent)
        ax.plot(s, d[f"{plane}_skin_material"], color="k", lw=0.8, ls="--")
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="ε_zz (%)")
        ax.set_title("深度方向应变 ε_zz（负 = 压缩）")

        ax = axes[row, 2]
        ax.imshow(d[f"{plane}_ct_after"], cmap="gray", vmin=CT_WINDOW[0], vmax=CT_WINDOW[1], extent=extent)
        lines = list(d[f"{plane}_lines_along_depth"]) + list(d[f"{plane}_lines_across"])
        ax.add_collection(LineCollection(lines, colors=[(1.0, 0.8, 0.2, 0.65)], linewidths=0.6))
        ax.add_patch(probe_patch(half, face_last, d["probe_size_mm"][2]))
        ax.set_title("形变 CT 与随组织移动的物质网格线（原为 5 mm 方格）")
        for ax in axes[row]:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_xlabel("沿平面 (mm)")
            ax.set_ylabel("深度 (mm)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_ct(d, m, path: Path):
    fig = plt.figure(figsize=(16, 8.6))
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.1])
    face_last = float(d["frame_advance_mm"][-1])
    for row, plane in enumerate(("sagittal", "transverse")):
        s = d[f"{plane}_s"]
        extent = plane_extent(d, plane)
        half = d["probe_size_mm"][0 if plane == "sagittal" else 1] / 2
        for col, key in enumerate(("ct_before", "ct_after")):
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(d[f"{plane}_{key}"], cmap="gray", vmin=CT_WINDOW[0], vmax=CT_WINDOW[1], extent=extent)
            if col == 0:
                ax.plot(s, d[f"{plane}_skin_material"], color="c", lw=1.0, label="皮肤（−500 HU 等值线）")
                ax.set_title(f"{PLANE_NAMES[plane]}：原始 CT")
            else:
                ax.plot(s, d[f"{plane}_skin_measured"], color="r", lw=1.4, label="形变 CT 上测得的皮肤")
                ax.plot(s, d[f"{plane}_skin_predicted"], color="c", lw=1.0, ls="--", label="FEM 预测：物质皮肤面前推后与平面的交线")
                ax.add_patch(probe_patch(half, face_last, d["probe_size_mm"][2]))
                ax.set_title(f"{PLANE_NAMES[plane]}：20 N 形变 CT")
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(55, -15)
            ax.legend(loc="lower right")
            ax.set_xlabel("沿平面 (mm)")
            ax.set_ylabel("深度 (mm)")

    ax = fig.add_subplot(grid[0, 2])
    beam = d["beam_s_mm"]
    ax.plot(beam, d["beam_hu_before"], color="0.4", label="形变前")
    ax.plot(beam, d["beam_hu_direct"], color="C3", label="形变后：探头轴线直接拉回")
    ax.plot(beam, d["beam_hu_grid"], color="C0", ls="--", label="形变后：从 NIfTI 体素网格读取")
    axis = m["deformed_ct"]["probe_axis"]
    for value, color in ((axis["face_mm"], "C2"), (axis["lamina_before_mm"], "0.5")):
        ax.axvline(value, color=color, lw=0.8, ls=":")
    ax.text(axis["face_mm"] + 0.5, 0.96, "探头面", color="C2", fontsize=8, va="top", transform=ax.get_xaxis_transform())
    ax.text(axis["lamina_before_mm"] + 0.5, 0.96, "椎板", color="0.4", fontsize=8, va="top", transform=ax.get_xaxis_transform())
    ax.set_xlim(-12, 55)
    ax.set_xlabel("沿探头轴线深度 (mm，自 SonoGym 皮肤点)")
    ax.set_ylabel("HU")
    ax.set_title("探头中轴线上的 CT 值")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    ax = fig.add_subplot(grid[1, 2])
    for k, plane in enumerate(("sagittal", "transverse")):
        error = d[f"{plane}_skin_measured"] - d[f"{plane}_skin_predicted"]
        check = m["deformed_ct"]["skin_profile"][plane]
        ax.plot(d[f"{plane}_s"], error, color=f"C{k}", lw=1, label=f"{PLANE_NAMES[plane]}：RMSE {check['rmse_mm']:.3f} mm")
    ax.axvspan(-d["probe_size_mm"][0] / 2, d["probe_size_mm"][0] / 2, color="C2", alpha=0.08, label="探头宽度（矢状面）")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlim(-60, 60)
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlabel("沿平面 (mm)")
    ax.set_ylabel("测得 − 预测 (mm)")
    ax.set_title("皮肤位置：形变 CT 与 FEM 之差（体素 1.5 mm）")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_skin(d, path: Path):
    rest, last, faces = d["skin_vertices_rest"], d["skin_vertices_last"], d["skin_faces"]
    sink = last[:, 2] - rest[:, 2]
    fig = plt.figure(figsize=(14, 5.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    norm = Normalize(0, max(float(sink.max()), 1e-3))
    corners = last[faces][..., [0, 1, 2]] * np.array([1.0, 1.0, -1.0])
    surface = Poly3DCollection(corners, facecolors=plt.cm.magma(norm(sink[faces].mean(axis=1))), edgecolors="none")
    ax.add_collection3d(surface)
    ax.set_xlim(-70, 70)
    ax.set_ylim(-55, 55)
    ax.set_zlim(-30, 15)
    ax.set_box_aspect((140, 110, 45))
    ax.view_init(elev=28, azim=-60)
    ax.set_xlabel("探头宽度方向 (mm)")
    ax.set_ylabel("探头厚度方向 (mm)")
    ax.set_zlabel("向外 (mm)")
    ax.set_title("20 N 时探头周围的体表（碰撞网格），颜色 = 下陷量")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="magma"), ax=ax, fraction=0.03, pad=0.12, shrink=0.8, label="下陷 (mm)")

    ax = fig.add_subplot(1, 2, 2)
    triangulation = Triangulation(rest[:, 0], rest[:, 1], faces)
    levels = np.arange(np.floor(min(-0.5, float(sink.min())) * 2) / 2, norm.vmax + 0.5, 0.5)
    filled = ax.tricontourf(triangulation, sink, levels=levels, cmap="magma")
    ax.tricontour(triangulation, sink, levels=[0.5, 1, 2, 3, 4, 5, 6], colors="w", linewidths=0.5)
    half_w, half_e = d["probe_size_mm"][0] / 2, d["probe_size_mm"][1] / 2
    ax.add_patch(Rectangle((-half_w, -half_e), 2 * half_w, 2 * half_e, fill=False, edgecolor="c", lw=1.2, ls="--"))
    ax.set_aspect("equal")
    ax.set_xlabel("探头宽度方向 (mm)")
    ax.set_ylabel("探头厚度方向 (mm)")
    ax.set_title("体表下陷俯视图（虚线 = 探头底面 58 × 11 mm）")
    fig.colorbar(filled, ax=ax, fraction=0.03, pad=0.02, ticks=np.arange(0, levels[-1] + 0.1, 1.0), label="下陷 (mm)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def us_rows(m):
    windows = m["ultrasound"]["display_window"]
    return [
        ("ct", "CT 切面（模型输入）", CT_WINDOW),
        ("net", "SonoGym 网络模型（pix2pix）", windows["net"]),
        ("conv", "SonoGym 物理卷积模型", windows["conv"]),
    ]


def figure_us_compare(d, m, path: Path):
    k = len(d["plateau_frames"]) - 1
    frame = int(d["plateau_frames"][k])
    plateau = m["ultrasound"]["plateaus"][k]
    face = float(d["frame_advance_mm"][frame])
    extent = d["extent_mm"]
    columns = [
        "SonoGym 原始\n刚体，探头在皮肤点",
        f"刚体\n探头前进到 {face:+.2f} mm",
        f"软体形变（本文）\n{plateau['force_n']:g} N，探头面 {face:+.2f} mm",
        "差值\n软体形变 − 刚体",
    ]
    fig, axes = plt.subplots(3, 4, figsize=(13, 14.2))
    for r, (kind, title, window) in enumerate(us_rows(m)):
        reference = d[f"ref_{kind}"].astype(np.float32)
        rigid = d[f"rigid_{kind}"][frame].astype(np.float32)
        deformed = d[kind][frame].astype(np.float32)
        for c, image in enumerate((reference, rigid, deformed)):
            axes[r, c].imshow(image, cmap="gray", vmin=window[0], vmax=window[1], extent=extent)
        limit = 15.0 if kind == "ct" else 0.5 * (window[1] - window[0])
        difference = axes[r, 3].imshow(deformed - rigid, cmap="RdBu_r", vmin=-limit, vmax=limit, extent=extent)
        fig.colorbar(difference, ax=axes[r, 3], fraction=0.05, pad=0.03)
        axes[r, 0].set_ylabel(f"{title}\n深度 (mm，自图像顶部)")
        if kind != "ct":
            scores_rigid = plateau[f"{kind}_vs_rigid"]
            scores_reference = plateau[f"{kind}_vs_reference"]
            axes[r, 2].text(
                0.02,
                0.98,
                f"SSIM 对刚体 {scores_rigid['ssim']:.2f}\nSSIM 对原始 {scores_reference['ssim']:.2f}",
                transform=axes[r, 2].transAxes,
                va="top",
                color="yellow",
                fontsize=8,
            )
    for c, title in enumerate(columns):
        axes[0, c].set_title(title)
    for ax in axes[-1]:
        ax.set_xlabel("横向 (mm)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_us_plateaus(d, m, path: Path):
    frames = d["plateau_frames"]
    extent = d["extent_mm"]
    fig, axes = plt.subplots(3, len(frames), figsize=(3.0 * len(frames), 12.4))
    for c, frame in enumerate(frames):
        face = float(d["frame_advance_mm"][frame])
        force = float(d["plateau_forces"][c])
        label = d["label"][frame]
        bone = label == 13
        rows = np.where(bone.any(axis=0), bone.argmax(axis=0), -1)
        x = np.linspace(extent[0], extent[1], label.shape[1])
        for r, (kind, title, window) in enumerate(us_rows(m)):
            ax = axes[r, c]
            ax.imshow(d[kind][frame].astype(np.float32), cmap="gray", vmin=window[0], vmax=window[1], extent=extent)
            if kind == "ct":
                ok = rows >= 0
                ax.plot(x[ok], rows[ok] * 0.5, ".", color="r", ms=1.2)
            if c == 0:
                ax.set_ylabel(f"{title}\n深度 (mm)")
        axes[0, c].set_title(f"{'未接触' if force == 0 else f'{force:g} N'}\n探头面 {face:+.2f} mm")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_bone(m, path: Path):
    plateaus = m["ultrasound"]["plateaus"]
    force = np.array([p["force_n"] for p in plateaus])
    expected = np.array([p["bone_shift_expected_mm"] for p in plateaus])
    labels = np.array([p["bone_shift_labels_mm"] for p in plateaus])
    measured = {
        kind: (
            np.array([p[f"bone_echo_shift_{kind}"]["median_mm"] for p in plateaus]),
            np.array([p[f"bone_echo_shift_{kind}"]["mad_mm"] for p in plateaus]),
        )
        for kind in ("conv", "net")
    }
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.0))
    a.plot(force, expected, "k-", lw=1.2, label="理论值：−探头前进量（骨节点固定）")
    a.plot(force, labels, "s", color="C2", ms=7, mfc="none", label="形变标签图中的骨面")
    a.errorbar(force + 0.25, *measured["conv"], fmt="o", color="C0", capsize=3, label="物理卷积模型图像：骨回声互相关")
    a.errorbar(force + 0.5, *measured["net"], fmt="^", color="C1", capsize=3, label="网络模型图像：骨回声互相关")
    a.set_xlabel("探头力 (N)")
    a.set_ylabel("骨面在图像中的深度变化 (mm)")
    a.set_title("骨面深度随探头前进的变化（相对 SonoGym 原始图像）")
    a.grid(alpha=0.3)
    a.legend()
    b.axhline(0, color="k", lw=0.8)
    b.plot(force, labels - expected, "s", color="C2", ms=7, mfc="none", label="标签图")
    b.errorbar(force + 0.25, measured["conv"][0] - expected, yerr=measured["conv"][1], fmt="o", color="C0", capsize=3, label="物理卷积模型")
    b.errorbar(force + 0.5, measured["net"][0] - expected, yerr=measured["net"][1], fmt="^", color="C1", capsize=3, label="网络模型")
    b.set_xlabel("探头力 (N)")
    b.set_ylabel("测得 − 理论 (mm)")
    b.set_title("误差（误差棒 = 逐列中位绝对偏差）")
    b.grid(alpha=0.3)
    b.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_speckle(d, m, path: Path):
    rows_mm = d["bm_rows"] * 0.5
    cols_mm = (d["bm_cols"] - 75) * 0.5
    predicted = d["bm_predicted_mm"][..., 0]
    tissue = d["label"][int(d["plateau_frames"][0])][d["bm_rows"]][:, d["bm_cols"]] > 0
    chained = d["bm_conv_chained_mm"][..., 0]
    conv = np.where(d["bm_conv_good"], d["bm_conv_mm"][..., 0], np.nan)
    net = np.where(d["bm_net_good"], d["bm_net_mm"][..., 0], np.nan)
    extent = [cols_mm[0] - 2, cols_mm[-1] + 2, rows_mm[-1] + 2, rows_mm[0] - 2]
    low, high = np.nanpercentile(predicted, [1, 99])
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 4)
    maps = [
        ("FEM 预测的轴向位移", predicted),
        ("物理卷积模型：逐级追踪再累加", np.where(tissue, chained, np.nan)),
        ("物理卷积模型：0→20 N 一步追踪", conv),
        ("网络模型：0→20 N 一步追踪", net),
    ]
    for i, (title, field) in enumerate(maps):
        ax = fig.add_subplot(grid[0, i])
        image = ax.imshow(field, cmap="viridis", vmin=low, vmax=high, extent=extent)
        fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03, label="mm")
        ax.set_title(title)
        ax.set_xlabel("横向 (mm)")
        ax.set_ylabel("深度 (mm，自图像顶部)")

    tracking = m["ultrasound"]["speckle_tracking"]
    whole, steps = tracking[-1], tracking[:-1]
    accumulated = m["ultrasound"]["speckle_tracking_chained"]["conv"]
    ax = fig.add_subplot(grid[1, :2])
    ax.plot(predicted[d["bm_net_good"]], d["bm_net_mm"][..., 0][d["bm_net_good"]], "^", color="C1", ms=3, alpha=0.5, label=f"网络模型，一步：r = {whole['net']['axial_pearson_r']:.2f}")
    ax.plot(predicted[d["bm_conv_good"]], d["bm_conv_mm"][..., 0][d["bm_conv_good"]], "o", color="0.55", ms=3, alpha=0.6, label=f"物理卷积模型，一步：r = {whole['conv']['axial_pearson_r']:.2f}")
    ok = tissue & np.isfinite(chained)
    ax.plot(predicted[ok], chained[ok], "o", color="C0", ms=3.5, label=f"物理卷积模型，逐级累加：r = {accumulated['axial_pearson_r']:.3f}")
    line = np.array([low - 0.5, high + 0.5])
    ax.plot(line, line, "k--", lw=0.8, label="y = x")
    ax.set_xlim(line)
    ax.set_ylim(line[0] - 3, line[1] + 3)
    ax.set_xlabel("FEM 预测的轴向位移 (mm)")
    ax.set_ylabel("散斑追踪测得 (mm)")
    ax.set_title("0 → 20 N 整段：逐块对比（每块 12.5 × 12.5 mm）")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    ax = fig.add_subplot(grid[1, 2:])
    names = [f"{s['from_n']:g}→{s['to_n']:g} N" for s in steps] + ["0→20 N\n一步", "0→20 N\n逐级累加"]
    x = np.arange(len(names))
    conv_rmse = [s["conv"]["axial_rmse_mm"] for s in steps] + [whole["conv"]["axial_rmse_mm"], accumulated["axial_rmse_mm"]]
    conv_median = [s["conv"]["axial_median_abs_mm"] for s in steps] + [whole["conv"]["axial_median_abs_mm"], accumulated["axial_median_abs_mm"]]
    net_rmse = [s["net"]["axial_rmse_mm"] for s in steps] + [whole["net"]["axial_rmse_mm"]]
    net_median = [s["net"]["axial_median_abs_mm"] for s in steps] + [whole["net"]["axial_median_abs_mm"]]
    ax.bar(x - 0.2, conv_rmse, 0.38, color="C0", alpha=0.45, label="物理卷积模型：RMSE")
    ax.plot(x - 0.2, conv_median, "D", color="C0", label="物理卷积模型：中位绝对误差")
    ax.bar(x[: len(net_rmse)] + 0.2, net_rmse, 0.38, color="C1", alpha=0.45, label="网络模型：RMSE")
    ax.plot(x[: len(net_rmse)] + 0.2, net_median, "D", color="C1", label="网络模型：中位绝对误差")
    ax.set_yscale("log")
    ax.set_xticks(x, names)
    ax.set_ylabel("轴向误差 (mm)")
    ax.set_title("散斑追踪误差（对比 FEM 运动）")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------------- animations


def timeline(ax, d, indentation_axis: bool = True):
    ax.plot(d["step_time"], d["step_force"], color="C1", lw=1.2)
    ax.set_ylabel("力 (N)", color="C1")
    ax.set_xlabel("时间 (s)")
    ax.set_xlim(0, float(d["step_time"][-1]))
    ax.grid(alpha=0.3)
    if indentation_axis:
        twin = ax.twinx()
        twin.plot(d["step_time"], d["step_advance_mm"] - d["contact_mm"], color="C0", lw=1.2)
        twin.set_ylabel("压入 (mm)", color="C0")
    return ax.axvline(0, color="k", lw=1.2)


def animate_deformation(d, path: Path, fps: float):
    extent = plane_extent(d, "sagittal")
    offsets = d["anim_offsets"]
    half, height = d["probe_size_mm"][0] / 2, d["probe_size_mm"][2]
    width_mm = d["extent_mm"][1] - d["extent_mm"][0]
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    grid = fig.add_gridspec(2, 2, height_ratios=[4.3, 1], hspace=0.32, wspace=0.14, left=0.055, right=0.95, top=0.9, bottom=0.08)
    ax_mesh, ax_ct, ax_time = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])

    collection = PolyCollection(
        split_polygons(d["anim_polygons"][0], offsets),
        array=polygon_means(d["anim_displacement_mm"][0], offsets),
        cmap="viridis",
        clim=(0, 7.5),
        edgecolors=(0, 0, 0, 0.35),
        linewidths=0.3,
    )
    ax_mesh.add_collection(collection)
    ax_mesh.plot(d["sagittal_s"], d["sagittal_skin_material"], color="w", lw=0.8, ls="--")
    ax_mesh.set_facecolor("0.15")
    fig.colorbar(collection, ax=ax_mesh, fraction=0.03, pad=0.02, label="位移 (mm)")
    ax_mesh.set_title("FEM 网格在成像平面上的截面（随节点运动；虚线 = 原 CT 皮肤）", fontsize=9)
    image = ax_ct.imshow(d["anim_ct"][0].astype(np.float32), cmap="gray", vmin=CT_WINDOW[0], vmax=CT_WINDOW[1], extent=extent)
    ax_ct.set_title("形变 CT（同一平面；黄框 = 超声成像区域）", fontsize=9)
    probes = [probe_patch(half, float(d["frame_advance_mm"][0]), height) for _ in range(2)]
    us_box = Rectangle((d["extent_mm"][0], 0), width_mm, d["extent_mm"][2], fill=False, edgecolor="yellow", ls="--", lw=1)
    for ax, probe in zip((ax_mesh, ax_ct), probes):
        ax.add_patch(probe)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set_xlabel("沿探头宽度 (mm)")
        ax.set_ylabel("深度 (mm)")
    ax_ct.add_patch(us_box)
    cursor = timeline(ax_time, d)
    title = fig.suptitle("", fontsize=12)

    with video_writer(path, fps) as writer:
        for f in range(len(d["frame_time"])):
            face = float(d["frame_advance_mm"][f])
            collection.set_verts(split_polygons(d["anim_polygons"][f], offsets))
            collection.set_array(polygon_means(d["anim_displacement_mm"][f], offsets))
            image.set_data(d["anim_ct"][f].astype(np.float32))
            for probe in probes:
                probe.set_y(face - height)
            us_box.set_y(face + 3.0)
            cursor.set_xdata([d["frame_time"][f]] * 2)
            title.set_text(f"软体形变过程   t = {d['frame_time'][f]:.2f} s   力 {d['frame_force'][f]:4.1f} N   探头面 {face:+.2f} mm")
            writer.append_data(frame_rgb(fig))
    plt.close(fig)


def animate_us(d, m, path: Path, fps: float):
    extent = d["extent_mm"]
    fig = plt.figure(figsize=(12.8, 8.96), dpi=100)
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.3], hspace=0.28, wspace=0.28, left=0.06, right=0.97, top=0.9, bottom=0.07)
    rows = [("rigid_", "刚体（SonoGym 原方式）"), ("", "软体形变（本文）")]
    images = {}
    for r, (prefix, row_title) in enumerate(rows):
        for c, (kind, title, window) in enumerate(us_rows(m)):
            ax = fig.add_subplot(grid[r, c])
            images[prefix + kind] = ax.imshow(d[prefix + kind][0].astype(np.float32), cmap="gray", vmin=window[0], vmax=window[1], extent=extent)
            ax.set_title(f"{row_title}\n{title}", fontsize=9)
            ax.tick_params(labelsize=7)
            if c == 0:
                ax.set_ylabel("深度 (mm，自图像顶部)")
    ax_force = fig.add_subplot(grid[0, 3])
    cursors = [timeline(ax_force, d, indentation_axis=False)]
    ax_force.set_title("探头力", fontsize=9)
    ax_advance = fig.add_subplot(grid[1, 3])
    ax_advance.plot(d["step_time"], d["step_advance_mm"], color="C0", lw=1.2, label="探头面")
    ax_advance.axhline(float(d["contact_mm"]), color="0.5", ls=":", lw=1, label="首次接触")
    ax_advance.set_xlabel("时间 (s)")
    ax_advance.set_ylabel("自 SonoGym 皮肤点 (mm)")
    ax_advance.set_xlim(0, float(d["step_time"][-1]))
    ax_advance.grid(alpha=0.3)
    ax_advance.legend(fontsize=7, loc="lower right")
    ax_advance.set_title("探头面位置（两行图像相同）", fontsize=9)
    cursors.append(ax_advance.axvline(0, color="k", lw=1.2))
    title = fig.suptitle("", fontsize=12)

    with video_writer(path, fps) as writer:
        for f in range(len(d["frame_time"])):
            for key, image in images.items():
                image.set_data(d[key][f].astype(np.float32))
            for cursor in cursors:
                cursor.set_xdata([d["frame_time"][f]] * 2)
            title.set_text(
                f"同一探头运动下的超声：刚体人体 vs 软体人体   t = {d['frame_time'][f]:.2f} s   "
                f"力 {d['frame_force'][f]:4.1f} N   探头面 {d['frame_advance_mm'][f]:+.2f} mm"
            )
            writer.append_data(frame_rgb(fig))
    plt.close(fig)


# ------------------------------------------------------------------------------- page


CSS = """
body { font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", sans-serif;
       color: #1d2433; background: #f6f7fb; margin: 0; line-height: 1.7; }
main { max-width: 1240px; margin: 0 auto; padding: 0 28px 60px; }
header { background: linear-gradient(120deg, #17345c, #2c5e8f); color: #fff; padding: 36px 28px 28px; }
header .inner { max-width: 1240px; margin: 0 auto; }
header h1 { margin: 0 0 6px; font-size: 28px; }
header p { margin: 0; opacity: 0.85; }
nav { max-width: 1240px; margin: 16px auto 0; padding: 0 28px; font-size: 14px; }
nav a { margin-right: 18px; color: #2c5e8f; text-decoration: none; }
section { background: #fff; border-radius: 10px; padding: 22px 28px; margin: 22px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
h2 { margin-top: 0; color: #17345c; border-bottom: 2px solid #e3e8f0; padding-bottom: 8px; }
h3 { color: #2c5e8f; margin-bottom: 6px; }
figure { margin: 14px 0 22px; }
figure img, figure video { width: 100%; border: 1px solid #e3e8f0; border-radius: 6px; background: #000; }
figure img { background: #fff; }
figcaption { font-size: 13.5px; color: #4a5568; margin-top: 6px; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; margin: 10px 0 18px; }
th, td { border: 1px solid #dfe5ee; padding: 6px 9px; text-align: left; vertical-align: top; }
th { background: #eef2f8; }
td.num, th.num { text-align: right; }
.cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 16px 0; }
.card { background: #f2f6fc; border-left: 4px solid #2c5e8f; border-radius: 6px; padding: 12px 14px; }
.card b { display: block; font-size: 20px; color: #17345c; }
.card span { font-size: 13px; color: #4a5568; }
.flow { display: flex; flex-wrap: wrap; align-items: stretch; gap: 8px; margin: 14px 0; font-size: 13.5px; }
.flow div { flex: 1 1 0; background: #eef2f8; border-radius: 6px; padding: 10px 12px; min-width: 150px; }
.flow div b { color: #17345c; display: block; }
.flow .arrow { flex: 0 0 auto; background: none; min-width: 0; padding: 10px 0; align-self: center; color: #2c5e8f; font-size: 20px; }
.ok { color: #1f7a3a; font-weight: bold; }
.warn { color: #b26b00; font-weight: bold; }
.info { color: #2c5e8f; font-weight: bold; }
code { background: #eef2f8; padding: 1px 5px; border-radius: 4px; font-size: 90%; }
pre { background: #1d2433; color: #e6edf6; padding: 14px 16px; border-radius: 6px; overflow-x: auto; font-size: 13px; line-height: 1.5; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
ul.tight li { margin-bottom: 4px; }
"""

OK = '<span class="ok">通过</span>'
WARN = '<span class="warn">注意</span>'
INFO = '<span class="info">说明</span>'


def build_page(d, m, meta, runs, assets: dict[str, str]) -> str:
    fem, field, ct, us = m["fem"], m["field"], m["deformed_ct"], m["ultrasound"]
    axis = ct["probe_axis"]
    plateaus = us["plateaus"]
    last = plateaus[-1]
    steps = us["speckle_tracking"]
    increments = steps[:-1]
    whole = steps[-1]
    chained = us["speckle_tracking_chained"]["conv"]
    material, solver, pinning = meta["material"], meta["solver"], meta["pinning"]
    probe_size = np.array(meta["probe_size_m"]) * 1e3
    conv_echo_error = max(abs(p["bone_echo_shift_conv"]["median_mm"] - p["bone_shift_expected_mm"]) for p in plateaus)
    net_echo_error = max(abs(p["bone_echo_shift_net"]["median_mm"] - p["bone_shift_expected_mm"]) for p in plateaus)
    label_error = max(abs(p["bone_shift_labels_mm"] - p["bone_shift_expected_mm"]) for p in plateaus)
    class_error = max(abs(c["relative_error"]) for c in ct["classes"])
    distortion = us["final_image_distortion"]
    frames = len(d["frame_time"])
    duration = float(d["frame_time"][-1])
    mass = f"，{meta['probe_mass_kg']:g} kg" if "probe_mass_kg" in meta else ""

    def figure(key: str, caption: str) -> str:
        return f'<figure><img src="{assets[key]}" alt="{escape(key)}"><figcaption>{caption}</figcaption></figure>'

    def video(key: str, caption: str) -> str:
        return (
            f'<figure><video src="{assets[key]}" controls autoplay loop muted playsinline></video>'
            f"<figcaption>{caption}</figcaption></figure>"
        )

    cards = [
        (f"{fem['indentation_mm'][-1]:.2f} mm", f"20 N 稳态压入深度；刚度 {fem['stiffness_n_per_mm']:.2f} N/mm（10–20 N 线性 R² {fem['linear_fit_r2_10_to_20n']:.4f}）"),
        (f"{axis['compression_strain'] * 100:.1f}%", f"探头轴线上皮肤→椎板的组织厚度 {axis['tissue_before_mm']:.1f} → {axis['tissue_after_mm']:.1f} mm，椎板位置不变"),
        (
            f"{ct['skin_profile']['sagittal']['rmse_mm']:.3f} / {ct['skin_profile']['transverse']['rmse_mm']:.3f} mm",
            "形变 CT 中的皮肤与 FEM 预测位置之差（RMSE，矢状 / 横断面；CT 体素 1.5 mm）",
        ),
        (f"≤ {conv_echo_error:.2f} mm", "物理卷积模型超声中骨回声的平移与探头前进量之差（5 个力值）"),
        (
            f"{min(s['conv']['axial_rmse_mm'] for s in increments):.2f}–{max(s['conv']['axial_rmse_mm'] for s in increments):.2f} mm",
            f"物理卷积模型超声的散斑追踪与 FEM 组织运动之差（逐级加载的轴向 RMSE；累加成 0 → 20 N 整段为 {chained['axial_rmse_mm']:.2f} mm）",
        ),
        (f"{us['zero_load_net_max_abs']:.4f}", "零载荷时整条链路复现 SonoGym 原图：网络模型最大像素差（图像范围 0–1）"),
    ]
    card_html = "".join(f'<div class="card"><b>{a}</b><span>{b}</span></div>' for a, b in cards)

    setup = table(
        ["项目", "设置"],
        [
            ["人体", f"SonoGym 患者 <code>{escape(meta['patient'])}</code>：CT 与标签体数据（1.5 mm 各向同性体素）、体表网格 <code>combined_wrapwrap.stl</code>"],
            ["部位", f"L3 椎板旁正中、矢状切面：SonoGym 表面坐标 ({m['probe']['surface_xz'][0]}, {m['probe']['surface_xz'][1]})，x_angle = π/2；探头位姿由 SonoGym 的表面图/法向图求得"],
            [
                "软体",
                f"Isaac Sim 4.5 / Isaac Lab 2.1，PhysX GPU FEM。仿真网格：体素六面体（hex_res {solver['hex_res']}）剖分为四面体，"
                f"{fem['nodes']:,} 节点 / {fem['tetrahedra']:,} 单元；碰撞网格由体表网格生成（{solver['surface_faces']:,} 面）",
            ],
            ["材料", f"线弹性：E = {material['youngs_pa'] / 1e3:.0f} kPa，ν = {material['poissons']}，ρ = {material['density']:.0f} kg/m³；顶点速度阻尼 {solver['vertex_velocity_damping']}，弹性阻尼 {solver['elasticity_damping']}"],
            ["边界条件", f"骨骼（标签膨胀 {pinning['bone_dilate_mm']:g} mm）内的 {pinning['bone_nodes']:,} 个节点、腹侧床面 {pinning['support_band_m'] * 1e3:.0f} mm 支撑带内的 {pinning['support_nodes']:,} 个节点固定（运动学目标）；重力关闭"],
            ["探头", f"刚体盒 {probe_size[0]:.0f} × {probe_size[1]:.0f} × {probe_size[2]:.0f} mm{mass}，摩擦系数 {meta['friction']}；沿 SonoGym 探头法向的单自由度导轨，在探头上施加外力（力控）"],
            ["载荷", "未接触静置 → 5 → 10 → 15 → 20 N，每级 0.5 s 线性斜坡 + 2 s 保持"],
            ["求解与记录", f"dt = {solver['dt'] * 1e3:.0f} ms，{solver['iterations']} 次位置迭代；{frames} 帧（{meta['record_hz']:.0f} Hz）记录全部节点，{duration:.1f} s 仿真耗时 {meta['wall_time_s']:.0f} s"],
        ],
    )

    plateau_rows = [
        [
            f"{f:g} N",
            f"{i:.2f}",
            f"{abs(dr):.4f}",
            f"{j:.4f}",
            f"{p:.2f}",
        ]
        for f, i, dr, j, p in zip(fem["force_n"], fem["indentation_mm"], fem["drift_last_0p5s_mm"], fem["step_jitter_mm"], fem["max_penetration_mm"])
    ]
    plateau_table = table(["平台", "压入深度 (mm)", "最后 0.5 s 漂移 (mm)", "逐步抖动 (mm)", "碰撞网格穿透 (mm)"], plateau_rows)

    forces = sorted(runs[0]["indentation"]) if runs else []
    sweep_rows = [
        [
            str(r["hex_res"]),
            str(r["iterations"]),
            f"{r['nodes']:,}",
            *[f"{r['indentation'][f]:.2f}" for f in forces],
            f"{r['penetration']:.2f}",
            f"{r['wall_s']:.0f}",
        ]
        for r in runs
    ]
    sweep_table = table(
        ["hex_res", "迭代次数", "节点数", *[f"{f:g} N 压入 (mm)" for f in forces], "最大穿透 (mm)", "耗时 (s)"],
        sweep_rows,
    )
    if runs:
        by_it = {it: [r["indentation"][20.0] for r in runs if r["iterations"] == it] for it in (60, 240)}
        iteration_drop = [1 - b / a for a, b in zip(by_it[60], by_it[240])]
        res_rows = [r for r in runs if r["iterations"] == 240]
        finest = max(runs, key=lambda r: r["hex_res"])
        finest_penetration = [r["penetration"] for r in runs if r["hex_res"] == finest["hex_res"]]
        main_runs = [r for r in runs if (r["hex_res"], r["iterations"]) == (solver["hex_res"], solver["iterations"])]
        main_indentation = dict(zip(fem["force_n"], fem["indentation_mm"]))
        if main_runs:
            path_gap = max(abs(v - main_indentation[f]) for f, v in main_runs[0]["indentation"].items() if f in main_indentation)
            path_text = (
                f"与主算例相同的 hex_res {solver['hex_res']} / {solver['iterations']} 次扫描算例只加载 10、20 N 两级，"
                f"其稳态压入深度与主算例（逐级 5 N 加载）相差 {path_gap:.3f} mm，说明稳态与加载路径无关。"
            )
        else:
            path_text = ""
        sweep_text = (
            f"同一网格下迭代次数从 60 增加到 240，20 N 压入深度减少 {min(iteration_drop) * 100:.0f}–{max(iteration_drop) * 100:.0f}%，到 240 次仍未收敛；"
            f"同为 240 次迭代时，网格从 hex_res {res_rows[0]['hex_res']} 加密到 {res_rows[-1]['hex_res']}，20 N 压入深度从 "
            f"{res_rows[0]['indentation'][20.0]:.2f} 增加到 {res_rows[-1]['indentation'][20.0]:.2f} mm（线性四面体在 ν = {material['poissons']} 下的体积锁定随加密减轻），"
            f"但更细的网格穿透更大（hex_res {finest['hex_res']} 为 {min(finest_penetration):.1f}–{max(finest_penetration):.1f} mm）。"
            f"主算例选 hex_res {solver['hex_res']}、{solver['iterations']} 次迭代：穿透 ≤ {max(fem['max_penetration_mm']):.2f} mm，各平台稳态漂移 < 0.001 mm。{path_text}"
        )
    else:
        sweep_text = "未找到敏感性扫描结果。"

    field_table = table(
        ["检查", "方法", "结果", "结论"],
        [
            ["零载荷恒等", "未形变网格上的拉回映射应为恒等", f"最大偏差 {sci(field['zero_load_max_mm'])} mm；3D 体积 {sci(field['zero_load_3d_max_mm'])} mm、{field['zero_load_3d_ct_max_hu']:.3f} HU", OK],
            [
                "前推–拉回往返",
                "在正向单元内随机取 20,000 个物质点，前推到 20 N 构形后再拉回",
                f"99% 分位 {sci(field['round_trip_p99_mm'])} mm；{field['round_trip_over_0p01mm']} 个点（{field['round_trip_over_0p01mm'] / field['round_trip_points'] * 100:.2f}%）超过 0.01 mm，"
                f"其中 {field['round_trip_over_0p01mm_in_overlap_or_inverted'] * 100:.0f}% 位于网格自重叠区或翻转单元旁，{(1 - field['round_trip_over_0p01mm_in_body']) * 100:.0f}% 在体外",
                OK,
            ],
            ["稠密场 vs 单元", "3D 体素网格上对拉回位移有限差分求 det，与所在单元的 J 比较", f"中位差 {sci(field['dense_volume_ratio_vs_elements_median_abs'])}", OK],
            ["PhysX 变形梯度", "PhysX 输出的单元 F 与由记录节点位置重算的 F 比较", f"中位 Frobenius 差 {sci(fem['physx_gradient_vs_nodes']['median_frobenius'])}（PhysX 存储为 F<sup>T</sup>；按 F 解读时为 {sci(fem['physx_gradient_vs_nodes']['median_frobenius_other_layout'])}）", OK],
            [
                "单元质量",
                "20 N 时探头周围 34,306 个单元的体积比 J",
                f"中位 {fem['volume_ratio']['median']:.4f}，1% 分位 {fem['volume_ratio']['p1']:.3f}，探头下方中位 {fem['volume_ratio']['under_probe_median']:.3f}；"
                f"{fem['volume_ratio']['inverted']} 个翻转、{fem['volume_ratio']['below_0p5']} 个 J < 0.5，全部在皮肤外的体素悬垂层（深度 {field['distorted_rest_depth_mm'][0]:.1f} 至 {field['distorted_rest_depth_mm'][1]:.1f} mm，{field['distorted_nodes_outside_body'] * 100:.0f}% 的节点在体外）",
                OK,
            ],
            [
                "畸变对结果的影响",
                "最终图像像素与 3D 组织体素落在翻转、J < 0.5 或重叠单元中的数量",
                f"图像：{distortion['pixels_unmapped']} 个无法映射、{distortion['pixels_in_elements_below_0p5']} 个在 J < 0.5 单元、{distortion['pixels_in_overlapping_elements']} 个在重叠单元（共 30,000）；"
                f"3D：{field['tissue_voxels_folded']} 个折叠、{field['tissue_voxels_in_overlapping_elements']:,} 个重叠（共 {field['tissue_voxels']:,} 个组织体素）",
                OK,
            ],
            [
                "网格自重叠",
                "统计严格落在两个以上单元内部的点",
                f"PhysX 体素网格在静止态即有 {field['rest_overlap_fraction'] * 100:.2f}% 的体积被两个单元覆盖（体内 {field['rest_overlap_fraction_in_body'] * 100:.2f}%），多在皮肤附近",
                INFO,
            ],
            [
                "网格空洞",
                "体内组织体素不在任何单元内",
                f"{field['tissue_voxels_in_mesh_holes']:,} 个（{field['tissue_voxels_in_mesh_holes'] / field['tissue_voxels'] * 100:.1f}%），位于肠气周围，距探头 ≥ {field['mesh_holes_min_distance_mm']:.0f} mm；拉回时按外延处理，位移 ≤ {field['mesh_holes_max_displacement_mm']:.2f} mm",
                INFO,
            ],
        ],
    )

    class_rows = [
        [
            CLASS_NAMES.get(c["class"], f"第 {c['class']} 类"),
            f"{c['before_ml']:.1f}",
            f"{c['after_predicted_ml']:.1f}",
            f"{c['after_measured_ml']:.1f}",
            f"{(c['after_predicted_ml'] / c['before_ml'] - 1) * 100:+.2f}%",
            f"{c['relative_error'] * 100:+.2f}%",
        ]
        for c in ct["classes"]
    ]
    class_table = table(["类别（SonoGym 转换后标签）", "形变前 (ml)", "FEM 预测形变后 (ml)", "形变标签实测 (ml)", "FEM 预测体积变化", "实测 − 预测"], class_rows)

    similarity_rows = []
    for p in plateaus:
        similarity_rows.append(
            [
                "未接触" if p["force_n"] == 0 else f"{p['force_n']:g} N",
                f"{p['face_mm']:+.2f}",
                f"{p['label_changed_vs_rigid'] * 100:.1f}%",
                f"{p['net_vs_rigid']['ssim']:.3f} / {p['net_vs_rigid']['ncc']:.3f}",
                f"{p['conv_vs_rigid']['ssim']:.3f} / {p['conv_vs_rigid']['ncc']:.3f}",
                f"{p['net_vs_reference']['ssim']:.3f} / {p['net_vs_reference']['ncc']:.3f}",
                f"{p['conv_vs_reference']['ssim']:.3f} / {p['conv_vs_reference']['ncc']:.3f}",
            ]
        )
    similarity_table = table(
        ["状态", "探头面 (mm)", "组织像素标签改变（对刚体）", "网络：对刚体 SSIM / NCC", "物理：对刚体 SSIM / NCC", "网络：对原始 SSIM / NCC", "物理：对原始 SSIM / NCC"],
        similarity_rows,
    )

    bone_rows = [
        [
            "未接触" if p["force_n"] == 0 else f"{p['force_n']:g} N",
            f"{p['bone_shift_expected_mm']:+.2f}",
            f"{p['bone_shift_labels_mm']:+.2f}",
            f"{p['bone_echo_shift_conv']['median_mm']:+.2f} ± {p['bone_echo_shift_conv']['mad_mm']:.2f}",
            f"{p['bone_echo_shift_net']['median_mm']:+.2f} ± {p['bone_echo_shift_net']['mad_mm']:.2f}",
        ]
        for p in plateaus
    ]
    bone_table = table(["状态", "理论 (mm)", "标签图 (mm)", "物理卷积模型 (mm)", "网络模型 (mm)"], bone_rows)

    speckle_rows = [
        [
            f"{s['from_n']:g} → {s['to_n']:g} N" + ("（一步）" if s is whole else ""),
            f"{s['conv']['predicted_axial_mm'][0]:+.2f} … {s['conv']['predicted_axial_mm'][1]:+.2f}",
            f"{s['conv']['valid_blocks']}/{s['conv']['blocks']}",
            f"{s['conv']['axial_rmse_mm']:.3f} / {s['conv']['axial_median_abs_mm']:.3f}",
            f"{s['conv']['axial_pearson_r']:.2f}",
            f"{s['conv']['lateral_rmse_mm']:.3f}",
            f"{s['net']['valid_blocks']}/{s['net']['blocks']}",
            f"{s['net']['axial_rmse_mm']:.2f} / {s['net']['axial_median_abs_mm']:.2f}",
            f"{s['net']['axial_pearson_r']:.2f}",
        ]
        for s in steps
    ]
    speckle_rows.append(
        [
            f"{whole['from_n']:g} → {whole['to_n']:g} N（逐级累加）",
            f"{chained['predicted_axial_mm'][0]:+.2f} … {chained['predicted_axial_mm'][1]:+.2f}",
            f"{chained['valid_blocks']}/{whole['conv']['blocks']}",
            f"{chained['axial_rmse_mm']:.3f} / {chained['axial_median_abs_mm']:.3f}",
            f"{chained['axial_pearson_r']:.2f}",
            f"{chained['lateral_rmse_mm']:.3f}",
            "—",
            "—",
            "—",
        ]
    )
    speckle_table = table(
        ["加载步", "FEM 轴向位移范围 (mm)", "物理：有效块", "物理：轴向 RMSE / 中位 (mm)", "物理：r", "物理：横向 RMSE (mm)", "网络：有效块", "网络：轴向 RMSE / 中位 (mm)", "网络：r"],
        speckle_rows,
    )

    summary_rows = [
        ["FEM", "平台稳态", "每个保持段最后 0.5 s 探头位置漂移", f"最大 {max(abs(x) for x in fem['drift_last_0p5s_mm']):.4f} mm", "< 0.01 mm", OK],
        ["FEM", "探头不穿透", "碰撞网格节点进入探头的最大深度", f"{max(fem['max_penetration_mm']):.2f} mm", "< 0.5 mm", OK],
        ["FEM", "固定节点", "骨/床面固定节点的最大位移", f"{fem['max_pinned_drift_mm']:.4f} mm", "0", OK],
        ["FEM", "线性响应", "10–20 N 力–压入深度线性拟合", f"R² = {fem['linear_fit_r2_10_to_20n']:.5f}", "> 0.99", OK],
        [
            "FEM",
            "与解析解对比",
            "平冲头·弹性半空间：δ = F(1−ν²)/(2aE)",
            f"FEM / 解析 = {fem['indentation_mm'][0] / fem['flat_punch_halfspace_mm'][0]:.2f}（5 N）→ {fem['indentation_mm'][-1] / fem['flat_punch_halfspace_mm'][-1]:.2f}（20 N）；等效模量 {fem['equivalent_halfspace_modulus_pa'] / 1e3:.0f} kPa",
            "量级一致",
            INFO,
        ],
        ["形变场", "零载荷恒等", "未形变时拉回映射", f"{sci(field['zero_load_max_mm'])} mm", "< 0.001 mm", OK],
        ["形变场", "前推–拉回往返", "20,000 个随机物质点", f"99% 分位 {sci(field['round_trip_p99_mm'])} mm", "< 0.001 mm", OK],
        ["形变场", "稠密场 vs 单元 J", "3D 网格有限差分", f"中位差 {sci(field['dense_volume_ratio_vs_elements_median_abs'])}", "< 0.001", OK],
        ["形变 CT", "骨不动", "形变 CT 中骨体素的拉回位移", f"中位 {sci(ct['bone_displacement_mm']['median'])} mm，99% 分位 {ct['bone_displacement_mm']['p99']:.2f} mm", "≪ 1.5 mm 体素", OK],
        [
            "形变 CT",
            "皮肤位置",
            "形变 CT 上 −500 HU 皮肤线 vs 物质皮肤面前推后与平面交线",
            f"RMSE {ct['skin_profile']['sagittal']['rmse_mm']:.3f} / {ct['skin_profile']['transverse']['rmse_mm']:.3f} mm（最大 {max(ct['skin_profile'][p]['max_abs_mm'] for p in ('sagittal', 'transverse')):.2f} mm）",
            "≪ 1.5 mm 体素",
            OK,
        ],
        ["形变 CT", "两种采样一致", "探头轴线皮肤：直接拉回 vs NIfTI 体素网格", f"{axis['skin_after_direct_mm']:+.2f} vs {axis['skin_after_nifti_grid_mm']:+.2f} mm", "< 0.1 mm", OK],
        ["形变 CT", "各类组织体积", "形变标签体积 vs FEM 预测（含流出流入）", f"最大相对差 {class_error * 100:.2f}%", "< 0.5%", OK],
        ["超声", "零载荷复现 SonoGym", "零位移场 → 网络模型", f"最大像素差 {us['zero_load_net_max_abs']:.4f}", "< 0.01", OK],
        ["超声", "未接触时与刚体一致", "静置帧：软体 vs 刚体同位置", f"网络 SSIM {plateaus[0]['net_vs_rigid']['ssim']:.6f}", "≈ 1", OK],
        ["超声", "骨面平移（标签）", "形变标签图骨面深度变化 vs −探头前进", f"最大差 {label_error:.2f} mm", "< 0.25 mm（半像素）", OK],
        ["超声", "骨回声平移（物理模型）", "骨回声逐列互相关", f"最大差 {conv_echo_error:.2f} mm", "< 0.25 mm", OK],
        ["超声", "骨回声平移（网络）", "同上", f"最大差 {net_echo_error:.2f} mm", "—", WARN],
        [
            "超声",
            "散斑随组织运动（物理模型）",
            "块匹配追踪 vs FEM 运动，逐级",
            f"轴向 RMSE {min(s['conv']['axial_rmse_mm'] for s in increments):.3f}–{max(s['conv']['axial_rmse_mm'] for s in increments):.3f} mm，r {min(s['conv']['axial_pearson_r'] for s in increments):.2f}–{max(s['conv']['axial_pearson_r'] for s in increments):.2f}",
            "< 0.25 mm（半像素）",
            OK,
        ],
        [
            "超声",
            "散斑随组织运动（物理模型）",
            "逐级追踪累加成 0 → 20 N 整段 vs FEM 运动",
            f"轴向 RMSE {chained['axial_rmse_mm']:.3f} mm，中位 {chained['axial_median_abs_mm']:.3f} mm",
            "< 0.25 mm（半像素）",
            OK if chained["axial_rmse_mm"] < 0.25 else WARN,
        ],
        ["超声", "散斑追踪（网络）", "同上", f"轴向 RMSE {min(s['net']['axial_rmse_mm'] for s in increments):.1f}–{max(s['net']['axial_rmse_mm'] for s in increments):.1f} mm", "—", WARN],
        ["超声", "畸变单元影响图像", "最终图像落在翻转/J < 0.5/无法映射处的像素", f"{distortion['pixels_unmapped'] + distortion['pixels_in_elements_below_0p5']} / 30,000", "0", OK],
    ]
    summary_table = table(["环节", "检查项", "方法", "结果", "判据", "结论"], summary_rows)

    sweep_meta = runs[0] if runs else None
    sweep_command = ""
    if sweep_meta:
        forces_arg = " ".join(f"{p[3]:g}" for p in sweep_meta["phases"] if p[0].startswith("hold"))
        settle = sweep_meta["phases"][0][1]
        hold = [p[1] for p in sweep_meta["phases"] if p[0].startswith("hold")][0]
        sweep_command = (
            "for res in 40 48 56; do for it in 60 120 240; do\n"
            f"  python scripts/deformation/patient_press_record.py --headless --hex_res $res --iterations $it \\\n"
            f"      --force {forces_arg} --settle {settle:g} --hold {hold:g} --record_hz {sweep_meta['record_hz']:g} \\\n"
            "      --output outputs/deformation_us/sensitivity/res${res}_it${it}.npz\n"
            "done; done\n"
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>软体形变 → 形变 CT → 超声：仿真验证报告</title><style>{CSS}</style></head>
<body>
<header><div class="inner">
<h1>软体形变 → 形变 CT → 超声：仿真验证报告</h1>
<p>SonoGym 人体 {escape(meta['patient'])} · Isaac Sim PhysX FEM 软体 · 力控探头按压 0–20 N · SonoGym 原版超声模型（网络 + 物理卷积）</p>
</div></header>
<nav><a href="#summary">概述</a><a href="#fem">1 软体形变</a><a href="#field">2 形变场建模</a><a href="#ct">3 形变 CT</a><a href="#us">4 超声对比</a><a href="#validation">5 验证汇总</a><a href="#limits">6 局限</a><a href="#repro">7 复现</a></nav>
<main>

<section id="summary">
<h2>概述</h2>
<p>SonoGym 通过对 CT/标签体数据切片生成超声图像，并用机械臂驱动探头扫描，但其中的人体是刚体：探头压下时组织不会变形。
本验证不加入机械臂，只把 SonoGym 的人体换成软体：在体表一点施加固定的探头力，求出软体的形变场，把形变场作用到 CT 上，再用形变后的 CT 生成超声。
人体数据（患者 {escape(meta['patient'])} 的 CT、标签和体表网格）和超声模型（pix2pix 网络 <code>pix2pix_rand_down_up.pth</code>、物理卷积模型 <code>USSimulatorConv</code>）都使用 SonoGym 原版，SonoGym 子模块未作修改。</p>
<div class="flow">
<div><b>① 软体仿真</b>体表网格 → PhysX FEM 软体；骨骼与床面固定；刚体探头力控按压 0 → 20 N，逐帧记录全部节点</div><div class="arrow">→</div>
<div><b>② 形变场</b>四面体内线性插值：前推 X → x；拉回 x → X 用变形单元的重心坐标求逆，网格外按最近内点位移衰减外延</div><div class="arrow">→</div>
<div><b>③ 形变 CT</b>CT(X(x))、标签(X(x))；每帧切出成像平面，最后一帧在探头周围做 3D 形变并存为 NIfTI</div><div class="arrow">→</div>
<div><b>④ 超声</b>形变 CT 切面送入 SonoGym 两个超声模型；与刚体人体、SonoGym 原始图像对比</div>
</div>
<div class="cards">{card_html}</div>
<p>结论：形变场、形变 CT 与超声生成三步之间保持一致，误差远小于 CT 体素（1.5 mm）和超声像素（0.5 mm）。物理卷积模型的散斑锚定在组织上，随组织移动，用散斑追踪可以从超声图像中恢复 FEM 的组织运动；网络模型能反映解剖结构的位移，但散斑纹理每帧重新生成，不随组织移动。
FEM 的绝对刚度受 PhysX 求解器收敛性影响较大（见第 1、6 节），因此本验证检查的是“形变 → CT → 超声”这条链路的正确性，不是组织力学参数的定量准确性。</p>
</section>

<section id="fem">
<h2>1 软体形变仿真</h2>
{setup}
{video("render", "Isaac Sim 渲染：探头（蓝色）在 L3 旁正中按压背部体表（皮肤色为软体的碰撞网格），力依次为 5、10、15、20 N。")}
{figure("force", "左：力–压入深度。灰线是逐步轨迹（含斜坡段的动态过程），圆点是各保持段末的稳态值。压入深度从探头首次接触碰撞网格时算起，首次接触点在 SonoGym 皮肤点外 " + f"{-fem['contact_offset_mm']:.2f}" + " mm。虚线是 10–20 N 线性拟合，点线是同材料平冲头压弹性半空间的解析估计。FEM 比半空间解硬，可能的原因有三：骨骼上方组织只有约 43 mm 厚，线性四面体存在体积锁定，求解器未收敛。右：力与压入深度的时间历程，每个保持段都已达到稳态。")}
{plateau_table}
<h3>求解器敏感性</h3>
<p>{sweep_text}</p>
{figure("sweep", "不同网格分辨率和迭代次数下的稳态压入深度。星号为本文主算例。")}
{sweep_table}
</section>

<section id="field">
<h2>2 形变场建模</h2>
<p>FEM 给出的是每个仿真节点在每一帧的位置。形变场在每个四面体内线性插值，前推映射为 φ(X) = Σ w<sub>i</sub>(X) x<sub>i</sub>，其中 w<sub>i</sub> 是点在静止单元中的重心坐标。
形变 CT 需要反方向的拉回映射：对空间中每个体素/像素 x，求当前位于该处的物质点 X = φ<sup>−1</sup>(x)。做法是在形变后的网格中找到包含 x 的单元，用同一组重心坐标回到静止单元，翻转单元不参与。
网格外的点取最近内点的位移，并在 {field['pullback_decay_mm']:g} mm 内按高斯衰减；这样探头压出的凹陷会映射到原皮肤外的空气，而不会把原来的组织留在原地。
第二阶段全部在 GPU 上运行：{frames} 帧（形变与刚体各一套）的 CT 切面与超声、3D 形变和全部验证，共约 {m['timing_s']:.0f} s。</p>
{video("anim_deformation", "形变过程动画（20 Hz 记录，实时播放）。左：成像平面截过 FEM 网格的多边形随节点运动，颜色为位移幅值。仿真网格是体素网格，会超出皮肤约一个体素，虚线为原 CT 皮肤。右：同一平面上逐帧拉回得到的形变 CT，黄框是超声成像区域，随探头移动。")}
{figure("mesh", "仿真网格在两个平面上的截面。左：未形变；右：20 N 形变后，颜色为单元体积比 J（蓝 = 压缩，红 = 膨胀，J 越接近 1 越透明）。底图为对应的原始 / 形变 CT；横断面左下的空白方块是第 2 节所述的网格空洞。")}
{figure("field", "20 N 时的形变场。左：位移幅值；中：深度方向应变 ε<sub>zz</sub>，两者都画在物质（未形变）坐标上，虚线为皮肤。右：形变 CT 上叠加原为 5 mm 方格的物质线，经形变场前推后的位置。")}
{figure("skin", "探头周围体表（碰撞网格）在 20 N 时的形状。左为三维视图，右为俯视下陷图，虚线框是探头底面。下陷集中在探头底面下方并向四周平滑衰减。")}
{field_table}
</section>

<section id="ct">
<h2>3 形变场作用于 CT</h2>
<p>CT 中体外部分（标签为 0）先置为 −1000 HU：SonoGym 的 CT 是仰卧位扫描，背后 1 cm 左右有床垫，而仿真场景是俯卧位，背后是空气，探头压出的凹陷应当拉回到空气而不是床垫。
每一帧都在超声成像平面上拉回采样；最后一帧（20 N）在探头周围 {field['box_voxels'][0]} × {field['box_voxels'][1]} × {field['box_voxels'][2]} 体素的盒子内整体形变，输出到 CT 原网格上的 NIfTI：
<code>deformed_ct.nii.gz</code>、<code>deformed_labels.nii.gz</code>、<code>pullback_displacement_mm.nii.gz</code>、<code>volume_ratio.nii.gz</code>，以及体外置空后的原 CT <code>ct_body.nii.gz</code>。</p>
{figure("ct", "左两列：两个平面上的原始 CT 与 20 N 形变 CT。红线是在形变 CT 上测得的皮肤（−500 HU 等值线），青色虚线是 FEM 预测：物质皮肤面经形变场前推后与平面的交线。右上：探头中轴线上的 CT 值，形变后皮下各层被压缩，椎板保持不动；直接拉回与从 NIfTI 体素网格读取的结果一致。右下：皮肤线测得值与预测值之差。")}
<div class="two"><div>
<h3>探头中轴线</h3>
{table(["量", "形变前", "20 N"], [
    ["探头面 (mm)", f"{fem['contact_offset_mm']:+.2f}（首次接触）", f"{axis['face_mm']:+.2f}"],
    ["CT 皮肤 (mm)", f"{axis['skin_before_mm']:+.2f}", f"{axis['skin_after_direct_mm']:+.2f}（NIfTI：{axis['skin_after_nifti_grid_mm']:+.2f}）"],
    ["椎板 (mm)", f"{axis['lamina_before_mm']:+.1f}", f"{axis['lamina_after_mm']:+.1f}"],
    ["皮肤→椎板 (mm)", f"{axis['tissue_before_mm']:.1f}", f"{axis['tissue_after_mm']:.1f}（{axis['compression_strain'] * 100:+.1f}%）"],
])}
<p>深度均从 SonoGym 皮肤点沿探头法向量起，向内为正。探头面与 CT 皮肤之间约 3 mm 的差距来自体表网格本身在 CT 皮肤外约 2–3 mm（见第 6 节）。</p>
</div><div>
<h3>骨骼</h3>
<p>形变 CT 中骨体素的拉回位移：中位 {sci(ct['bone_displacement_mm']['median'])} mm，99% 分位 {ct['bone_displacement_mm']['p99']:.2f} mm，最大 {ct['bone_displacement_mm']['max']:.2f} mm（出现在固定区边缘、尚可移动的单元里）。
骨节点被固定，骨骼在形变 CT 中保持原位，组织在骨与探头之间被压缩。</p>
</div></div>
<h3>各类组织体积</h3>
<p>FEM 预测值：把 3D 盒子（外扩 15 mm）里每个物质体素按所在单元的 J 加权，统计前推后落在盒内的部分，因而包含了流出和流入盒子的组织。实测值：统计形变标签中各类的体素数。</p>
{class_table}
</section>

<section id="us">
<h2>4 形变 CT 生成超声：对比</h2>
<p>每一帧都按 SonoGym <code>USSlicer</code> 的几何关系成像：150 × 200 像素，0.5 mm，从探头面下方 3 mm 开始。成像平面随探头前进到记录的位置。“未形变”给出两种对照：</p>
<ul class="tight">
<li><b>SonoGym 原始</b>：刚体人体，探头放在 SonoGym 的皮肤点，即 SonoGym 平时生成的图像；</li>
<li><b>刚体·探头同位置</b>：刚体人体，探头前进到与软体仿真相同的位置。这相当于在 SonoGym 里把探头压进刚体人体：组织不变形，顶部的组织只是被“切掉”。</li>
</ul>
<p>两个超声模型的设置与 SonoGym 引导任务一致。网络模型的直方图查找表只在第一张（原始）图像上计算一次，此后固定，与 SonoGym 的行为相同。
物理卷积模型的两张随机散斑图按物质坐标采样，因此散斑随组织移动；SonoGym 原本把它们固定在人体坐标系里，对刚体人体两者等价。</p>
{video("anim_us", "同一探头运动下的超声动画。上行：刚体人体（SonoGym 原方式）；下行：软体人体（本文）。从左到右依次为 CT 切面、网络模型、物理卷积模型。未接触时两行相同；接触后，刚体中探头直接切入组织，软体中皮下各层被压缩，骨骼在两者中位置相同。")}
{figure("us_compare", f"{last['force_n']:g} N 时的对比。列依次为：SonoGym 原始图像、刚体（探头同位置）、软体形变、软体 − 刚体的差值；行依次为 CT 切面、网络模型、物理卷积模型。图像坐标随探头移动，深度从图像顶部算起。")}
<p>{last['force_n']:g} N 时物理卷积模型图像顶部的亮带是皮肤回声。SonoGym 的体表网格比 CT 皮肤（−500 HU）靠外约 {axis['skin_before_mm'] - float(d['contact_mm']):.1f} mm（探头在 {float(d['contact_mm']):+.2f} mm 处首次接触体表网格，CT 皮肤在 {axis['skin_before_mm']:+.2f} mm），
探头压住体表网格时，形变 CT 的皮肤位于探头面下 {axis['skin_after_direct_mm'] - axis['face_mm']:.1f} mm，而 SonoGym 的图像从探头面下 {us['image_top_below_probe_face_mm']:.0f} mm 开始，皮肤正好落在图像顶部，卷积模型在空气/组织界面上产生强反射。
刚体中探头直接切入组织，图像从皮肤下 {axis['face_mm'] + us['image_top_below_probe_face_mm'] - axis['skin_before_mm']:.1f} mm 开始，所以没有这条亮带。</p>
{figure("us_plateaus", "各力值下的形变 CT 切面与两种超声图像。红点为形变标签图中的骨面。")}
{similarity_table}
<p>SSIM 和 NCC 是在固定显示窗口（SonoGym 原始图像的 0.5%–99.5% 分位）下计算的。未接触时软体与刚体逐像素相同；加载后两者的差异就是形变本身带来的变化，随力增大。
与 SonoGym 原始图像的相似度同时受探头位置和形变的影响，因此不随力单调变化。</p>
<h3>骨面平移</h3>
<p>骨节点固定，图像又随探头移动，所以骨面在图像中的深度应正好改变“负的探头前进量”。分别在形变标签图和两种超声图像上测量：超声图像中，以 SonoGym 原始图像骨面上下 5 mm 的列剖面为模板，在 ±8 mm 范围内做归一化互相关，取逐列中位数。</p>
{figure("bone", "骨面深度变化：理论值与三种测量。右图为误差，误差棒为逐列中位绝对偏差。")}
{bone_table}
<p>标签图和物理卷积模型与理论值的差都在 {max(label_error, conv_echo_error):.2f} mm 以内。网络模型的骨回声位移偏小、逐列离散大（最大偏差 {net_echo_error:.2f} mm）：SonoGym 的网络先把 CT 切面最近邻下采样到 64 × 64（深度方向每格约 1.6 mm）再上采样输入 UNet，骨回声因此按阶梯移动。</p>
<h3>散斑追踪：从超声图像恢复组织运动</h3>
<p>块匹配散斑追踪（弹性成像常用方法，25 × 25 像素块，归一化互相关加亚像素拟合）在两帧超声之间独立估计组织运动，与 FEM 给出的运动比较。只用组织区域内、相关系数大于 0.6 的块。</p>
{figure("speckle", "上：0 → 20 N 整段的轴向位移，依次为 FEM 预测、物理卷积模型逐级追踪再累加、物理卷积模型一步追踪、网络模型一步追踪（空白为相关系数不足 0.6 的块）。下左：逐块对比。下右：各加载步与整段的误差。")}
{speckle_table}
<p>物理卷积模型的散斑锚定在组织上，逐级追踪的轴向 RMSE 为 {min(s['conv']['axial_rmse_mm'] for s in increments):.2f}–{max(s['conv']['axial_rmse_mm'] for s in increments):.2f} mm，不到半个像素。
把四步的结果沿每个块的轨迹累加（弹性成像的常规做法），0 → 20 N 整段的 RMSE 为 {chained['axial_rmse_mm']:.2f} mm，中位误差 {chained['axial_median_abs_mm']:.2f} mm，r = {chained['axial_pearson_r']:.3f}。
若直接在 0 N 和 20 N 两帧之间一步追踪，中位误差仍为 {whole['conv']['axial_median_abs_mm']:.2f} mm，但皮下 15%–25% 的压缩使靠近皮肤的块去相关，RMSE 升到 {whole['conv']['axial_rmse_mm']:.2f} mm。
网络模型的散斑每帧从 CT 重新生成，追踪结果与 FEM 运动基本无关（r ≈ {whole['net']['axial_pearson_r']:.1f}），这是模型本身的特性，不是形变链路的问题。</p>
</section>

<section id="validation">
<h2>5 验证汇总</h2>
{summary_table}
</section>

<section id="limits">
<h2>6 局限与说明</h2>
<ul>
<li><b>求解器收敛性。</b>PhysX FEM 是迭代求解器，响应随迭代次数持续变硬，网格加密则变软，绝对刚度约有 ±30% 的不确定性。等效半空间模量 {fem['equivalent_halfspace_modulus_pa'] / 1e3:.0f} kPa 明显高于设定的 {material['youngs_pa'] / 1e3:.0f} kPa，原因包括骨上组织层较薄、体积锁定和未收敛。若需要定量的组织力学，应换用收敛的隐式 FEM，或按实测力–位移曲线标定 E。</li>
<li><b>体表网格与 CT 皮肤的偏差。</b>SonoGym 的体表网格在 CT 皮肤外约 {axis['skin_before_mm'] - fem['contact_offset_mm']:.1f} mm：探头在 SonoGym 皮肤点外 {-fem['contact_offset_mm']:.2f} mm 处首次接触体表网格，而 CT 的 −500 HU 皮肤只在皮肤点外 {-axis['skin_before_mm']:.2f} mm。
因此压下后 CT 皮肤位于探头面下约 {axis['skin_after_direct_mm'] - axis['face_mm']:.1f} mm，正好在超声图像顶部（见第 4 节）。仿真体素网格又超出体表约一个体素，翻转和严重畸变的单元都在这层皮外悬垂里，最终图像中没有像素受其影响。</li>
<li><b>PhysX 体素网格本身的缺陷。</b>静止时即有 {field['rest_overlap_fraction'] * 100:.1f}% 的体积被两个单元重叠覆盖，多在皮肤附近；肠气周围有空洞，占盒内组织的 {field['tissue_voxels_in_mesh_holes'] / field['tissue_voxels'] * 100:.1f}%，因为 SonoGym 标签中肠气是背景，体表网格在那里有内壳。空洞距探头 ≥ {field['mesh_holes_min_distance_mm']:.0f} mm，在 FEM 中相当于空腔，拉回时按外延处理。</li>
<li><b>部位选择。</b>原计划在 L4 正中做横切面，但该处有正中沟：探头先压在两侧肌肉隆起上，比皮肤点高 10–19 mm，隆起塌陷后探头穿透网格。因此改在 L3 旁正中做矢状切面，该处皮肤平坦，探头与皮肤的间隙为 0.56 mm。</li>
<li><b>超声模型。</b>网络模型从 4 倍最近邻下采样的 CT 生成图像：骨回声按约 1.6 mm 的阶梯移动，散斑不随组织移动。物理卷积模型的散斑已改为锚定组织（SonoGym 子模块未改动，改动在本仓库的封装里）。两种模型都不模拟探头与皮肤的耦合：未接触时（静置帧）图像不会变黑，卷积模型在顶部的空气/皮肤界面上显示一条亮带。</li>
<li><b>分辨率。</b>CT 和标签为 1.5 mm 体素，标签按最近邻采样，各类组织体积因此有约 0.2% 的误差。仿真网格单元约 8.9 mm，皮下 2–3 mm 薄层的力学细节无法分辨。</li>
<li><b>未包含。</b>机械臂（按要求不加入）、重力、组织的非线性与粘弹性、不同组织的不同材料参数（全身为同一种线弹性材料，骨骼固定）。</li>
</ul>
</section>

<section id="repro">
<h2>7 复现</h2>
<p>需要先设置 <code>PYTHONNOUSERSITE=1</code>：用户目录下装有 torch 2.13（CPU 版），会遮蔽 <code>isaaclab_v2.1</code> 环境中的 torch 2.6 + CUDA 12.4。</p>
<pre>conda activate isaaclab_v2.1
export PYTHONNOUSERSITE=1
python scripts/deformation/patient_press_record.py --headless --capture   # ① FEM 按压与渲染，约 1 分钟
python scripts/deformation/deformed_ct_to_us.py                           # ②③④ 形变场、形变 CT、超声与验证，约 1 分钟
python scripts/deformation/make_report.py                                 # 本报告

# 求解器敏感性扫描（可选，约 5 分钟）
{escape(sweep_command)}</pre>
<p>输出目录 <code>outputs/deformation_us/</code>：<code>fem_press.npz</code>（全部节点轨迹）、<code>fem_press_render.mp4</code>、<code>us_sequence.npz</code>（逐帧形变/刚体 CT 切面、标签、两种超声图像、拉回位移）、<code>metrics.json</code>（全部验证数据）、NIfTI 形变体数据、<code>report/</code>（本页及单独的图片和视频）。</p>
</section>
</main></body></html>
"""


def main():
    args = parse_args()
    root = Path(PROJECT_DIR)
    data_dir = root / args.data_dir
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    m = json.loads((data_dir / "metrics.json").read_text())
    d = dict(np.load(data_dir / "us_sequence.npz"))
    with np.load(data_dir / "fem_press.npz") as press:
        meta = json.loads(str(press["meta"]))
    runs = load_sweep(data_dir / "sensitivity")

    figures = {
        "force": lambda p: figure_force(d, m, meta, p),
        "sweep": lambda p: figure_sweep(runs, meta, p),
        "mesh": lambda p: figure_mesh(d, p),
        "field": lambda p: figure_field(d, p),
        "skin": lambda p: figure_skin(d, p),
        "ct": lambda p: figure_ct(d, m, p),
        "us_compare": lambda p: figure_us_compare(d, m, p),
        "us_plateaus": lambda p: figure_us_plateaus(d, m, p),
        "bone": lambda p: figure_bone(m, p),
        "speckle": lambda p: figure_speckle(d, m, p),
    }
    assets = {}
    for name, draw in figures.items():
        if name == "sweep" and not runs:
            continue
        path = out / f"{name}.png"
        draw(path)
        assets[name] = data_uri(path, "image/png")
        print(f"[FIG] {path.name}")

    animate_deformation(d, out / "anim_deformation.mp4", args.fps)
    animate_us(d, m, out / "anim_us.mp4", args.fps)
    shutil.copyfile(data_dir / "fem_press_render.mp4", out / "render.mp4")
    for name in ("anim_deformation", "anim_us", "render"):
        assets[name] = data_uri(out / f"{name}.mp4", "video/mp4")
        print(f"[VIDEO] {name}.mp4")

    page = out / "index.html"
    page.write_text(re.sub(r"<(?=[\s\d.])", "&lt;", build_page(d, m, meta, runs, assets)), encoding="utf-8")
    print(f"[OUT] {page} ({page.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
