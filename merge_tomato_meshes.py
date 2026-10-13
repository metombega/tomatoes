"""
Register the 5 segmented tomato/cone meshes in
rgbd_photos/five_meshes_segmented/ into one merged tomato mesh.

Mesh counterpart of merge_tomato_clouds.py -- identical anchor logic (fit
a 3D circle through the 5 cone peaks, recover the turntable's rotation
axis and each shot's actual angle, rotate every shot into a common frame)
but operating on *_tomato.ply / *_cone.ply mesh pairs (see
segment_tomato_cone_mesh.py) instead of point clouds. The 5 aligned
tomato meshes are then simply concatenated (each is already a solid
triangulated surface, so there's no point-cloud-style downsample/outlier
cleanup step here -- that only makes sense for loose points).

Usage:
    python merge_tomato_meshes.py [input_folder] [output_folder]

Defaults convert rgbd_photos/five_meshes_segmented/*_tomato.ply (+ matching
*_cone.ply) into rgbd_photos/five_meshes_merged/tomato_meshes_merged.ply,
plus a diagnostic render of the fitted circle and the before/after
alignment.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

INPUT_FOLDER = "rgbd_photos/five_meshes_segmented"
OUTPUT_FOLDER = "rgbd_photos/five_meshes_merged"

SHOT_COLORS = [[0.85, 0.2, 0.2], [0.2, 0.65, 0.2], [0.2, 0.4, 0.9], [0.9, 0.6, 0.0], [0.6, 0.2, 0.8]]


def cone_peak(cone_mesh):
    pts = np.asarray(cone_mesh.vertices)
    center = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - center)
    axis = vt[0]
    proj = (pts - center) @ axis

    lo_mask = proj < np.percentile(proj, 10)
    hi_mask = proj > np.percentile(proj, 90)

    def local_spread(mask):
        sel = pts[mask]
        return np.linalg.norm(sel - sel.mean(axis=0), axis=1).std()

    tip_idx = np.argmin(proj) if local_spread(lo_mask) < local_spread(hi_mask) else np.argmax(proj)
    return pts[tip_idx]


def fit_circle_3d(points):
    """Best-fit circle through 3D points: fit a plane (SVD), project onto
    it, fit a 2D circle (algebraic least squares) in that plane. Returns
    the circle's center (3D), radius, plane normal, and an in-plane
    (e1, e2) basis used to measure angles around the circle."""
    centroid = points.mean(axis=0)
    centered = points - centroid

    _, _, vt = np.linalg.svd(centered)
    normal = vt[2]

    seed = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(normal, seed)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)

    x, y = centered @ e1, centered @ e2
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    radius = float(np.sqrt(sol[2] + cx ** 2 + cy ** 2))
    center_3d = centroid + cx * e1 + cy * e2

    return center_3d, radius, normal, e1, e2


def angle_around_circle(point, center, e1, e2):
    local = point - center
    return np.arctan2(local @ e2, local @ e1)


def rotate_about_axis(mesh, axis, center, angle_rad):
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle_rad)
    return mesh.rotate(R, center=center)


def save_overlay(clouds_points, filepath, title):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    for i, pts in enumerate(clouds_points):
        step = max(1, len(pts) // 3000)
        ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2],
                    c=[SHOT_COLORS[i % len(SHOT_COLORS)]], s=1.0, label=f"shot {i + 1}")
    ax.set_title(title)
    ax.legend(markerscale=8)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}")


def save_circle_diagnostic(peaks, center, radius, normal, e1, e2, filepath):
    theta = np.linspace(0, 2 * np.pi, 200)
    circle_pts = center + radius * (np.outer(np.cos(theta), e1) + np.outer(np.sin(theta), e2))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(circle_pts[:, 0], circle_pts[:, 1], circle_pts[:, 2], c="black", lw=1, label="fitted circle")
    for i, p in enumerate(peaks):
        ax.scatter(*p, c=[SHOT_COLORS[i % len(SHOT_COLORS)]], s=80, marker="*", label=f"cone peak {i + 1}")
    ax.scatter(*center, c="black", s=40, marker="x", label="circle center")
    ax.set_title(f"Cone peaks and fitted turntable circle (radius {radius * 1000:.1f} mm)")
    ax.legend(markerscale=1, fontsize=7)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}")


def main():
    input_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(INPUT_FOLDER)
    output_folder = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(OUTPUT_FOLDER)
    output_folder.mkdir(parents=True, exist_ok=True)

    stems = sorted({p.name[: -len("_tomato.ply")] for p in input_folder.glob("*_tomato.ply")})
    if len(stems) < 3:
        raise SystemExit(f"Expected at least 3 *_tomato.ply/_cone.ply pairs in {input_folder}, found {len(stems)}")

    shots = []
    for stem in stems:
        tomato = o3d.io.read_triangle_mesh(str(input_folder / f"{stem}_tomato.ply"))
        cone = o3d.io.read_triangle_mesh(str(input_folder / f"{stem}_cone.ply"))
        peak = cone_peak(cone)
        shots.append({"stem": stem, "tomato": tomato, "peak": peak})
        print(f"{stem}: {len(tomato.vertices)}v/{len(tomato.triangles)}t tomato, "
              f"{len(cone.vertices)}v/{len(cone.triangles)}t cone, peak {peak.round(4)}")

    peaks = np.array([s["peak"] for s in shots])
    center, radius, normal, e1, e2 = fit_circle_3d(peaks)

    plane_residual = (peaks - center) @ normal
    radial_residual = np.linalg.norm(
        np.stack([(peaks - center) @ e1, (peaks - center) @ e2], axis=1), axis=1
    ) - radius
    print(f"\nFitted circle: center {center.round(4)}, radius {radius * 1000:.2f} mm, axis {normal.round(4)}")
    print(f"  out-of-plane residuals (mm): {(plane_residual * 1000).round(3)}")
    print(f"  radial residuals (mm): {(radial_residual * 1000).round(3)}")

    save_circle_diagnostic(peaks, center, radius, normal, e1, e2, output_folder / "cone_peak_circle.png")

    raw_points = [np.asarray(s["tomato"].vertices) for s in shots]
    save_overlay(raw_points, output_folder / "before_registration.png", "Raw (no alignment)")

    angles = np.array([angle_around_circle(p, center, e1, e2) for p in peaks])
    ref_idx = 0
    ref_angle = angles[ref_idx]
    print(f"\nReference shot: {shots[ref_idx]['stem']}")

    aligned_tomato_points = []
    for s, angle in zip(shots, angles):
        delta = ref_angle - angle
        rotate_about_axis(s["tomato"], normal, center, delta)
        aligned_tomato_points.append(np.asarray(s["tomato"].vertices))
        print(f"{s['stem']}: rotated {np.degrees(delta):+.2f} deg about the turntable axis")

    save_overlay(aligned_tomato_points, output_folder / "after_registration.png",
                 "After circle-based turntable-angle alignment")

    merged = o3d.geometry.TriangleMesh()
    for s in shots:
        merged += s["tomato"]

    out_path = output_folder / "tomato_meshes_merged.ply"
    o3d.io.write_triangle_mesh(str(out_path), merged)
    print(f"\nMerged mesh: {len(merged.vertices)} vertices, {len(merged.triangles)} triangles -> {out_path}")


if __name__ == "__main__":
    main()
