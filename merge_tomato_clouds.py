"""
Register the 5 segmented tomato/cone point clouds in
rgbd_photos/five_shots_segmented/ into one merged tomato point cloud.

Each *_tomato.ply / *_cone.ply pair (see segment_tomato_cone.py) comes from
its own independent scan session, in its own coordinate frame. The rig is
a turntable: the tomato sits close to the rotation axis (so it barely
moves between shots), while the small violet cone marker sits farther out
and sweeps around that same axis as the turntable is stepped between the
5 shots. That means the 5 cone peaks all lie on a single circle -- the
turntable's own circle of rotation -- which is exactly the "center of the
ball" analogue from the old pipeline, generalized to a whole circle since
we no longer get to assume the nominal step angles are already known:

  1. Fit a 3D circle through the 5 cone peaks: its plane's normal is the
     turntable's rotation axis, its center is a point on that axis, and
     each peak's angular position around the circle tells us exactly how
     far that shot's turntable step actually was.
  2. Pick one shot as the reference angle. For every other shot, rotate
     its whole tomato cloud around the fitted axis (pivoting on the
     circle's center) by the angle that brings its own cone peak to the
     reference peak's angular position.
  3. With every shot now expressed at the same turntable angle, simply
     concatenate the 5 tomato clouds.

Usage:
    python merge_tomato_clouds.py [input_folder] [output_folder]

Defaults convert rgbd_photos/five_shots_segmented/*_tomato.ply (+ matching
*_cone.ply) into rgbd_photos/five_shots_merged/tomato_merged.ply, plus a
diagnostic render of the fitted circle and the before/after alignment.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

INPUT_FOLDER = "rgbd_photos/five_shots_segmented"
OUTPUT_FOLDER = "rgbd_photos/five_shots_merged"

FINAL_VOXEL_SIZE = 0.0012
OUTLIER_NB_NEIGHBORS = 30
OUTLIER_STD_RATIO = 1.5
RADIUS_OUTLIER_NB_POINTS = 8
RADIUS_OUTLIER_RADIUS = 0.003

SHOT_COLORS = [[0.85, 0.2, 0.2], [0.2, 0.65, 0.2], [0.2, 0.4, 0.9], [0.9, 0.6, 0.0], [0.6, 0.2, 0.8]]


def cone_peak(cone_pcd):
    """The cone's apex: the narrow tapering end, found via PCA along the
    cone's long axis and picking whichever end has the tighter local
    spread (the tip, not the wide circular base)."""
    pts = np.asarray(cone_pcd.points)
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


def rotate_about_axis(pcd, axis, center, angle_rad):
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle_rad)
    return pcd.rotate(R, center=center)


def save_overlay(clouds_points, filepath, title, extra=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    for i, pts in enumerate(clouds_points):
        step = max(1, len(pts) // 3000)
        ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2],
                    c=[SHOT_COLORS[i % len(SHOT_COLORS)]], s=1.0, label=f"shot {i + 1}")
    if extra is not None:
        extra(ax)
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
        tomato = o3d.io.read_point_cloud(str(input_folder / f"{stem}_tomato.ply"))
        cone = o3d.io.read_point_cloud(str(input_folder / f"{stem}_cone.ply"))
        peak = cone_peak(cone)
        shots.append({"stem": stem, "tomato": tomato, "peak": peak})
        print(f"{stem}: {len(tomato.points)} tomato points, {len(cone.points)} cone points, peak {peak.round(4)}")

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

    raw_points = [np.asarray(s["tomato"].points) for s in shots]
    save_overlay(raw_points, output_folder / "before_registration.png", "Raw (no alignment)")

    angles = np.array([angle_around_circle(p, center, e1, e2) for p in peaks])
    ref_idx = 0
    ref_angle = angles[ref_idx]
    print(f"\nReference shot: {shots[ref_idx]['stem']}")

    aligned_tomato_points = []
    for s, angle in zip(shots, angles):
        delta = ref_angle - angle
        rotate_about_axis(s["tomato"], normal, center, delta)
        aligned_tomato_points.append(np.asarray(s["tomato"].points))
        print(f"{s['stem']}: rotated {np.degrees(delta):+.2f} deg about the turntable axis")

    save_overlay(aligned_tomato_points, output_folder / "after_registration.png",
                 "After circle-based turntable-angle alignment")

    merged = o3d.geometry.PointCloud()
    for s in shots:
        merged += s["tomato"]

    merged = merged.voxel_down_sample(FINAL_VOXEL_SIZE)
    merged, _ = merged.remove_statistical_outlier(nb_neighbors=OUTLIER_NB_NEIGHBORS, std_ratio=OUTLIER_STD_RATIO)
    merged, _ = merged.remove_radius_outlier(nb_points=RADIUS_OUTLIER_NB_POINTS, radius=RADIUS_OUTLIER_RADIUS)

    out_path = output_folder / "tomato_merged.ply"
    o3d.io.write_point_cloud(str(out_path), merged)
    print(f"\nMerged cloud: {len(merged.points)} points -> {out_path}")


if __name__ == "__main__":
    main()
