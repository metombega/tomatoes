"""
Segment each raw fused tomato-rig scan in rgbd_photos/five_shots/ into a
tomato-only point cloud and a cone-marker-only point cloud.

Each *.ply in the input folder is a complete, already-fused multi-view scan
straight off a structured-light scanner (not a single-view RGBD capture
like the old try10 pipeline), in millimeters, with per-point RGB color
already baked in. Points are converted to meters on load so downstream
scripts can reuse the same distance-based thresholds as the rest of this
pipeline.

Two objects are color-segmented out of the scene per file:
  - the tomato itself: strongly red, with green and blue both clearly
    below red. That excludes the warm gold/tan wood tray (which has much
    higher green) and the white light dome.
  - a small violet cone-shaped registration marker mounted in the rig,
    visible in every scan. Its peak (apex) is a precise, repeatable 3D
    point used by merge_tomato_clouds.py to register the 5 scans together.

Usage:
    python segment_tomato_cone.py [input_folder] [output_folder] [pattern]

Defaults convert rgbd_photos/five_shots/*.ply into
rgbd_photos/five_shots_segmented/<stem>_tomato.ply and
rgbd_photos/five_shots_segmented/<stem>_cone.ply.
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d

INPUT_FOLDER = "rgbd_photos/five_shots"
OUTPUT_FOLDER = "rgbd_photos/five_shots_segmented"
PATTERN = "*.ply"

MM_TO_M = 1e-3

# Tomato: red must dominate, with green and blue both clearly below it.
# Excludes the gold/tan wood tray (high green) and the white light dome.
TOMATO_R_MIN = 0.25
TOMATO_G_MAX_RATIO = 0.35
TOMATO_B_MAX_RATIO = 0.5
TOMATO_CLUSTER_EPS_M = 0.0025
TOMATO_CLUSTER_MIN_POINTS = 30

# Cone: violet -- blue channel clearly dominant over both red and green,
# which distinguishes it from a pinkish/magenta flat sticker elsewhere in
# the scene (that one has red, not blue, as its dominant channel).
CONE_B_MIN = 0.3
CONE_R_MIN = 0.15
CONE_R_MAX = 0.55
CONE_G_MAX = 0.4
CONE_B_OVER_G_MIN = 1.3
CONE_CLUSTER_EPS_M = 0.002
CONE_CLUSTER_MIN_POINTS = 15

# Bright yellow marker sphere dropped at the detected cone peak, purely so
# it's easy to eyeball whether the peak landed in the right place.
PEAK_MARKER_RADIUS_M = 0.002
PEAK_MARKER_POINTS = 500
PEAK_MARKER_COLOR = (1.0, 1.0, 0.0)


def load_scaled(path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * MM_TO_M)
    return pcd


def largest_cluster(pcd, eps, min_points):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    if labels.max() < 0:
        return None
    counts = np.bincount(labels[labels >= 0])
    best = int(np.argmax(counts))
    return pcd.select_by_index(np.where(labels == best)[0])


def segment_tomato(pcd):
    cols = np.asarray(pcd.colors)
    r, g, b = cols[:, 0], cols[:, 1], cols[:, 2]
    mask = (r > TOMATO_R_MIN) & (g < TOMATO_G_MAX_RATIO * r) & (b < TOMATO_B_MAX_RATIO * r)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    return largest_cluster(pcd.select_by_index(idx), TOMATO_CLUSTER_EPS_M, TOMATO_CLUSTER_MIN_POINTS)


def segment_cone(pcd):
    cols = np.asarray(pcd.colors)
    r, g, b = cols[:, 0], cols[:, 1], cols[:, 2]
    mask = (
        (b > CONE_B_MIN) & (b > r) & (b > g * CONE_B_OVER_G_MIN)
        & (r > CONE_R_MIN) & (r < CONE_R_MAX) & (g < CONE_G_MAX)
    )
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    return largest_cluster(pcd.select_by_index(idx), CONE_CLUSTER_EPS_M, CONE_CLUSTER_MIN_POINTS)


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


def peak_marker(peak, radius=PEAK_MARKER_RADIUS_M, n_points=PEAK_MARKER_POINTS, color=PEAK_MARKER_COLOR):
    """A small bright sphere of points centered on `peak`, purely so the
    detected cone peak is easy to spot by eye when viewing the segmented
    cloud in any point-cloud viewer."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    marker = sphere.sample_points_uniformly(number_of_points=n_points)
    marker.translate(peak)
    marker.paint_uniform_color(color)
    return marker


def main():
    input_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(INPUT_FOLDER)
    output_folder = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(OUTPUT_FOLDER)
    pattern = sys.argv[3] if len(sys.argv) > 3 else PATTERN

    files = sorted(input_folder.glob(pattern))
    if not files:
        raise SystemExit(f"No files matching {pattern!r} found in {input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)

    for ply_path in files:
        pcd = load_scaled(ply_path)

        tomato = segment_tomato(pcd)
        cone = segment_cone(pcd)

        if tomato is None or len(tomato.points) == 0:
            print(f"Skipping {ply_path.name}: no tomato points found")
            continue
        if cone is None or len(cone.points) == 0:
            print(f"Skipping {ply_path.name}: no cone points found")
            continue

        peak = cone_peak(cone)

        stem = ply_path.stem
        tomato_path = output_folder / f"{stem}_tomato.ply"
        cone_path = output_folder / f"{stem}_cone.ply"
        peak_preview_path = output_folder / f"{stem}_peak_preview.ply"
        o3d.io.write_point_cloud(str(tomato_path), tomato)
        o3d.io.write_point_cloud(str(cone_path), cone)
        o3d.io.write_point_cloud(str(peak_preview_path), tomato + cone + peak_marker(peak))

        print(
            f"{ply_path.name}: {len(tomato.points)} tomato points -> {tomato_path.name}, "
            f"{len(cone.points)} cone points (peak {peak.round(4)}) -> {cone_path.name}, "
            f"preview with peak marker -> {peak_preview_path.name}"
        )


if __name__ == "__main__":
    main()
