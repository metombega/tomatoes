"""
Segment each raw fused tomato-rig scan in rgbd_photos/five_shots/ into a
tomato-only point cloud and a cone-marker-only point cloud.

Each *.ply in the input folder is a complete, already-fused multi-view scan
straight off a structured-light scanner (not a single-view RGBD capture
like the old try10 pipeline), in millimeters, with per-point RGB color
already baked in. Points are converted to meters on load so downstream
scripts can reuse the same distance-based thresholds as the rest of this
pipeline.

The tomato is found by elimination, not by matching "looks red": an
earlier version required a red-dominant color, which quietly excluded
anything that didn't match (specular highlights, non-red cultivars,
blemishes) and left holes at those exact spots. Instead:

  1. Remove the three background elements that DO have a fixed, known
     color regardless of the tomato's own ripeness or condition -- the
     violet cone marker, the warm gold/tan wood tray, and the white light
     dome.
  2. Cluster what's left and keep whichever cluster is both tomato-sized
     (longest bounding-box dimension in a plausible range) and blob-shaped
     (comparable extent on all three principal axes, unlike a flat tray
     panel or thin curved dome shell).

This works for a tomato of any color or with surface defects, since
neither factors into its size or shape. It isn't perfect: where the
tomato visibly rests against the tray, a thin, densely-fused sliver of
tray sometimes survives the color exclusion and stays attached in the
final cluster (its color there is genuinely ambiguous, between tomato and
wood). That's a small, cosmetic edge case relative to the holes/wrong-
cluster failures the color-only and pure-geometry approaches had.

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

# Wood tray: warm gold/tan, blue well below green.
WOOD_B_OVER_G_MAX = 0.35
WOOD_G_MIN = 0.15

# Light dome: bright and close to neutral (low saturation). Thresholds are
# loose enough to catch its own highlights/shadow gradient without also
# catching the tomato's own (more saturated, still clearly reddish) highlight.
DOME_MIN_CHANNEL_MIN = 0.4
DOME_MAX_SATURATION = 0.25

# Tomato candidate clustering, run on whatever's left after the above are
# removed.
TOMATO_CLUSTER_EPS_M = 0.0025
TOMATO_CLUSTER_MIN_POINTS = 30

# A tomato-sized candidate cluster's longest bounding-box dimension must
# fall in this range -- generous enough to span cherry to beefsteak
# tomatoes, while well short of the ~300mm wood tray/dome (or the ~140mm
# chunks left when the dome separates from the tray at clustering time).
TOMATO_MIN_DIM_M = 0.025
TOMATO_MAX_DIM_M = 0.10

# And it must be blob-shaped, not a flat panel or thin curved shell: the
# ratio between its largest and smallest PCA extent has to stay modest. A
# real tomato measures ~4-5 here; flat wood tray panels measure ~40-230.
TOMATO_MAX_FLATNESS_RATIO = 15

# Bright yellow marker sphere dropped at the detected cone peak, purely so
# it's easy to eyeball whether the peak landed in the right place.
PEAK_MARKER_RADIUS_M = 0.002
PEAK_MARKER_POINTS = 500
PEAK_MARKER_COLOR = (1.0, 1.0, 0.0)


def load_scaled(path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * MM_TO_M)
    return pcd


def cone_color_mask(cols):
    r, g, b = cols[:, 0], cols[:, 1], cols[:, 2]
    return (
        (b > CONE_B_MIN) & (b > r) & (b > g * CONE_B_OVER_G_MIN)
        & (r > CONE_R_MIN) & (r < CONE_R_MAX) & (g < CONE_G_MAX)
    )


def background_mask(cols):
    """Known-background color signatures: cone marker, wood tray, light
    dome. Deliberately says nothing about what the tomato itself should
    look like."""
    r, g, b = cols[:, 0], cols[:, 1], cols[:, 2]
    max_c = cols.max(axis=1)
    min_c = cols.min(axis=1)
    saturation = (max_c - min_c) / (max_c + 1e-9)

    cone = cone_color_mask(cols)
    wood = (b < WOOD_B_OVER_G_MAX * g) & (g > WOOD_G_MIN)
    dome = (min_c > DOME_MIN_CHANNEL_MIN) & (saturation < DOME_MAX_SATURATION)
    return cone | wood | dome


def pca_extents(points):
    """Rough extent of a point set along each of its own principal axes
    (widest first), used to tell a solid blob (comparable extent on all
    three axes) apart from a flat panel or thin curved shell (one axis
    much smaller than the other two)."""
    centered = points - points.mean(axis=0)
    cov = centered.T @ centered / len(centered)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    return np.sqrt(np.clip(eigvals, 0, None)) * 2


def segment_tomato(pcd, bg_mask):
    idx = np.where(~bg_mask)[0]
    if len(idx) == 0:
        return None
    sub = pcd.select_by_index(idx)
    labels = np.array(sub.cluster_dbscan(eps=TOMATO_CLUSTER_EPS_M, min_points=TOMATO_CLUSTER_MIN_POINTS))
    if labels.max() < 0:
        return None

    pts = np.asarray(sub.points)
    best_label, best_size = None, -1
    for label in range(labels.max() + 1):
        mask = labels == label
        cluster_pts = pts[mask]

        bbox_extent = cluster_pts.max(axis=0) - cluster_pts.min(axis=0)
        if not (TOMATO_MIN_DIM_M < bbox_extent.max() < TOMATO_MAX_DIM_M):
            continue

        extents = pca_extents(cluster_pts)
        flatness = extents[0] / max(extents[2], 1e-9)
        if flatness > TOMATO_MAX_FLATNESS_RATIO:
            continue

        if mask.sum() > best_size:
            best_label, best_size = label, int(mask.sum())

    if best_label is None:
        return None
    return sub.select_by_index(np.where(labels == best_label)[0])


def segment_cone(pcd, cone_mask):
    idx = np.where(cone_mask)[0]
    if len(idx) == 0:
        return None
    labels = np.array(pcd.select_by_index(idx).cluster_dbscan(eps=CONE_CLUSTER_EPS_M, min_points=CONE_CLUSTER_MIN_POINTS))
    if labels.max() < 0:
        return None
    counts = np.bincount(labels[labels >= 0])
    best = int(np.argmax(counts))
    return pcd.select_by_index(idx).select_by_index(np.where(labels == best)[0])


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
        cols = np.asarray(pcd.colors)

        tomato = segment_tomato(pcd, background_mask(cols))
        cone = segment_cone(pcd, cone_color_mask(cols))

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
