"""
Segment each raw fused tomato-rig mesh in rgbd_photos/five_meshes/ into a
tomato-only mesh and a cone-marker-only mesh.

Mesh counterpart of segment_tomato_cone.py: same rig, same color-based
segmentation, same cone-peak computation, but the *.ply files here are
already triangulated (a structured-light scanner's own mesh output) rather
than raw point clouds. Triangle connectivity is preserved throughout --
segmentation selects whole triangles (via vertex indices), not just loose
points, so the outputs are still meshes, not clouds.

Usage:
    python segment_tomato_cone_mesh.py [input_folder] [output_folder] [pattern]

Defaults convert rgbd_photos/five_meshes/*.ply into
rgbd_photos/five_meshes_segmented/<stem>_tomato.ply and
rgbd_photos/five_meshes_segmented/<stem>_cone.ply.
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d

INPUT_FOLDER = "rgbd_photos/five_meshes"
OUTPUT_FOLDER = "rgbd_photos/five_meshes_segmented"
PATTERN = "*.ply"

MM_TO_M = 1e-3

# Same thresholds as segment_tomato_cone.py -- same rig, same colors.
TOMATO_R_MIN = 0.25
TOMATO_G_MAX_RATIO = 0.35
TOMATO_B_MAX_RATIO = 0.5
TOMATO_CLUSTER_EPS_M = 0.0025
TOMATO_CLUSTER_MIN_POINTS = 30

CONE_B_MIN = 0.3
CONE_R_MIN = 0.15
CONE_R_MAX = 0.55
CONE_G_MAX = 0.4
CONE_B_OVER_G_MIN = 1.3
CONE_CLUSTER_EPS_M = 0.002
CONE_CLUSTER_MIN_POINTS = 15

PEAK_MARKER_RADIUS_M = 0.002
PEAK_MARKER_COLOR = (1.0, 1.0, 0.0)


def load_scaled(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path))
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * MM_TO_M)
    return mesh


def largest_cluster_submesh(mesh, indices, eps, min_points):
    """Given candidate vertex indices, DBSCAN-cluster their positions and
    return the sub-mesh (whole triangles, not loose points) for the
    largest connected cluster."""
    if len(indices) == 0:
        return None
    points = np.asarray(mesh.vertices)[indices]
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    labels = np.array(cloud.cluster_dbscan(eps=eps, min_points=min_points))
    if labels.max() < 0:
        return None
    counts = np.bincount(labels[labels >= 0])
    best = int(np.argmax(counts))
    global_idx = indices[labels == best]
    return mesh.select_by_index(global_idx.tolist())


def segment_tomato(mesh):
    cols = np.asarray(mesh.vertex_colors)
    r, g, b = cols[:, 0], cols[:, 1], cols[:, 2]
    mask = (r > TOMATO_R_MIN) & (g < TOMATO_G_MAX_RATIO * r) & (b < TOMATO_B_MAX_RATIO * r)
    indices = np.where(mask)[0]
    return largest_cluster_submesh(mesh, indices, TOMATO_CLUSTER_EPS_M, TOMATO_CLUSTER_MIN_POINTS)


def segment_cone(mesh):
    cols = np.asarray(mesh.vertex_colors)
    r, g, b = cols[:, 0], cols[:, 1], cols[:, 2]
    mask = (
        (b > CONE_B_MIN) & (b > r) & (b > g * CONE_B_OVER_G_MIN)
        & (r > CONE_R_MIN) & (r < CONE_R_MAX) & (g < CONE_G_MAX)
    )
    indices = np.where(mask)[0]
    return largest_cluster_submesh(mesh, indices, CONE_CLUSTER_EPS_M, CONE_CLUSTER_MIN_POINTS)


def cone_peak(cone_mesh):
    """The cone's apex: the narrow tapering end, found via PCA along the
    cone's long axis and picking whichever end has the tighter local
    spread (the tip, not the wide circular base)."""
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


def peak_marker_mesh(peak, radius=PEAK_MARKER_RADIUS_M, color=PEAK_MARKER_COLOR):
    """A small bright sphere mesh centered on `peak`, purely so the
    detected cone peak is easy to spot by eye in any mesh viewer."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(peak)
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()
    return sphere


def main():
    input_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(INPUT_FOLDER)
    output_folder = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(OUTPUT_FOLDER)
    pattern = sys.argv[3] if len(sys.argv) > 3 else PATTERN

    files = sorted(input_folder.glob(pattern))
    if not files:
        raise SystemExit(f"No files matching {pattern!r} found in {input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)

    for mesh_path in files:
        mesh = load_scaled(mesh_path)

        tomato = segment_tomato(mesh)
        cone = segment_cone(mesh)

        if tomato is None or len(tomato.vertices) == 0:
            print(f"Skipping {mesh_path.name}: no tomato mesh found")
            continue
        if cone is None or len(cone.vertices) == 0:
            print(f"Skipping {mesh_path.name}: no cone mesh found")
            continue

        peak = cone_peak(cone)

        stem = mesh_path.stem
        tomato_path = output_folder / f"{stem}_tomato.ply"
        cone_path = output_folder / f"{stem}_cone.ply"
        peak_preview_path = output_folder / f"{stem}_peak_preview.ply"
        o3d.io.write_triangle_mesh(str(tomato_path), tomato)
        o3d.io.write_triangle_mesh(str(cone_path), cone)
        o3d.io.write_triangle_mesh(str(peak_preview_path), tomato + cone + peak_marker_mesh(peak))

        print(
            f"{mesh_path.name}: {len(tomato.vertices)}v/{len(tomato.triangles)}t tomato -> {tomato_path.name}, "
            f"{len(cone.vertices)}v/{len(cone.triangles)}t cone (peak {peak.round(4)}) -> {cone_path.name}, "
            f"preview with peak marker -> {peak_preview_path.name}"
        )


if __name__ == "__main__":
    main()
