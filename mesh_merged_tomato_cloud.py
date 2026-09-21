"""
Turn a merged tomato point cloud into a watertight-ish triangle mesh via
screened Poisson reconstruction.

Poisson reconstruction extrapolates a smooth surface everywhere it has
*any* normal support, including thin/sparse regions where the 5-shot
capture barely grazes the surface (e.g. near the poles) -- this produces
low-density "bloat" outside the real surface. That's trimmed by dropping
the lowest-density fraction of reconstructed vertices, then cropping to
the input cloud's bounding box as a final safety net.

Two more cleanup passes run after that:
  - small disconnected triangle islands (little blobs disjoint from the
    main tomato surface, also a side effect of Poisson filling sparse
    regions) are dropped, keeping only clusters that are a decent
    fraction of the largest one.
  - Taubin smoothing knocks down high-frequency spikes left by Poisson
    without the shrinkage plain Laplacian smoothing would cause.

Usage:
    python mesh_merged_tomato_cloud.py [input.ply] [output.ply]

Defaults convert rgbd_photos/five_shots_merged/tomato_merged.ply into
rgbd_photos/five_shots_merged/tomato_mesh.ply.
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d

INPUT_PATH = "rgbd_photos/five_shots_merged/tomato_merged.ply"
OUTPUT_PATH = "rgbd_photos/five_shots_merged/tomato_mesh.ply"

OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO = 2.0

# Scaled off this cloud's ~1.2 mm mean nearest-neighbor spacing (see
# FINAL_VOXEL_SIZE in merge_tomato_clouds.py).
NORMAL_RADIUS = 0.006
NORMAL_MAX_NN = 30
NORMAL_ORIENT_KNN = 50

POISSON_DEPTH = 9
DENSITY_TRIM_QUANTILE = 0.02  # drop the sparsest-supported 2% of vertices

# A connected triangle cluster is dropped unless it has at least this
# fraction of the largest cluster's triangle count.
MIN_CLUSTER_SIZE_FRACTION = 0.05

SMOOTH_ITERATIONS = 10
TAUBIN_LAMBDA = 0.5
TAUBIN_MU = -0.53


def load_cloud(path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise SystemExit(f"No points loaded from {path}")
    print(f"Loaded {len(pcd.points)} points from {path}")
    return pcd


def clean_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=OUTLIER_NB_NEIGHBORS, std_ratio=OUTLIER_STD_RATIO
    )
    print(f"After outlier removal: {len(pcd.points)} points")
    return pcd


def estimate_oriented_normals(pcd: o3d.geometry.PointCloud) -> None:
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=NORMAL_MAX_NN)
    )
    pcd.orient_normals_consistent_tangent_plane(NORMAL_ORIENT_KNN)


def poisson_mesh(pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH
    )
    densities = np.asarray(densities)

    threshold = np.quantile(densities, DENSITY_TRIM_QUANTILE)
    mesh.remove_vertices_by_mask(densities < threshold)
    print(f"Poisson mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles "
          f"(after trimming vertices below density quantile {DENSITY_TRIM_QUANTILE})")

    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)
    mesh.remove_unreferenced_vertices()
    print(f"After bbox crop: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    return mesh


def remove_small_clusters(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    min_size = cluster_n_triangles.max() * MIN_CLUSTER_SIZE_FRACTION
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_size
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    print(f"After dropping {(cluster_n_triangles < min_size).sum()} small cluster(s): "
          f"{len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    return mesh


def smooth_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh = mesh.filter_smooth_taubin(
        number_of_iterations=SMOOTH_ITERATIONS, lambda_filter=TAUBIN_LAMBDA, mu=TAUBIN_MU
    )
    mesh.compute_vertex_normals()
    return mesh


def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(INPUT_PATH)
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unsmoothed_path = output_path.with_name(f"{output_path.stem}_unsmoothed{output_path.suffix}")

    pcd = load_cloud(input_path)
    pcd = clean_cloud(pcd)
    estimate_oriented_normals(pcd)
    mesh = poisson_mesh(pcd)
    mesh = remove_small_clusters(mesh)

    o3d.io.write_triangle_mesh(str(unsmoothed_path), mesh)
    print(f"Saved unsmoothed mesh: {unsmoothed_path}")

    mesh = smooth_mesh(mesh)

    o3d.io.write_triangle_mesh(str(output_path), mesh)
    print(f"Saved smoothed mesh: {output_path}")


if __name__ == "__main__":
    main()
