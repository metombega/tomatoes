"""
Turn a merged tomato point cloud into a watertight-ish triangle mesh via
screened Poisson reconstruction.

Poisson reconstruction extrapolates a smooth surface everywhere it has
*any* normal support, including thin/sparse regions where the 5-shot
capture barely grazes the surface (e.g. near the poles) -- this produces
low-density "bloat" outside the real surface. That's trimmed by dropping
the lowest-density fraction of reconstructed vertices, then cropping to
the input cloud's bounding box as a final safety net.

That density trim is deliberately light (see DENSITY_TRIM_QUANTILE): an
aggressive trim strips the entire unsupported pole cap and leaves a big
open hole there, while too light a trim leaves a thin unrealistic "spike"
poking out of the pole (low-density vertices Poisson extrapolated from
almost no real data). The chosen value keeps the smoothly-curved shoulder
around the pole while dropping just the spike tip -- what's left is a
single small, clean hole rather than either extreme.

Three more cleanup passes run after that:
  - small disconnected triangle islands (little blobs disjoint from the
    main tomato surface, also a side effect of Poisson filling sparse
    regions) are dropped, keeping only clusters that are a decent
    fraction of the largest one.
  - remaining holes (the pole cap, plus small gaps from sparse/noisy
    coverage elsewhere) are closed with a manual fan triangulation
    (see close_holes below). Open3D's own o3d.t.geometry fill_holes()
    was tried first and rejected: on this mesh it intermittently produced
    runaway vertices (coordinates order 1e30+) or segfaulted outright,
    so we don't depend on it.
  - Taubin smoothing knocks down high-frequency spikes left by Poisson
    without the shrinkage plain Laplacian smoothing would cause.

After that, the mesh is simplified: quadric decimation collapses it down
to a fixed target triangle count, which -- on top of shrinking the file --
rounds off a lot of the remaining small-scale bumpiness by construction,
since it can no longer represent detail finer than the simplified triangle
size. A short final smoothing pass cleans up whatever new sharp edges
decimation introduces.

Usage:
    python mesh_merged_tomato_cloud.py [input.ply] [output.ply]

Defaults convert rgbd_photos/five_shots_merged/tomato_merged.ply into
rgbd_photos/five_shots_merged/tomato_mesh.ply.
"""

import collections
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

POISSON_DEPTH = 7
DENSITY_TRIM_QUANTILE = 0.002  # drop only the sparsest-supported 0.2% of vertices

# A connected triangle cluster is dropped unless it has at least this
# fraction of the largest cluster's triangle count.
MIN_CLUSTER_SIZE_FRACTION = 0.05

# New hole-closing caps are pushed outward along the local surface normal
# by this much so they bulge gently instead of sitting perfectly flat.
HOLE_FILL_BULGE = 0.0015

SMOOTH_ITERATIONS = 40
TAUBIN_LAMBDA = 0.5
TAUBIN_MU = -0.53

# Quadric decimation target, as an absolute triangle count. Simplification
# is what actually gets rid of a lot of the remaining fine bumpiness (it
# physically can't represent detail smaller than the simplified triangles),
# on top of shrinking the file. An absolute count (rather than a fraction
# of the pre-decimation count) gives a predictable, repeatable low-poly
# look regardless of how many triangles Poisson happens to produce.
SIMPLIFY_TARGET_TRIANGLES = 10000

# A second Taubin pass to round off the faceted look decimation leaves
# behind. This needs to be a real pass, not a token cleanup -- a handful
# of iterations visibly still looked low-poly/crystalline; smoothness
# kept improving up to ~60 iterations and flattened out after that.
POST_SIMPLIFY_SMOOTH_ITERATIONS = 60


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


def find_boundary_loops(mesh: o3d.geometry.TriangleMesh):
    """Boundary edges (belonging to exactly one triangle), walked into
    closed vertex loops. Only simple loops (every vertex touches exactly
    two boundary edges) are returned -- a vertex shared by more than one
    hole boundary is left alone rather than guessed at."""
    edge_count = collections.Counter()
    for a, b, c in np.asarray(mesh.triangles):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_count[(u, v) if u < v else (v, u)] += 1
    boundary_edges = [e for e, count in edge_count.items() if count == 1]

    adjacency = collections.defaultdict(list)
    for a, b in boundary_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    visited = set()
    loops = []
    for start in list(adjacency):
        if start in visited or len(adjacency[start]) != 2:
            continue
        loop = [start]
        visited.add(start)
        prev, current = None, start
        closed = True
        while True:
            candidates = [n for n in adjacency[current] if n != prev]
            nxt = start if start in candidates else next(
                (n for n in candidates if n not in visited and len(adjacency[n]) == 2), None
            )
            if nxt is None:
                closed = False
                break
            if nxt == start:
                break
            loop.append(nxt)
            visited.add(nxt)
            prev, current = current, nxt
        if closed and len(loop) >= 3:
            loops.append(loop)
    return loops


def close_holes(mesh: o3d.geometry.TriangleMesh, bulge=HOLE_FILL_BULGE) -> o3d.geometry.TriangleMesh:
    """Close each boundary loop with a fan of triangles from its centroid,
    nudged outward along the loop's average vertex normal so the cap
    bulges gently instead of sitting dead flat."""
    loops = find_boundary_loops(mesh)
    if not loops:
        return mesh

    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices).copy()
    normals = np.asarray(mesh.vertex_normals)
    triangles = np.asarray(mesh.triangles).tolist()

    for loop in loops:
        loop_normal = normals[loop].mean(axis=0)
        loop_normal /= np.linalg.norm(loop_normal)
        center = vertices[loop].mean(axis=0) + loop_normal * bulge
        center_idx = len(vertices)
        vertices = np.vstack([vertices, center])

        n = len(loop)
        for i in range(n):
            a, b = loop[i], loop[(i + 1) % n]
            face_normal = np.cross(vertices[b] - vertices[a], center - vertices[a])
            # Match winding to the surrounding surface, not the arbitrary loop order.
            tri = [a, b, center_idx] if np.dot(face_normal, normals[a] + normals[b]) >= 0 else [b, a, center_idx]
            triangles.append(tri)

    filled = o3d.geometry.TriangleMesh()
    filled.vertices = o3d.utility.Vector3dVector(vertices)
    filled.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
        pad = np.tile(colors.mean(axis=0), (len(vertices) - len(colors), 1))
        filled.vertex_colors = o3d.utility.Vector3dVector(np.vstack([colors, pad]))

    print(f"Closed {len(loops)} hole(s): {len(mesh.vertices)} -> {len(filled.vertices)} vertices, "
          f"{len(mesh.triangles)} -> {len(filled.triangles)} triangles")
    return filled


def repair_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Drop degenerate/duplicated geometry and edges shared by more than
    two triangles (seam artifacts from the 5 registered shots slightly
    overlapping) before hole-closing, so find_boundary_loops sees clean,
    simple loops instead of tangled non-manifold junctions."""
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    return mesh


def smooth_mesh(mesh: o3d.geometry.TriangleMesh, iterations: int) -> o3d.geometry.TriangleMesh:
    mesh = mesh.filter_smooth_taubin(
        number_of_iterations=iterations, lambda_filter=TAUBIN_LAMBDA, mu=TAUBIN_MU
    )
    mesh.compute_vertex_normals()
    return mesh


def simplify_mesh(mesh: o3d.geometry.TriangleMesh, target=SIMPLIFY_TARGET_TRIANGLES) -> o3d.geometry.TriangleMesh:
    simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
    simplified.compute_vertex_normals()
    print(f"Simplified: {len(mesh.vertices)} -> {len(simplified.vertices)} vertices, "
          f"{len(mesh.triangles)} -> {len(simplified.triangles)} triangles")
    return simplified


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
    mesh = repair_mesh(mesh)
    mesh = close_holes(mesh)

    o3d.io.write_triangle_mesh(str(unsmoothed_path), mesh)
    print(f"Saved unsmoothed mesh: {unsmoothed_path}")

    mesh = smooth_mesh(mesh, SMOOTH_ITERATIONS)
    mesh = simplify_mesh(mesh)
    mesh = smooth_mesh(mesh, POST_SIMPLIFY_SMOOTH_ITERATIONS)

    o3d.io.write_triangle_mesh(str(output_path), mesh)
    print(f"Saved smoothed + simplified mesh: {output_path}")


if __name__ == "__main__":
    main()
