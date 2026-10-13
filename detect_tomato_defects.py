"""
Detect dark / rotten / off-color spots on a tomato mesh and report where they are.

Everything runs on the final reconstructed triangle mesh (vertices, vertex
colors, triangle connectivity); no point cloud is used. A vertex is flagged
when it is clearly darker than the mesh color in its own neighborhood, or its
hue/chroma is off (tan, brown, green blemishes). No absolute darkness cutoff
is used: video exposure varies, and a whole dimly lit tomato can sit at low L.

Flagged vertices are grouped into spots by mesh connectivity (neighbors across
a shared triangle edge). Each spot is reported with its 3D centroid, surface
area, darkness, and position along / around the tomato's axis. Spots in the
pole regions (stem scar / calyx are naturally dark) are ignored. A copy of
the mesh with spots painted bright magenta is written for visual checking.
"""
import argparse

import cv2
import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

DEFAULT_MESH = (
    r"rgbd_photos\video_on_ratation_table_output_20261013_105001"
    r"\merged\tomato_mesh.ply"
)

AXIS_DIM = {"x": 0, "y": 1, "z": 2}

NEIGHBOR_RADIUS = 0.008      # m; scale of the "local normal color" baseline
MIN_DELTA_L = 10.0           # Lab-L drop vs. local median to count as dark
MAX_BRIGHTER_L = 5.0         # brighter than this vs. surroundings = glare, skipped
MIN_DELTA_AB = 8.0          # Lab a/b distance from local median (off-red blemish)
MIN_SPOT_AREA = 4e-6         # m^2 (4 mm^2); smaller clusters are noise
POLE_FRACTION = 0.10         # ignore this fraction of height at each end
SPOT_COLOR = (1.0, 0.0, 1.0)


def to_lab(rgb):
    """Lab (L 0..100, a/b ~ -128..127) for float RGB in 0..1."""
    img = rgb.astype(np.float32).reshape(-1, 1, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:, 0, :].astype(np.float64)


def local_median(points, values, radius):
    """Per-point median of `values` (N or N,k) over neighbors within radius."""
    nbrs = cKDTree(points).query_ball_point(points, r=radius, workers=-1)
    return np.array([np.median(values[idx], axis=0) for idx in nbrs])


def vertex_areas(verts, tris):
    """Surface area owned by each vertex (1/3 of each adjacent triangle)."""
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    tri_area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    areas = np.zeros(len(verts))
    for k in range(3):
        np.add.at(areas, tris[:, k], tri_area / 3.0)
    return areas, tri_area.sum()


def flagged_components(tris, mask):
    """Connected components of flagged vertices, linked by shared mesh edges."""
    n = len(mask)
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = e[mask[e[:, 0]] & mask[e[:, 1]]]
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, labels = connected_components(g, directed=False)
    return labels


def detect_defects(
    mesh_path,
    axis="y",
    radius=NEIGHBOR_RADIUS,
    min_delta_l=MIN_DELTA_L,
    min_delta_ab=MIN_DELTA_AB,
    max_brighter_l=MAX_BRIGHTER_L,
    min_spot_area=MIN_SPOT_AREA,
    pole_fraction=POLE_FRACTION,
):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(mesh.triangles) == 0:
        raise ValueError(f"Could not load a triangle mesh from: {mesh_path}")
    if not mesh.has_vertex_colors():
        raise ValueError("Mesh has no vertex colors.")
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    rgb = np.asarray(mesh.vertex_colors)

    lab = to_lab(rgb)
    diff = local_median(verts, lab, radius) - lab
    delta = diff[:, 0]                           # positive = darker than around
    delta_ab = np.hypot(diff[:, 1], diff[:, 2])  # color shift regardless of brightness
    not_glare = delta > -max_brighter_l
    flagged = (delta > min_delta_l) | ((delta_ab > min_delta_ab) & not_glare)

    # Pole exclusion along the chosen axis.
    a = AXIS_DIM[axis]
    lo, hi = verts[:, a].min(), verts[:, a].max()
    margin = pole_fraction * (hi - lo)
    flagged &= (verts[:, a] > lo + margin) & (verts[:, a] < hi - margin)

    areas, total_area = vertex_areas(verts, tris)
    r1, r2 = [d for d in range(3) if d != a]
    center = np.median(verts, axis=0)

    labels = flagged_components(tris, flagged)
    spots = []
    for lab_id in np.unique(labels[flagged]):
        members = np.flatnonzero(flagged & (labels == lab_id))
        area = areas[members].sum()
        if area < min_spot_area:
            continue
        c = np.average(verts[members], axis=0, weights=areas[members])
        spots.append({
            "members": members,
            "centroid": c,
            "area_mm2": area * 1e6,
            "darkness": float(np.mean(delta[members])),
            "chroma_shift": float(np.mean(delta_ab[members])),
            "height": float((c[a] - lo) / (hi - lo)),  # 0 = bottom pole, 1 = top
            "angle_deg": float(np.degrees(
                np.arctan2(c[r2] - center[r2], c[r1] - center[r1])) % 360),
        })
    spots.sort(key=lambda s: -s["area_mm2"])

    painted = o3d.geometry.TriangleMesh(mesh)
    colors = rgb.copy()
    for s in spots:
        colors[s["members"]] = SPOT_COLOR
    painted.vertex_colors = o3d.utility.Vector3dVector(colors)
    painted.compute_vertex_normals()

    return {
        "spots": spots,
        "defect_fraction": sum(s["area_mm2"] for s in spots) * 1e-6 / total_area,
        "painted_mesh": painted,
        "total_area_m2": total_area,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mesh", nargs="?", default=DEFAULT_MESH)
    p.add_argument("--axis", default="y", choices=["x", "y", "z"])
    p.add_argument("--out", default="defects_painted_mesh.ply")
    p.add_argument("--min-delta-l", type=float, default=MIN_DELTA_L,
                   help=f"Lab-L drop vs. surroundings to flag as dark (default {MIN_DELTA_L})")
    p.add_argument("--min-delta-ab", type=float, default=MIN_DELTA_AB,
                   help=f"Lab a/b color shift to flag as off-color (default {MIN_DELTA_AB})")
    p.add_argument("--max-brighter-l", type=float, default=MAX_BRIGHTER_L,
                   help=f"skip off-color points brighter than this (glare) (default {MAX_BRIGHTER_L})")
    p.add_argument("--radius-mm", type=float, default=NEIGHBOR_RADIUS * 1e3,
                   help=f"neighborhood for the local baseline color, mm (default {NEIGHBOR_RADIUS * 1e3:g})")
    p.add_argument("--min-area-mm2", type=float, default=MIN_SPOT_AREA * 1e6,
                   help=f"drop spots smaller than this (default {MIN_SPOT_AREA * 1e6:g})")
    p.add_argument("--pole-fraction", type=float, default=POLE_FRACTION,
                   help=f"fraction of height ignored at each end (default {POLE_FRACTION})")
    p.add_argument("--show", action="store_true", help="open a viewer")
    args = p.parse_args()

    r = detect_defects(
        args.mesh, args.axis,
        radius=args.radius_mm * 1e-3,
        min_delta_l=args.min_delta_l,
        min_delta_ab=args.min_delta_ab,
        max_brighter_l=args.max_brighter_l,
        min_spot_area=args.min_area_mm2 * 1e-6,
        pole_fraction=args.pole_fraction,
    )
    spots = r["spots"]
    print(f"\n=== Tomato Defects ===\nFile: {args.mesh}")
    print(f"Surface area: {r['total_area_m2'] * 1e4:.1f} cm^2")
    print(f"Spots found : {len(spots)}")
    print(f"Defect area : {100 * r['defect_fraction']:.2f}% of surface")
    if spots:
        print(f"\n{'#':>2} {'area mm^2':>10} {'darkness':>9} {'chroma':>7} {'height':>7} {'angle':>6}  centroid (m)")
        for i, s in enumerate(spots, 1):
            c = s["centroid"]
            print(f"{i:>2} {s['area_mm2']:>10.1f} {s['darkness']:>9.1f} {s['chroma_shift']:>7.1f} "
                  f"{s['height']:>7.2f} {s['angle_deg']:>5.0f}°  "
                  f"({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})")
        print("\nheight: 0 = bottom pole, 1 = top pole;  angle: around the axis")
    o3d.io.write_triangle_mesh(args.out, r["painted_mesh"])
    print(f"\nWrote painted mesh: {args.out}")
    if args.show:
        o3d.visualization.draw_geometries([r["painted_mesh"]])
