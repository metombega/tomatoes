"""
Extract quality features from a reconstructed tomato mesh and print them.

Features:
  - Volume (signed-tetrahedron sum; mesh units assumed to be meters)
  - Symmetry score (from symmetry_calculator, normalized)
  - Average color score  (mean "redness" of the vertex colors)
  - Color consistency score (1 - spread of redness)
  - Defect score (1 - defect area / 10%, from detect_tomato_defects)
  - Total score (weighted blend of the four scores)
All scores are normalized to 0 (bad) .. 1 (best).
"""
import argparse
import colorsys

import numpy as np
import open3d as o3d

from detect_tomato_defects import detect_defects
from symmetry_calculator import compute_axis_symmetry

DEFAULT_MESH = (
    r"rgbd_photos\video_on_ratation_table_output_20261013_105001"
    r"\merged\tomato_mesh.ply"
)

# DEFAULT_MESH = (
#     r"random_tomato_mesh.obj"
# )

# Total-score weights (sum to 1).
W_COLOR = 0.25
W_CONSISTENCY = 0.20
W_SYMMETRY = 0.30
W_DEFECTS = 0.25

# Defect area fraction at/above which the defect score hits 0 (0% -> 1.0).
DEFECTS_WORST_FRACTION = 0.10

# Raw symmetry defect mapped linearly: <= BEST -> 1.0, >= WORST -> 0.0
# (so the midpoint 0.18 -> 0.5).
SYMMETRY_BEST = 0.03
SYMMETRY_WORST = 0.33
# Redness std at/above which the consistency sub-score hits 0.
CONSISTENCY_WORST_STD = 0.25


def mesh_volume(mesh):
    """Signed volume via the divergence theorem. Exact only for closed meshes."""
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    return abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def redness(rgb):
    """
    Per-vertex ripeness proxy in 0..1: how red the color is. Combines hue
    closeness to pure red with saturation, so green/yellow/pale areas score low.
    """
    out = np.empty(len(rgb))
    for i, (r, g, b) in enumerate(rgb):
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        hue_dist = min(h, 1.0 - h)  # distance from red (hue 0), 0..0.5
        hue_score = max(0.0, 1.0 - hue_dist / 0.17)  # ~60 deg away -> 0
        out[i] = hue_score * s
    return out


def extract_features(mesh_path, axis="y", num_points=40000):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(f"Could not load a triangle mesh from: {mesh_path}")
    if not mesh.has_vertex_colors():
        raise ValueError("Mesh has no vertex colors; cannot compute color scores.")

    volume_m3 = mesh_volume(mesh)
    watertight = mesh.is_watertight()

    pts = np.asarray(mesh.sample_points_uniformly(num_points).points)
    sym_raw = compute_axis_symmetry(pts, axis=axis)["combined_symmetry_score"]
    sym = float(np.clip(
        (SYMMETRY_WORST - sym_raw) / (SYMMETRY_WORST - SYMMETRY_BEST), 0.0, 1.0))

    red = redness(np.asarray(mesh.vertex_colors))
    avg_color = float(np.mean(red))
    color_std = float(np.std(red))
    consistency = float(max(0.0, 1.0 - color_std / CONSISTENCY_WORST_STD))

    defect_fraction = detect_defects(mesh_path, axis=axis)["defect_fraction"]
    defects = float(np.clip(1.0 - defect_fraction / DEFECTS_WORST_FRACTION, 0.0, 1.0))

    total = (W_COLOR * avg_color + W_CONSISTENCY * consistency
             + W_SYMMETRY * sym + W_DEFECTS * defects)

    return {
        "volume_m3": volume_m3,
        "watertight": watertight,
        "symmetry_score": sym,
        "symmetry_raw": sym_raw,
        "avg_color_score": avg_color,
        "color_consistency_score": consistency,
        "defect_fraction": defect_fraction,
        "defect_score": defects,
        "total_score": total,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mesh", nargs="?", default=DEFAULT_MESH)
    p.add_argument("--axis", default="y", choices=["x", "y", "z"],
                   help="symmetry axis (default: y)")
    args = p.parse_args()

    f = extract_features(args.mesh, axis=args.axis)
    print(f"\n=== Tomato Features ===\nFile: {args.mesh}")
    print(f"Volume                  : {f['volume_m3'] * 1e6:.1f} cm^3"
          + ("" if f["watertight"] else "  (mesh not watertight; approximate)"))
    print(f"Symmetry score          : {f['symmetry_score']:.3f}  (raw defect {f['symmetry_raw']:.4f})")
    print(f"Average color score     : {f['avg_color_score']:.3f}")
    print(f"Color consistency score : {f['color_consistency_score']:.3f}")
    print(f"Defect score            : {f['defect_score']:.3f}  ({100 * f['defect_fraction']:.2f}% of surface)")
    print(f"Total score             : {f['total_score']:.3f}")
