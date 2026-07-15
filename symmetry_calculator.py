import numpy as np
import open3d as o3d


# ============================================================
# Settings
# ============================================================

MESH_PATH = "random_tomato_mesh.ply"
# MESH_PATH = "tomato_model_smooth.stl"   # or "random_tomato_mesh.obj"
NUM_SAMPLE_POINTS = 40000

NUM_SLICES = 50
ANGLE_BINS = 72
MIN_POINTS_PER_SLICE = 80

# Ignore a little at top/bottom to reduce stem / bottom scar influence
IGNORE_TOP_FRACTION = 0.12
IGNORE_BOTTOM_FRACTION = 0.05

# Axis of rotational symmetry to check: "x", "y", or "z"
SYMMETRY_AXIS = "y"

SHOW_VISUALIZATION = True


# ============================================================
# Loading mesh / point cloud
# ============================================================

def load_as_point_cloud(path, num_sample_points=40000):
    """
    Load either:
    - a triangle mesh (.ply / .obj / etc.), then sample points from it
    - or a point cloud directly

    Returns:
        points: (N,3) numpy array
        geometry_for_view: geometry to visualize
    """

    # Try mesh first
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=num_sample_points)
        points = np.asarray(pcd.points)
        return points, mesh

    # If not a mesh, try point cloud
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) > 0:
        points = np.asarray(pcd.points)
        return points, pcd

    raise ValueError(f"Could not load a mesh or point cloud from: {path}")


# ============================================================
# Symmetry calculation
# ============================================================

# Maps axis name to (along_dim, radial_dim1, radial_dim2)
_AXIS_DIMS = {
    "x": (0, 1, 2),
    "y": (1, 0, 2),
    "z": (2, 0, 1),
}


def compute_axis_symmetry(
    points,
    axis="y",
    num_slices=50,
    angle_bins=72,
    min_points_per_slice=80,
    ignore_top_fraction=0.12,
    ignore_bottom_fraction=0.05,
):
    """
    Measure rotational symmetry around the given axis ("x", "y", or "z").

    Main ideas:
    1. Center the object so the candidate axis passes through the origin.
    2. Slice perpendicular to the chosen axis.
    3. For each slice:
       - measure how much the radius varies with angle  -> roundness
       - measure how far that slice's center drifts     -> axis wobble

    Returns a dictionary with scores and details.
    Lower score = more symmetric.
    """
    if axis not in _AXIS_DIMS:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got: {axis!r}")

    along, r1, r2 = _AXIS_DIMS[axis]

    points = points.copy()

    cx = np.median(points[:, 0])
    cy = np.median(points[:, 1])
    cz = np.median(points[:, 2])

    points[:, 0] -= cx
    points[:, 1] -= cy
    points[:, 2] -= cz

    along_coords = points[:, along]

    a_low = np.quantile(along_coords, ignore_bottom_fraction)
    a_high = np.quantile(along_coords, 1.0 - ignore_top_fraction)

    slice_edges = np.linspace(a_low, a_high, num_slices + 1)
    angle_edges = np.linspace(0, 2 * np.pi, angle_bins + 1)

    roundness_scores = []
    wobble_scores = []
    slice_midpoints = []

    for i in range(num_slices):
        a0, a1 = slice_edges[i], slice_edges[i + 1]
        mask = (along_coords >= a0) & (along_coords < a1)
        slice_pts = points[mask]

        if len(slice_pts) < min_points_per_slice:
            continue

        s1 = slice_pts[:, r1]
        s2 = slice_pts[:, r2]

        radii = np.sqrt(s1**2 + s2**2)
        thetas = np.mod(np.arctan2(s2, s1), 2 * np.pi)

        bin_ids = np.clip(np.digitize(thetas, angle_edges) - 1, 0, angle_bins - 1)
        bin_radii = []
        for b in range(angle_bins):
            r_bin = radii[bin_ids == b]
            if len(r_bin) >= 2:
                bin_radii.append(np.median(r_bin))

        if len(bin_radii) < max(8, angle_bins // 6):
            continue

        bin_radii = np.array(bin_radii)
        mean_r = np.mean(bin_radii)

        if mean_r < 1e-8:
            continue

        roundness = np.std(bin_radii) / mean_r

        slice_c1 = np.median(s1)
        slice_c2 = np.median(s2)
        wobble = np.sqrt(slice_c1**2 + slice_c2**2) / mean_r

        roundness_scores.append(roundness)
        wobble_scores.append(wobble)
        slice_midpoints.append(0.5 * (a0 + a1))

    if len(roundness_scores) == 0:
        raise ValueError(
            "No valid slices found. Try lowering MIN_POINTS_PER_SLICE or NUM_SLICES."
        )

    roundness_scores = np.array(roundness_scores)
    wobble_scores = np.array(wobble_scores)
    slice_midpoints = np.array(slice_midpoints)

    mean_roundness = float(np.mean(roundness_scores))
    mean_wobble = float(np.mean(wobble_scores))
    combined_score = 0.7 * mean_roundness + 0.3 * mean_wobble

    return {
        "axis": axis,
        "global_center_original_coords": np.array([cx, cy, cz]),
        "mean_roundness_score": mean_roundness,
        "mean_axis_wobble_score": mean_wobble,
        "combined_symmetry_score": combined_score,
        "slice_along_values_centered": slice_midpoints,
        "slice_roundness_scores": roundness_scores,
        "slice_wobble_scores": wobble_scores,
        "num_valid_slices": len(roundness_scores),
    }


# ============================================================
# Visualization helpers
# ============================================================

def make_axis_line(points, axis="y", length_scale=1.2):
    """
    Create a red line showing the symmetry axis through the centered object.
    """
    along = _AXIS_DIMS[axis][0]
    a_min = np.min(points[:, along])
    a_max = np.max(points[:, along])
    center_a = 0.5 * (a_min + a_max)
    half_len = 0.5 * (a_max - a_min) * length_scale

    p0 = [0.0, 0.0, 0.0]
    p1 = [0.0, 0.0, 0.0]
    p0[along] = center_a - half_len
    p1[along] = center_a + half_len

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([p0, p1])
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])
    return line_set


def classify_score(score):
    """
    Rough interpretation for the combined score.
    """
    if score < 0.03:
        return "very symmetric"
    elif score < 0.06:
        return "reasonably symmetric"
    elif score < 0.10:
        return "noticeably asymmetric"
    else:
        return "strongly asymmetric"


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    points, geometry = load_as_point_cloud(MESH_PATH, NUM_SAMPLE_POINTS)

    result = compute_axis_symmetry(
        points,
        axis=SYMMETRY_AXIS,
        num_slices=NUM_SLICES,
        angle_bins=ANGLE_BINS,
        min_points_per_slice=MIN_POINTS_PER_SLICE,
        ignore_top_fraction=IGNORE_TOP_FRACTION,
        ignore_bottom_fraction=IGNORE_BOTTOM_FRACTION,
    )

    axis_label = SYMMETRY_AXIS.upper()
    print(f"\n=== Tomato Symmetry Around {axis_label}-Axis ===")
    print(f"File: {MESH_PATH}")
    print(f"Valid slices used: {result['num_valid_slices']}")
    print(f"Estimated center (original coords): {result['global_center_original_coords']}")
    print(f"Mean roundness score     : {result['mean_roundness_score']:.5f}")
    print(f"Mean axis wobble score   : {result['mean_axis_wobble_score']:.5f}")
    print(f"Combined symmetry score  : {result['combined_symmetry_score']:.5f}")
    print(f"Interpretation           : {classify_score(result['combined_symmetry_score'])}")

    if SHOW_VISUALIZATION:
        centered_points = points.copy()
        c = result["global_center_original_coords"]
        centered_points[:, 0] -= c[0]
        centered_points[:, 1] -= c[1]
        centered_points[:, 2] -= c[2]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(centered_points)
        pcd.paint_uniform_color([0.8, 0.2, 0.2])

        axis_line = make_axis_line(centered_points, axis=SYMMETRY_AXIS)

        o3d.visualization.draw_geometries(
            [pcd, axis_line],
            window_name=f"Tomato and candidate symmetry axis ({axis_label}-axis)",
        )
