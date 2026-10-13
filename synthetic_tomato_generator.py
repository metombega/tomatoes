import numpy as np
import open3d as o3d


def _hsv_to_rgb(h, s, v):
    """
    Vectorized HSV -> RGB conversion (h, s, v in [0, 1], any shape).
    """
    h = np.mod(h, 1.0)
    s = np.clip(s, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)

    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i_mod = i.astype(int) % 6

    conditions = [i_mod == k for k in range(6)]
    r = np.select(conditions, [v, q, p, p, t, v])
    g = np.select(conditions, [t, v, v, q, p, p])
    b = np.select(conditions, [p, p, t, v, v, q])

    return np.stack([r, g, b], axis=-1)


def _tomato_vertex_colors(theta_per_vertex, phi_per_vertex, rng, ripeness, consistency, richness):
    """
    Generates per-vertex tomato colors from three human-meaningful sliders:

    - ripeness (0..1):    green -> red (passes naturally through yellow/orange,
                           like a real tomato ripening)
    - consistency (0..1): 1.0 = even color all over; lower values create
                           blotchy patches where local ripeness drifts from
                           the target (unevenly ripened look)
    - richness (0..1):    0.0 = dull/pale/washed-out color; 1.0 = deep,
                           saturated color
    """

    ripeness = np.clip(ripeness, 0.0, 1.0)
    consistency = np.clip(consistency, 0.0, 1.0)
    richness = np.clip(richness, 0.0, 1.0)

    n = len(theta_per_vertex)
    blotchiness = 1.0 - consistency

    # Smooth, blob-like spatial noise built from a handful of random
    # sinusoids over the (theta, phi) surface -> looks like patches, not static.
    num_blotches = rng.integers(3, 7)
    blotch = np.zeros(n)
    for _ in range(num_blotches):
        freq_theta = rng.uniform(1, 4)
        freq_phi = rng.uniform(1, 4)
        phase_theta = rng.uniform(0, 2 * np.pi)
        phase_phi = rng.uniform(0, 2 * np.pi)
        blotch += np.sin(freq_theta * theta_per_vertex + phase_theta) * np.sin(
            freq_phi * phi_per_vertex + phase_phi
        )
    blotch /= num_blotches  # roughly in [-1, 1]

    local_ripeness = np.clip(ripeness + blotchiness * 0.6 * blotch, 0.0, 1.0)

    # Hue: green (~0.30) -> red (~0.0), sweeping through yellow/orange on the way.
    hue = 0.30 * (1.0 - local_ripeness)

    # Richness sets the overall saturation/value target (dull vs. rich),
    # with a touch of natural per-vertex jitter plus extra blotch-driven
    # unevenness when consistency is low.
    jitter = rng.uniform(-0.03, 0.03, n)
    saturation = 0.35 + 0.55 * richness + jitter + blotchiness * 0.15 * blotch
    value = 0.45 + 0.25 * (1.0 - richness) + jitter - blotchiness * 0.10 * blotch

    return _hsv_to_rgb(hue, saturation, value)


def create_random_tomato_mesh(
    seed=None,
    n_theta=160,
    n_phi=100,
    radius=1.0,
    height_scale=0.85,
    noise_strength=0.045,
    lobe_strength=0.08,
    stem_indent_strength=0.18,
    ripeness=1.0,
    consistency=1.0,
    richness=1.0,
):
    """
    Creates a random tomato-like triangle mesh.

    Coordinate system:
    - y is the vertical axis
    - x-z is the horizontal cross-section

    Color sliders (all 0..1):
    - ripeness: green -> red
    - consistency: 1.0 = even color, lower = blotchy/uneven patches
    - richness: 0.0 = dull/pale, 1.0 = deep/saturated
    """

    rng = np.random.default_rng(seed)

    vertices = []
    triangles = []

    # theta: around vertical y-axis
    # phi: top to bottom
    theta_vals = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi_vals = np.linspace(0, np.pi, n_phi)

    # Random tomato parameters
    lobe_count = rng.integers(5, 9)
    lobe_phase = rng.uniform(0, 2 * np.pi)

    asym_x = rng.uniform(-0.08, 0.08)
    asym_z = rng.uniform(-0.08, 0.08)
    
    for i, phi in enumerate(phi_vals):
        # y goes from top to bottom
        y = height_scale * np.cos(phi)

        # base sphere radius at this height
        base_r = radius * np.sin(phi)

        # Make tomato slightly flattened top/bottom
        flatten = 1.0 - 0.10 * np.cos(2 * phi)

        # Stem indentation near top
        top_factor = np.exp(-((phi - 0.18) ** 2) / 0.018)
        stem_indent = 1.0 - stem_indent_strength * top_factor

        # Slight bottom bulge
        bottom_bulge = 1.0 + 0.07 * np.exp(-((phi - 2.55) ** 2) / 0.08)

        for j, theta in enumerate(theta_vals):
            # Lobes around the tomato
            lobes = 1.0 + lobe_strength * np.sin(lobe_count * theta + lobe_phase)

            # Smooth random-looking surface variation
            noise = (
                noise_strength * np.sin(3 * theta + rng.uniform(-0.2, 0.2)) * np.sin(2 * phi)
                + noise_strength * 0.5 * np.sin(7 * theta + 1.5 * phi)
            )

            r = base_r * flatten * stem_indent * bottom_bulge * lobes
            r *= 1.0 + noise

            x = r * np.cos(theta)
            z = r * np.sin(theta)

            # Small global asymmetry
            x += asym_x * np.sin(phi) ** 2
            z += asym_z * np.sin(phi) ** 2

            vertices.append([x, y, z])

    # Create triangle faces
    for i in range(n_phi - 1):
        for j in range(n_theta):
            current = i * n_theta + j
            next_theta = i * n_theta + (j + 1) % n_theta
            below = (i + 1) * n_theta + j
            below_next = (i + 1) * n_theta + (j + 1) % n_theta

            triangles.append([current, below, next_theta])
            triangles.append([next_theta, below, below_next])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    # Tomato vertex colors driven by ripeness / consistency / richness
    phi_per_vertex = np.repeat(phi_vals, n_theta)
    theta_per_vertex = np.tile(theta_vals, n_phi)
    colors = _tomato_vertex_colors(
        theta_per_vertex, phi_per_vertex, rng, ripeness, consistency, richness
    )
    colors = np.clip(colors, 0, 1)

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return mesh


def add_simple_stem(mesh, seed=None):
    """
    Adds a small green stem on top of the tomato.
    """

    rng = np.random.default_rng(seed)

    stem_radius = 0.08
    stem_height = 0.28
    stem_resolution = 24

    stem = o3d.geometry.TriangleMesh.create_cylinder(
        radius=stem_radius,
        height=stem_height,
        resolution=stem_resolution,
    )

    # Open3D cylinder is along z by default, so rotate it to align with y-axis
    R = stem.get_rotation_matrix_from_xyz([np.pi / 2, 0, 0])
    stem.rotate(R, center=(0, 0, 0))

    # Move stem to tomato top
    stem.translate([0, 0.85 + stem_height / 2 - 0.03, 0])

    # Slight random tilt
    tilt_x = rng.uniform(-0.25, 0.25)
    tilt_z = rng.uniform(-0.25, 0.25)
    R_tilt = stem.get_rotation_matrix_from_xyz([tilt_x, 0, tilt_z])
    stem.rotate(R_tilt, center=(0, 0.85, 0))

    stem.paint_uniform_color([0.1, 0.45, 0.08])
    stem.compute_vertex_normals()

    combined = mesh + stem
    combined.compute_vertex_normals()

    return combined


if __name__ == "__main__":
    seed = None  # Change to an integer, for example 7, to get repeatable results

    tomato_mesh = create_random_tomato_mesh(
        seed=seed,
        n_theta=180,
        n_phi=120,
        radius=1.0,
        height_scale=1.5,
        noise_strength=0.0,#0.035,
        lobe_strength=0.0, #0.055,
        stem_indent_strength=0.22,
        ripeness=0.9,     # 0.0 = green, 1.0 = red
        consistency=0.75, # 0.0 = blotchy/uneven, 1.0 = uniform color
        richness=0.8,    # 0.0 = dull, 1.0 = rich/saturated
    )

    # tomato_mesh = add_simple_stem(tomato_mesh, seed=seed)

    # Optional smoothing
    tomato_mesh = tomato_mesh.filter_smooth_taubin(number_of_iterations=8)
    tomato_mesh.compute_vertex_normals()

    # Save files
    o3d.io.write_triangle_mesh("random_tomato_mesh.ply", tomato_mesh)
    o3d.io.write_triangle_mesh("random_tomato_mesh.obj", tomato_mesh)

    print("Saved:")
    print("random_tomato_mesh.ply")
    print("random_tomato_mesh.obj")

    # Show mesh
    o3d.visualization.draw_geometries([tomato_mesh])