import numpy as np
import open3d as o3d


def create_random_tomato_mesh(
    seed=None,
    n_theta=160,
    n_phi=100,
    radius=1.0,
    height_scale=0.85,
    noise_strength=0.045,
    lobe_strength=0.08,
    stem_indent_strength=0.18,
):
    """
    Creates a random tomato-like triangle mesh.

    Coordinate system:
    - y is the vertical axis
    - x-z is the horizontal cross-section
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

    # Tomato red-ish vertex colors
    colors = np.zeros((len(vertices), 3))
    colors[:, 0] = 0.85 + rng.uniform(-0.05, 0.05, len(vertices))  # red
    colors[:, 1] = 0.12 + rng.uniform(-0.03, 0.03, len(vertices))  # green
    colors[:, 2] = 0.08 + rng.uniform(-0.02, 0.02, len(vertices))  # blue
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