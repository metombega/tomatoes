"""
Build centered, pear-only point clouds from single-view RGBD captures.

Pipeline per rgbd_photos/try10/tomato_single_view_*.ply file:
1. RANSAC-fit the dominant plane (the box/table surface) and compute the
   small rotation that levels its normal onto the Z-axis, i.e. makes the
   camera parallel to the table. Skip the file if that correction would
   exceed MAX_LEVEL_ANGLE_DEG -- that means RANSAC likely locked onto the
   wrong surface.
2. Reproject every point with the same virtual camera ply_to_image.py used
   to render rgbd_photos/try10_rgb/*.png, and keep only the points that
   land on the pear mask in rgbd_photos/try10_rgb/rembg_out/*_mask.png.
3. Apply the leveling rotation to the pear-only points and recenter them so
   the pear's centroid sits at the origin.

Usage:
    python center_pear_clouds.py [input_folder] [rgb_folder] [output_folder]

Defaults convert rgbd_photos/try10/tomato_single_view_*.ply to
rgbd_photos/try10_centered/tomato_single_view_*.ply, using masks from
rgbd_photos/try10_rgb/rembg_out/.
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

INPUT_FOLDER = "rgbd_photos/try10"
RGB_FOLDER = "rgbd_photos/try10_rgb"
OUTPUT_FOLDER = "rgbd_photos/try10_centered"
PATTERN = "tomato_single_view_*.ply"

WIDTH, HEIGHT = 640, 480
PLANE_DISTANCE_THRESHOLD = 0.01
MAX_LEVEL_ANGLE_DEG = 5.0
MASK_THRESHOLD = 128


def default_render_intrinsic(width=WIDTH, height=HEIGHT):
    """
    Same default pinhole intrinsic Open3D's Visualizer falls back to. This
    is what ply_to_image.py implicitly rendered rgbd_photos/try10_rgb/*.png
    with, so points reprojected through it line up with the rembg masks.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    ctr = vis.get_view_control()
    intrinsic = ctr.convert_to_pinhole_camera_parameters().intrinsic.intrinsic_matrix.copy()
    vis.destroy_window()
    return intrinsic


def render_extrinsic(points):
    """
    Same eye-at-origin, look-at-centroid camera as ply_to_image.py's
    render_to_image, needed to reproduce the pixel grid the masks were
    computed on.
    """
    center = points.mean(axis=0)
    eye = np.zeros(3)
    forward = center - eye
    forward /= np.linalg.norm(forward)
    up_hint = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = np.stack([right, -true_up, forward], axis=0)
    extrinsic[:3, 3] = -extrinsic[:3, :3] @ eye
    return extrinsic


def project_points(points, intrinsic, extrinsic):
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    cam = points @ R.T + t
    z = cam[:, 2]

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    u = cam[:, 0] * fx / z + cx
    v = cam[:, 1] * fy / z + cy
    return u, v, z


def pear_mask_indices(points, mask, intrinsic):
    """Indices of points that land inside the pear mask when reprojected
    through the same virtual camera used to render the source PNGs."""
    extrinsic = render_extrinsic(points)
    u, v, z = project_points(points, intrinsic, extrinsic)

    height, width = mask.shape
    u_i = np.round(u).astype(int)
    v_i = np.round(v).astype(int)
    in_bounds = (z > 0) & (u_i >= 0) & (u_i < width) & (v_i >= 0) & (v_i < height)

    on_mask = np.zeros(len(points), dtype=bool)
    on_mask[in_bounds] = mask[v_i[in_bounds], u_i[in_bounds]] > MASK_THRESHOLD
    return np.where(on_mask)[0]


def leveling_rotation(pcd, distance_threshold=PLANE_DISTANCE_THRESHOLD):
    """Rotation that levels the dominant plane's (the table's) normal onto
    the Z-axis. Returns (R, angle_deg)."""
    plane_model, _ = pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=3, num_iterations=2000
    )
    normal = np.array(plane_model[:3])
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal

    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, target)
    axis_norm = np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(normal, target), -1.0, 1.0))

    if axis_norm < 1e-8:
        return np.eye(3), np.degrees(angle)

    axis = axis / axis_norm
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
    return R, np.degrees(angle)


def process_file(ply_path, mask_path, intrinsic):
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        print(f"Skipping empty cloud: {ply_path.name}")
        return None

    R_level, angle_deg = leveling_rotation(pcd)
    if angle_deg > MAX_LEVEL_ANGLE_DEG:
        print(
            f"Skipping {ply_path.name}: table-leveling correction of "
            f"{angle_deg:.2f} deg exceeds {MAX_LEVEL_ANGLE_DEG} deg "
            f"(RANSAC likely found the wrong plane)"
        )
        return None

    mask = np.array(Image.open(mask_path))
    points = np.asarray(pcd.points)

    pear_idx = pear_mask_indices(points, mask, intrinsic)
    if len(pear_idx) == 0:
        print(f"Skipping {ply_path.name}: no points matched the pear mask")
        return None

    pear = pcd.select_by_index(pear_idx)
    pear, _ = pear.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    if len(pear.points) == 0:
        print(f"Skipping {ply_path.name}: no pear points left after outlier removal")
        return None

    pear.rotate(R_level, center=(0, 0, 0))
    pear.translate(-pear.get_center())

    print(
        f"{ply_path.name}: level correction {angle_deg:.2f} deg, "
        f"{len(pear_idx)} pear points -> {len(pear.points)} after cleanup"
    )
    return pear


def main():
    input_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(INPUT_FOLDER)
    rgb_folder = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(RGB_FOLDER)
    output_folder = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(OUTPUT_FOLDER)
    mask_folder = rgb_folder / "rembg_out"

    files = sorted(input_folder.glob(PATTERN))
    if not files:
        raise SystemExit(f"No files matching {PATTERN!r} found in {input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)
    intrinsic = default_render_intrinsic()

    for ply_path in files:
        mask_path = mask_folder / f"{ply_path.stem}_mask.png"
        if not mask_path.exists():
            print(f"Skipping {ply_path.name}: no mask found at {mask_path}")
            continue

        pear = process_file(ply_path, mask_path, intrinsic)
        if pear is None:
            continue

        out_path = output_folder / ply_path.name
        o3d.io.write_point_cloud(str(out_path), pear)
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
