"""
Render each colored point cloud in a folder back into an RGB PNG.

These single-view .ply files are built by deprojecting an RGBD frame with the
camera at the world origin (see single_image.py), so placing a virtual camera
back at the origin and looking at the cloud reproduces the original shot.

Usage:
    python ply_to_image.py [input_folder] [output_folder] [pattern]

Defaults convert rgbd_photos/try10/tomato_single_view_*.ply to
rgbd_photos/try10_rgb/tomato_single_view_*.png
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d

INPUT_FOLDER = "rgbd_photos/try10"
OUTPUT_FOLDER = "rgbd_photos/try10_rgb"
PATTERN = "tomato_single_view_*.ply"

WIDTH, HEIGHT = 640, 480
POINT_SIZE = 3.0
BACKGROUND_COLOR = (0.0, 0.0, 0.0)


def render_to_image(pcd, output_path, width=WIDTH, height=HEIGHT):
    points = np.asarray(pcd.points)
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

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    try:
        vis.add_geometry(pcd)

        opt = vis.get_render_option()
        opt.background_color = np.array(BACKGROUND_COLOR)
        opt.point_size = POINT_SIZE

        ctr = vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()
        params.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(output_path), do_render=True)
    finally:
        vis.destroy_window()


def main():
    input_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(INPUT_FOLDER)
    output_folder = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(OUTPUT_FOLDER)
    pattern = sys.argv[3] if len(sys.argv) > 3 else PATTERN

    files = sorted(input_folder.glob(pattern))
    if not files:
        raise SystemExit(f"No files matching {pattern!r} found in {input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)

    for ply_path in files:
        pcd = o3d.io.read_point_cloud(str(ply_path))
        if len(pcd.points) == 0:
            print(f"Skipping empty cloud: {ply_path.name}")
            continue

        png_path = output_folder / (ply_path.stem + ".png")
        render_to_image(pcd, png_path)
        print(f"{ply_path.name} -> {png_path}")


if __name__ == "__main__":
    main()
