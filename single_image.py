from datetime import datetime

import pyrealsense2 as rs
import numpy as np
import open3d as o3d

folder = "rgbd_photos/try1/"
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

profile = pipeline.start(config)

align = rs.align(rs.stream.color)

try:
    # Let auto-exposure stabilize
    for _ in range(30):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Get camera intrinsics
    intr = color_frame.profile.as_video_stream_profile().intrinsics

    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intr.width,
        intr.height,
        intr.fx,
        intr.fy,
        intr.ppx,
        intr.ppy
    )

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_image),
        o3d.geometry.Image(depth_image),
        depth_scale=1000.0,
        depth_trunc=2.0,
        convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        o3d_intrinsic
    )
    
    # Flip because Open3D camera coordinates are different
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    # Save and show point cloud
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    o3d.io.write_point_cloud(folder + f"tomato_single_view_{timestamp}.ply", pcd)
    print(f"Saved point cloud: {folder}tomato_single_view_{timestamp}.ply")

finally:
    pipeline.stop()