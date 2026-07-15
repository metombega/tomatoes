import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

os.makedirs("photos", exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        cv2.imshow("RGB", color_image)
        cv2.imshow("Depth", depth_colormap)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            rgb_path = f"photos/rgb_{timestamp}.png"
            depth_path = f"photos/depth_raw_{timestamp}.png"
            depth_view_path = f"photos/depth_view_{timestamp}.png"

            cv2.imwrite(rgb_path, color_image)

            # Saves real raw depth values, not just a visual image
            cv2.imwrite(depth_path, depth_image)

            # Saves the colored visualization
            cv2.imwrite(depth_view_path, depth_colormap)

            print("Saved:")
            print(rgb_path)
            print(depth_path)
            print(depth_view_path)

        elif key == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()