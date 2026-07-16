"""
Run the full pear reconstruction pipeline on a folder of single-view RGBD
.ply captures, chaining the individual pipeline scripts:

  1. ply_to_image.py               -> RGB renders
  2. rembg_detect.py                -> pear masks/cutouts
  3. center_pear_clouds.py          -> centered, pear-only clouds
  4. merge_centered_pear_clouds.py  -> merged cloud
  5. mesh_merged_pear_cloud.py      -> final mesh

Input .ply files don't need any specific naming -- every stage is driven off
each file's own stem, so any *.ply files in the input folder work.

Usage:
    python run_pipeline.py [input_folder] [output_folder]

Defaults: input_folder=rgbd_photos/try10, output_folder=<input_folder>_output_<timestamp>

Each of the 5 underlying scripts can still be run standalone with their own
default input/output folders.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

INPUT_FOLDER = "rgbd_photos/try10"
PLY_PATTERN = "*.ply"

SCRIPT_DIR = Path(__file__).resolve().parent


def run(script, *args):
    cmd = [sys.executable, str(SCRIPT_DIR / script), *(str(a) for a in args)]
    print(f"\n=== {script} {' '.join(str(a) for a in args)} ===")
    subprocess.run(cmd, check=True)


def main():
    input_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(INPUT_FOLDER)
    if not input_folder.is_dir():
        raise SystemExit(f"Input folder not found: {input_folder}")

    if len(sys.argv) > 2:
        output_folder = Path(sys.argv[2])
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = input_folder.parent / f"{input_folder.name}_output_{timestamp}"

    rgb_folder = output_folder / "rgb"
    centered_folder = output_folder / "centered"
    merged_folder = output_folder / "merged"
    mesh_path = merged_folder / "tomato_mesh.ply"

    output_folder.mkdir(parents=True, exist_ok=True)

    run("ply_to_image.py", input_folder, rgb_folder, PLY_PATTERN)
    run("rembg_detect.py", rgb_folder)
    run("center_pear_clouds.py", input_folder, rgb_folder, centered_folder, PLY_PATTERN)
    run("merge_centered_pear_clouds.py", centered_folder, merged_folder, PLY_PATTERN)
    run("mesh_merged_pear_cloud.py", merged_folder / "tomato_merged_before_tweak.ply", mesh_path)

    print(f"\nPipeline complete -> {output_folder}")


if __name__ == "__main__":
    main()
