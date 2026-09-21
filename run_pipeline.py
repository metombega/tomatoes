"""
Run the full tomato reconstruction pipeline on a folder of pre-fused,
colored turntable scans, chaining the individual pipeline scripts:

  1. segment_tomato_cone.py  -> per-shot tomato-only and cone-only clouds
  2. merge_tomato_clouds.py  -> cone-circle registered, merged cloud
  3. mesh_merged_tomato_cloud.py -> final mesh

Each input *.ply is a complete, already-fused scan (e.g. from a
structured-light turntable scanner) with per-point RGB color baked in --
not a single-view RGBD capture. Every stage is driven off each file's own
stem, so any *.ply files in the input folder work; segment_tomato_cone.py
expects at least 3 of them (5 is typical) so merge_tomato_clouds.py can fit
a turntable circle through their cone-marker peaks.

Usage:
    python run_pipeline.py [input_folder] [output_folder]

Defaults: input_folder=rgbd_photos/five_shots, output_folder=<input_folder>_output_<timestamp>

Each of the 3 underlying scripts can still be run standalone with their own
default input/output folders.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

INPUT_FOLDER = "rgbd_photos/video_on_ratation_table"
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

    segmented_folder = output_folder / "segmented"
    merged_folder = output_folder / "merged"
    mesh_path = merged_folder / "tomato_mesh.ply"

    output_folder.mkdir(parents=True, exist_ok=True)

    run("segment_tomato_cone.py", input_folder, segmented_folder, PLY_PATTERN)
    run("merge_tomato_clouds.py", segmented_folder, merged_folder)
    run("mesh_merged_tomato_cloud.py", merged_folder / "tomato_merged.ply", mesh_path)

    print(f"\nPipeline complete -> {output_folder}")


if __name__ == "__main__":
    main()
