"""
Run the full tomato reconstruction pipeline on a folder of pre-fused,
colored turntable scans, chaining the individual pipeline scripts:

  1. segment_tomato_cone.py  -> per-shot tomato-only and cone-only clouds
  2. merge_tomato_clouds.py  -> cone-circle registered, merged cloud
     (or merge_stacked_levels.py when the folder holds several camera
     heights, i.e. files named bottom_*.ply and top_*.ply)
  3. mesh_merged_tomato_cloud.py -> final mesh

Each input *.ply is a complete, already-fused scan (e.g. from a
structured-light turntable scanner) with per-point RGB color baked in --
not a single-view RGBD capture. Every stage is driven off each file's own
stem, so any *.ply files in the input folder work; segment_tomato_cone.py
expects at least 3 of them (5 is typical) so merge_tomato_clouds.py can fit
a turntable circle through their cone-marker peaks.

Usage:
    python run_pipeline.py [input_folder] [output_folder] [--levels bottom top]

--levels lists filename prefixes of the camera heights, lowest first; each
level is registered on its own turntable circle and the levels are then
overlaid (0 in apart, see merge_stacked_levels.py). Without it, the stacked
merge is used automatically if the folder has files for every level in
DEFAULT_LEVELS, else the single-ring merge.

Defaults: input_folder=rgbd_photos/five_shots, output_folder=<input_folder>_output_<timestamp>

Each of the 3 underlying scripts can still be run standalone with their own
default input/output folders.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

INPUT_FOLDER = "rgbd_photos/top_and_bottom_view"
DEFAULT_LEVELS = ["bottom", "top"]

# The stacked multi-height cloud is less accurate than a single-ring scan, so
# it gets a more aggressive (lower-poly, smoother) final mesh.
STACKED_SIMPLIFY_TARGET_TRIANGLES = 15000
PLY_PATTERN = "*.ply"

SCRIPT_DIR = Path(__file__).resolve().parent


def run(script, *args):
    cmd = [sys.executable, str(SCRIPT_DIR / script), *(str(a) for a in args)]
    print(f"\n=== {script} {' '.join(str(a) for a in args)} ===")
    subprocess.run(cmd, check=True)


def main():
    args = sys.argv[1:]
    levels = None
    if "--levels" in args:
        i = args.index("--levels")
        j = i + 1
        while j < len(args) and not args[j].startswith("--"):
            j += 1
        levels = args[i + 1:j]
        del args[i:j]

    input_folder = Path(args[0]) if len(args) > 0 else Path(INPUT_FOLDER)
    if not input_folder.is_dir():
        raise SystemExit(f"Input folder not found: {input_folder}")

    if levels is None and all(any(input_folder.glob(f"{l}_*.ply")) for l in DEFAULT_LEVELS):
        levels = DEFAULT_LEVELS

    if len(args) > 1:
        output_folder = Path(args[1])
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = input_folder.parent / f"{input_folder.name}_output_{timestamp}"

    segmented_folder = output_folder / "segmented"
    merged_folder = output_folder / "merged"
    mesh_path = merged_folder / "tomato_mesh.ply"

    output_folder.mkdir(parents=True, exist_ok=True)

    run("segment_tomato_cone.py", input_folder, segmented_folder, PLY_PATTERN)
    if levels:
        run("merge_stacked_levels.py", segmented_folder, merged_folder, "--levels", *levels)
    else:
        run("merge_tomato_clouds.py", segmented_folder, merged_folder)
    mesh_args = [STACKED_SIMPLIFY_TARGET_TRIANGLES] if levels else []
    run("mesh_merged_tomato_cloud.py", merged_folder / "tomato_merged.ply", mesh_path, *mesh_args)

    print(f"\nPipeline complete -> {output_folder}")


if __name__ == "__main__":
    main()
