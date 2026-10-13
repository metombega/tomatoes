"""
Merge turntable scans taken at several camera heights ("levels", e.g. a
bottom ring of 5 shots and a top ring of 5 shots) into one tomato cloud.

Input is the output of segment_tomato_cone.py: <level>_<shot>_tomato.ply /
<level>_<shot>_cone.ply pairs, e.g. bottom_b_tomato.ply, top_g_cone.ply. The
number of levels and of shots per level is flexible (>= 3 shots per level so
a circle can be fitted).

  1. Per level, exactly as merge_tomato_clouds.py: fit a 3D circle through
     that level's cone peaks and rotate every shot about the fitted axis so
     they all sit at the level's reference shot (first by name).
  2. Stack the levels: the first level given is the reference frame. Every
     other level is rigidly moved so its circle lies on top of the previous
     level's circle -- same axis direction, same angular phase (reference
     shots' cone peaks line up), and its center GAP_INCHES further along the
     axis ("up" = the side of the circle the tomato sits on).
  3. Concatenate, voxel downsample and clean outliers.

Usage:
    python merge_stacked_levels.py [input_folder] [output_folder]
        [--levels bottom top] [--gap-inches 0] [--flip]

--levels lists level filename prefixes from lowest to highest. --flip stacks
in the opposite axis direction if "up" came out the wrong way.
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from merge_tomato_clouds import (
    FINAL_VOXEL_SIZE, OUTLIER_NB_NEIGHBORS, OUTLIER_STD_RATIO,
    RADIUS_OUTLIER_NB_POINTS, RADIUS_OUTLIER_RADIUS,
    angle_around_circle, cone_peak, fit_circle_3d, rotate_about_axis,
)

INPUT_FOLDER = "rgbd_photos/top_and_bottom_view_segmented"
OUTPUT_FOLDER = "rgbd_photos/top_and_bottom_view_merged"
DEFAULT_LEVELS = ["bottom", "top"]
GAP_INCHES = 0.0
INCH_TO_M = 0.0254


def load_level(folder, level):
    stems = sorted(p.name[: -len("_tomato.ply")] for p in folder.glob(f"{level}_*_tomato.ply"))
    if len(stems) < 3:
        raise SystemExit(f"Level {level!r}: need at least 3 *_tomato.ply/_cone.ply pairs in {folder}, found {len(stems)}")
    shots = []
    for stem in stems:
        tomato = o3d.io.read_point_cloud(str(folder / f"{stem}_tomato.ply"))
        cone = o3d.io.read_point_cloud(str(folder / f"{stem}_cone.ply"))
        shots.append({"stem": stem, "tomato": tomato, "cone": cone, "peak": cone_peak(cone)})
    return shots


def register_level(shots, label):
    """Fit the level's circle, orient its axis toward the tomato, rotate all
    shots onto the first shot's turntable angle. Returns the level's cloud
    (all shots concatenated, still in the scanner frame) and the circle
    description."""
    peaks = np.array([s["peak"] for s in shots])
    center, radius, normal, e1, _ = fit_circle_3d(peaks)

    tomato_center = np.vstack([np.asarray(s["tomato"].points) for s in shots]).mean(axis=0)
    if (tomato_center - center) @ normal < 0:
        normal = -normal
    e1 = e1 - (e1 @ normal) * normal
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)

    radial = np.linalg.norm(np.stack([(peaks - center) @ e1, (peaks - center) @ e2], axis=1), axis=1) - radius
    print(f"\n[{label}] circle center {center.round(4)}, radius {radius * 1000:.2f} mm, up-axis {normal.round(3)}")
    print(f"[{label}] out-of-plane residuals (mm): {(((peaks - center) @ normal) * 1000).round(2)}, "
          f"radial residuals (mm): {(radial * 1000).round(2)}")

    angles = np.array([angle_around_circle(p, center, e1, e2) for p in peaks])
    for s, angle in zip(shots, angles):
        rotate_about_axis(s["tomato"], normal, center, angles[0] - angle)
        print(f"[{label}] {s['stem']}: rotated {np.degrees(angles[0] - angle):+.2f} deg")

    cloud = o3d.geometry.PointCloud()
    for s in shots:
        cloud += s["tomato"]
    return cloud, {"center": center, "radius": radius, "normal": normal, "e1": e1, "e2": e2,
                   "ref_angle": angles[0], "peaks": peaks}


def frame(circle):
    """Right-handed (u, v, n) frame whose u points at the level's reference
    cone peak (the shared angular phase across levels)."""
    n, a = circle["normal"], circle["ref_angle"]
    u = np.cos(a) * circle["e1"] + np.sin(a) * circle["e2"]
    return np.stack([u, np.cross(n, u), n], axis=1)


def stack_onto(cloud, circle, target, gap_m):
    """Rigidly move `cloud` so `circle` lies on `target`'s circle, raised by
    `gap_m` along the target's up-axis, with matching angular phase."""
    R = frame(target) @ frame(circle).T
    new_center = target["center"] + gap_m * target["normal"]
    cloud.rotate(R, center=circle["center"])
    cloud.translate(new_center - circle["center"])
    return {"center": new_center, "radius": circle["radius"], "normal": target["normal"],
            "e1": target["e1"], "e2": target["e2"], "ref_angle": target["ref_angle"]}


def main():
    args = sys.argv[1:]
    flip = "--flip" in args
    if flip:
        args.remove("--flip")
    gap_inches, levels = GAP_INCHES, DEFAULT_LEVELS
    if "--gap-inches" in args:
        i = args.index("--gap-inches")
        gap_inches = float(args[i + 1])
        del args[i:i + 2]
    if "--levels" in args:
        i = args.index("--levels")
        j = i + 1
        while j < len(args) and not args[j].startswith("--"):
            j += 1
        levels = args[i + 1:j]
        del args[i:j]

    input_folder = Path(args[0]) if len(args) > 0 else Path(INPUT_FOLDER)
    output_folder = Path(args[1]) if len(args) > 1 else Path(OUTPUT_FOLDER)
    output_folder.mkdir(parents=True, exist_ok=True)

    gap_m = gap_inches * INCH_TO_M
    full = o3d.geometry.PointCloud()
    prev = None
    for k, level in enumerate(levels):
        cloud, circle = register_level(load_level(input_folder, level), label=level)
        if k == 0:
            prev = circle
        else:
            prev = stack_onto(cloud, circle, prev, -gap_m if flip else gap_m)
            print(f"[{level}] stacked {gap_inches} in {'against' if flip else 'along'} the up-axis, "
                  f"circle center now {prev['center'].round(4)}")
        full += cloud

    full_path = output_folder / "tomato_merged_full.ply"
    o3d.io.write_point_cloud(str(full_path), full)
    print(f"\nStacked cloud (every aligned point): {len(full.points)} points -> {full_path}")

    merged = full.voxel_down_sample(FINAL_VOXEL_SIZE)
    merged, _ = merged.remove_statistical_outlier(nb_neighbors=OUTLIER_NB_NEIGHBORS, std_ratio=OUTLIER_STD_RATIO)
    merged, _ = merged.remove_radius_outlier(nb_points=RADIUS_OUTLIER_NB_POINTS, radius=RADIUS_OUTLIER_RADIUS)
    out_path = output_folder / "tomato_merged.ply"
    o3d.io.write_point_cloud(str(out_path), merged)
    print(f"Merged cloud (downsampled + cleaned): {len(merged.points)} points -> {out_path}")


if __name__ == "__main__":
    main()
