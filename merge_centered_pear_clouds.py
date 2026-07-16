"""
Merge the 4 centered single-view pear point clouds in
rgbd_photos/try10_centered/ (captured at nominal 0/90/180/270 degree
turntable steps) into one point cloud.

Each input cloud is already centered at the origin, but on ITS OWN
visible-surface centroid (see center_pear_clouds.py) -- not on a shared
"true center of the pear". Since a single view only sees a bulging cap of
the surface, that cap's centroid sits offset from the pear's real center
along the camera's viewing axis. To correct for this, each view is first
shifted PRE_ROTATION_CAMERA_SHIFT toward the camera (in its own local
per-file frame, before the shared-frame nominal rotation is applied).
Rotating each file's points about the shared world origin is still only
correct in orientation, though -- the remaining offset is absorbed by a
translation correction during the tweak step below.

The turntable's rotation angle itself, on the other hand, is known well
(nominal 0/90/180/270 deg, +-5 deg at most) -- so the rotation applied here
must stay close to that nominal value. Letting a generic 6-DOF ICP solve
both at once is what went wrong the first time: with neighboring views
sharing only a thin sliver of true overlap, ICP found "matches" by
snapping one view's cap ~90 deg onto its neighbor's cap instead of sliding
it a few mm/degrees into place, collapsing all 4 views onto roughly the
same surface.

To avoid that, the correction here is solved as two bounded steps,
alternated a couple of times:
  1. translation-only alignment (nearest-neighbor centroid matching), with
     no limit on distance -- this is where the ~1 cm centering offset gets
     absorbed.
  2. rotation-only alignment, via a grid search restricted to
     +-MAX_TWEAK_DEG around the nominal angle -- this can never run away
     to a wrong 90 deg lock-on.

Since the rotation direction (sign) of the turntable isn't known a priori,
both signs are tried and the one that produces better alignment is kept.

Usage:
    python merge_centered_pear_clouds.py [input_folder] [output_folder] [pattern]

Defaults convert rgbd_photos/try10_centered/tomato_single_view_*.ply into
rgbd_photos/try10_merged/, writing:
    tomato_merged_before_tweak.ply     - nominal rotation only, no tweak
    tomato_merged.ply                  - after the bounded tweak
    before_tweak.png / after_tweak.png - diagnostic renders
"""

import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

INPUT_FOLDER = "rgbd_photos/try10_centered"
OUTPUT_FOLDER = "rgbd_photos/try10_merged"
PATTERN = "tomato_single_view_*.ply"

NOMINAL_ANGLES_DEG = [0, 270, 180, 90]
ROTATION_AXIS = np.array([0.0, 1.0, 0.0])  # vertical axis in these clouds

# Each file's own local "toward the camera" direction, in its own (per-file)
# leveled frame -- measured by replaying center_pear_clouds.py's leveling
# rotation on the raw try10 captures and checking where the camera (origin)
# ends up relative to the pear-only centroid. Very close to +Z, with a small
# +Y tilt from the table-leveling correction.
CAMERA_DIRECTION = np.array([0.01715, 0.09554, 0.99524])
CAMERA_DIRECTION = CAMERA_DIRECTION / np.linalg.norm(CAMERA_DIRECTION)

# Each view was centered on its own visible-surface centroid, which sits
# slightly off the pear's true center along the camera's viewing axis.
# Shift every view a bit toward the camera, in its own local frame, before
# applying the (shared-frame) nominal turntable rotation.
PRE_ROTATION_CAMERA_SHIFT = 0.025

# Rotation correction is never allowed to stray outside this range around
# the nominal turntable angle.
MAX_TWEAK_DEG = 6.0
ROTATION_SEARCH_STEP_DEG = 0.5

# Correspondence threshold for the translation-only search. The bulk of the
# per-view centering offset is now handled by PRE_ROTATION_CAMERA_SHIFT, so
# this only needs to cover the leftover residual, not the full ~1 cm offset
# -- too generous a threshold lets the tweak slide onto the wrong part of
# the (mostly non-overlapping) neighboring view and warp a good nominal
# alignment into a worse one.
TRANSLATION_SEARCH_THRESHOLD = 0.005
# Tighter threshold used to score/compare candidate rotations and to report
# final alignment quality.
FINE_EVAL_THRESHOLD = 0.0018

ALTERNATIONS = 3  # translate -> rotate -> translate -> rotate -> ...

FINAL_VOXEL_SIZE = 0.0012
OUTLIER_NB_NEIGHBORS = 30
OUTLIER_STD_RATIO = 1.5
RADIUS_OUTLIER_NB_POINTS = 8
RADIUS_OUTLIER_RADIUS = 0.003


def shift_toward_camera(pcd, distance=PRE_ROTATION_CAMERA_SHIFT):
    pcd = copy.deepcopy(pcd)
    pcd.translate(CAMERA_DIRECTION * distance)
    return pcd


def rotate_around_axis(pcd, angle_deg, axis=ROTATION_AXIS, center=(0.0, 0.0, 0.0)):
    pcd = copy.deepcopy(pcd)
    R = pcd.get_rotation_matrix_from_axis_angle(axis * np.deg2rad(angle_deg))
    pcd.rotate(R, center=center)
    return pcd


def y_rotation_matrix(angle_deg):
    return o3d.geometry.get_rotation_matrix_from_axis_angle(ROTATION_AXIS * np.deg2rad(angle_deg))


def best_translation(points, target_tree, target_pts, threshold, max_iter=50, tol=1e-7):
    """Nearest-neighbor centroid-matching translation search (translation-only
    ICP): repeatedly find correspondences within `threshold` and shift by
    their mean offset, with no cap on the total distance moved."""
    t = np.zeros(3)
    for _ in range(max_iter):
        query = points + t
        dists, idx = target_tree.query(query, k=1)
        mask = dists < threshold
        if mask.sum() < 15:
            break
        delta = (target_pts[idx[mask]] - query[mask]).mean(axis=0)
        t = t + delta
        if np.linalg.norm(delta) < tol:
            break
    return t


def score_alignment(points, target_tree, threshold):
    dists, _ = target_tree.query(points, k=1)
    inliers = dists < threshold
    if inliers.sum() == 0:
        return 0.0, np.inf
    return inliers.mean(), dists[inliers].mean()


def best_rotation_deg(points, t, target_tree, angle_range, step, threshold):
    """Grid search over small rotation deltas (translation held fixed),
    scored by inlier fraction at `threshold`. Bounded by construction --
    can never return more than `angle_range` deg."""
    best_angle, best_score = 0.0, -1.0
    for delta in np.arange(-angle_range, angle_range + 1e-9, step):
        R = y_rotation_matrix(delta)
        candidate = points @ R.T + t
        fitness, mean_dist = score_alignment(candidate, target_tree, threshold)
        score = fitness - 0.001 * mean_dist  # fitness first, distance as tiebreaker
        if score > best_score:
            best_score, best_angle = score, delta
    return best_angle


def bounded_tweak(source, target):
    """Alternate translation-only and rotation-only (bounded) alignment.
    Returns a 4x4 transform to apply to `source` (already at its nominal
    rotation) plus the total rotation delta applied, for reporting."""
    source_pts = np.asarray(source.points)
    target_pts = np.asarray(target.points)
    target_tree = cKDTree(target_pts)

    angle_total = 0.0
    t = best_translation(source_pts, target_tree, target_pts, TRANSLATION_SEARCH_THRESHOLD)

    # Each rotation search is re-run from the ORIGINAL nominal points (not the
    # already-rotated ones), so the total offset from nominal always stays
    # within +-MAX_TWEAK_DEG no matter how many alternations run -- it never
    # compounds across iterations.
    for _ in range(ALTERNATIONS):
        angle_total = best_rotation_deg(
            source_pts, t, target_tree, MAX_TWEAK_DEG, ROTATION_SEARCH_STEP_DEG, FINE_EVAL_THRESHOLD * 2
        )
        rotated = source_pts @ y_rotation_matrix(angle_total).T
        t = best_translation(rotated, target_tree, target_pts, TRANSLATION_SEARCH_THRESHOLD)

    R = y_rotation_matrix(angle_total)
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    return transform, angle_total, t


def evaluate_fitness(source, target, transform, threshold=FINE_EVAL_THRESHOLD):
    result = o3d.pipelines.registration.evaluate_registration(source, target, threshold, transform)
    return result.fitness


def align_chain(pcds, sign):
    """Rotate pcds[1:] by sign * nominal angle, then chain each onto a
    growing merged target starting from pcds[0], applying the bounded
    translation+rotation tweak at each step."""
    aligned = [copy.deepcopy(pcds[0])]
    target = copy.deepcopy(pcds[0])
    total_fitness = 0.0

    for pcd, angle in zip(pcds[1:], NOMINAL_ANGLES_DEG[1:]):
        source = rotate_around_axis(pcd, sign * angle)
        transform, angle_delta, t = bounded_tweak(source, target)
        print(f"  {angle} deg view: rotation tweak {angle_delta:+.2f} deg, "
              f"translation {np.linalg.norm(t) * 1000:.2f} mm")

        total_fitness += evaluate_fitness(source, target, transform)

        source_aligned = copy.deepcopy(source)
        source_aligned.transform(transform)
        aligned.append(source_aligned)

        target = target + source_aligned
        target = target.voxel_down_sample(FINAL_VOXEL_SIZE)

    return aligned, total_fitness


def save_overlay(pcds, filepath, title):
    colors = [[0.85, 0.2, 0.2], [0.2, 0.65, 0.2], [0.2, 0.4, 0.9], [0.9, 0.6, 0.0]]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    for pcd, color, angle in zip(pcds, colors, NOMINAL_ANGLES_DEG):
        pts = np.asarray(pcd.points)
        step = max(1, len(pts) // 3000)
        ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2],
                    c=[color], s=1.0, label=f"{angle} deg")
    ax.set_title(title)
    ax.legend(markerscale=8)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}")


def merge(pcds):
    merged = o3d.geometry.PointCloud()
    for pcd in pcds:
        merged += pcd
    return merged


def main():
    input_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(INPUT_FOLDER)
    output_folder = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(OUTPUT_FOLDER)
    pattern = sys.argv[3] if len(sys.argv) > 3 else PATTERN
    output_folder.mkdir(parents=True, exist_ok=True)

    files = sorted(input_folder.glob(pattern))
    if len(files) != 4:
        raise SystemExit(f"Expected 4 files matching {pattern!r} in {input_folder}, found {len(files)}")

    pcds = [o3d.io.read_point_cloud(str(f)) for f in files]
    for f, pcd in zip(files, pcds):
        print(f"{f.name}: {len(pcd.points)} points")

    pcds = [shift_toward_camera(pcd) for pcd in pcds]
    print(f"Shifted each view {PRE_ROTATION_CAMERA_SHIFT * 100:.1f} cm toward the camera "
          f"(direction {CAMERA_DIRECTION}) before rotation")

    best_sign, best_aligned, best_fitness = None, None, -1.0
    for sign in (1, -1):
        print(f"\n--- sign={sign:+d} ---")
        aligned, fitness = align_chain(pcds, sign)
        print(f"sign={sign:+d}: total fitness = {fitness:.4f}")
        if fitness > best_fitness:
            best_sign, best_aligned, best_fitness = sign, aligned, fitness

    print(f"\nUsing rotation sign {best_sign:+d} (total fitness {best_fitness:.4f})")

    nominal_pcds = [rotate_around_axis(p, best_sign * a) for p, a in zip(pcds, NOMINAL_ANGLES_DEG)]
    save_overlay(nominal_pcds, output_folder / "before_tweak.png", "Nominal rotation only (before tweak)")
    save_overlay(best_aligned, output_folder / "after_tweak.png", "After bounded tweak")

    before_tweak_merged = merge(nominal_pcds)
    before_path = output_folder / "tomato_merged_before_tweak.ply"
    o3d.io.write_point_cloud(str(before_path), before_tweak_merged)
    print(f"Merged (nominal rotation only): {len(before_tweak_merged.points)} points -> {before_path}")

    merged = merge(best_aligned)
    merged = merged.voxel_down_sample(FINAL_VOXEL_SIZE)
    merged, _ = merged.remove_statistical_outlier(
        nb_neighbors=OUTLIER_NB_NEIGHBORS, std_ratio=OUTLIER_STD_RATIO
    )
    merged, _ = merged.remove_radius_outlier(
        nb_points=RADIUS_OUTLIER_NB_POINTS, radius=RADIUS_OUTLIER_RADIUS
    )

    out_path = output_folder / "tomato_merged.ply"
    o3d.io.write_point_cloud(str(out_path), merged)
    print(f"Merged cloud: {len(merged.points)} points -> {out_path}")


if __name__ == "__main__":
    main()
