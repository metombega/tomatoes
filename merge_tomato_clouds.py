"""
Register the 5 segmented tomato/cone point clouds in
rgbd_photos/five_shots_segmented/ into one merged tomato point cloud.

Each *_tomato.ply / *_cone.ply pair (see segment_tomato_cone.py) comes from
its own independent scan session, in its own coordinate frame. The rig is
a turntable: the tomato sits close to the rotation axis (so it barely
moves between shots), while the small violet cone marker sits farther out
and sweeps around that same axis as the turntable is stepped between the
5 shots. That means the 5 cone peaks all lie on a single circle -- the
turntable's own circle of rotation -- which is exactly the "center of the
ball" analogue from the old pipeline, generalized to a whole circle since
we no longer get to assume the nominal step angles are already known:

  1. Fit a 3D circle through the 5 cone peaks: its plane's normal is the
     turntable's rotation axis, its center is a point on that axis, and
     each peak's angular position around the circle tells us exactly how
     far that shot's turntable step actually was.
  2. Pick one shot as the reference angle. For every other shot, rotate
     its whole tomato cloud around the fitted axis (pivoting on the
     circle's center) by the angle that brings its own cone peak to the
     reference peak's angular position.
  3. Fold the 5 rotated shots into one cloud in angular order around the
     circle (not file order), one at a time. Concatenating them
     unconditionally would double up near-coincident points wherever two
     shots see the same patch of surface, since registration is close but
     not pixel-perfect -- visible as a "double-layered" shell in cross
     section. A plain closest-point dedup isn't safe here though: at that
     distance scale it can't tell "this is the same surface point seen
     twice, at a slightly wrong depth" apart from "this is genuinely the
     next point over on a continuously-sampled surface", and being wrong
     in the second case eats real coverage.
     The distinguishing signal is direction from the tomato's own center:
     two points that are close together *and* sit along essentially the
     same ray from center are almost certainly the same physical spot
     seen twice (registration error mostly reads as a radial/depth
     offset, not a lateral one); two points that are close together but
     along noticeably different rays are neighboring surface samples and
     both should be kept. The merge order is circular (shot 5 is
     angularly adjacent to shot 1, same as any other consecutive pair),
     so the restricting also wraps around: 1 restricts 2, 2 restricts 3,
     ..., 4 restricts 5, and finally 5 restricts 1 -- the shot that went
     in first unfiltered gets pruned last, against everything else that
     survived. See dedup_across_shots.

Usage:
    python merge_tomato_clouds.py [input_folder] [output_folder] [--no-dedup | --dedup] [--no-refine]

Defaults convert rgbd_photos/five_shots_segmented/*_tomato.ply (+ matching
*_cone.ply) into rgbd_photos/five_shots_merged/tomato_merged.ply
(deduplicated, voxel downsampled and outlier-cleaned) and
tomato_merged_full.ply (every aligned point, before any filtering), plus a
diagnostic render of the fitted circle and the before/after alignment.

The double-layer dedup pass (step 3 above) is on by default
(DIRECTIONAL_DEDUP_DEFAULT) -- pass --no-dedup to fall back to plain
concatenation, or --dedup to force it on regardless of the default.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

INPUT_FOLDER = "rgbd_photos/video_on_ratation_table_output_20261013_105001/segmented"
OUTPUT_FOLDER = "rgbd_photos/video_on_ratation_table_output_20261013_105001/merged"

# Refinement after the cone-circle alignment (see refine_alignment).
# The cone peak is a single noisy point lever-armed out to the tomato, so
# the circle fit leaves ~0.5-1 mm of residual misalignment between shots,
# visible as a small gap/double shell at the seams. Each shot is therefore
# re-registered against the others, but only with the freedom the rig
# physically has: a rotation about the fitted turntable axis plus a small
# translation (the tomato can shift a little on the platform). Free 6-DOF
# ICP is deliberately avoided: the tomato is nearly spherical, so ICP can
# rotate a shot a few degrees about the tomato's own center for free, which
# slides surface texture/color tangentially for no geometric gain.
REFINE_DEFAULT = True
REFINE_ROUNDS = 4
REFINE_VOXEL = 0.001
REFINE_MAX_DIST = 0.003
REFINE_MAX_SHIFT = 0.004
REFINE_MAX_ANGLE_DEG = 6.0

# Whether the directional double-layer dedup (dedup_across_shots) runs at
# all. Overridable per-run with --dedup / --no-dedup.
DIRECTIONAL_DEDUP_DEFAULT = False

FINAL_VOXEL_SIZE = 0.0004
OUTLIER_NB_NEIGHBORS = 30
OUTLIER_STD_RATIO = 1.5
RADIUS_OUTLIER_NB_POINTS = 8
RADIUS_OUTLIER_RADIUS = 0.001

# Cross-shot dedup (see dedup_across_shots). The spatial radius is
# generous -- large enough to span the few mm of residual misalignment
# the circle-fit rotation can leave at the tomato's surface (the cone
# peak's own detection noise, lever-armed out by the ~35-40mm distance
# between the peak and the tomato) -- because it's the angle check, not
# this radius, that's doing the real discrimination.
DEDUP_RADIUS = 0.005

# How close two points' directions from the tomato center must be (in
# degrees) to count as "the same ray", and so as a depth-duplicate rather
# than a neighboring surface sample. This is a real tradeoff, not a
# solved value: looser catches more of the double-layering but starts
# thinning out genuine coverage right at shot-boundary seams (where both
# shots' points are naturally sparser, so removing either one leaves a
# visible gap); tighter protects those seams but lets more double-
# layering back in. 2 degrees was the best balance found by inspection --
# revisit if either symptom becomes the more visible problem.
DEDUP_ANGLE_DEG = 2.0

SHOT_COLORS = [[0.85, 0.2, 0.2], [0.2, 0.65, 0.2], [0.2, 0.4, 0.9], [0.9, 0.6, 0.0], [0.6, 0.2, 0.8]]


def cone_peak(cone_pcd):
    """The cone's apex: the narrow tapering end, found via PCA along the
    cone's long axis and picking whichever end has the tighter local
    spread (the tip, not the wide circular base)."""
    pts = np.asarray(cone_pcd.points)
    center = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - center)
    axis = vt[0]
    proj = (pts - center) @ axis

    lo_mask = proj < np.percentile(proj, 10)
    hi_mask = proj > np.percentile(proj, 90)

    def local_spread(mask):
        sel = pts[mask]
        return np.linalg.norm(sel - sel.mean(axis=0), axis=1).std()

    tip_idx = np.argmin(proj) if local_spread(lo_mask) < local_spread(hi_mask) else np.argmax(proj)
    return pts[tip_idx]


def fit_circle_3d(points):
    """Best-fit circle through 3D points: fit a plane (SVD), project onto
    it, fit a 2D circle (algebraic least squares) in that plane. Returns
    the circle's center (3D), radius, plane normal, and an in-plane
    (e1, e2) basis used to measure angles around the circle."""
    centroid = points.mean(axis=0)
    centered = points - centroid

    _, _, vt = np.linalg.svd(centered)
    normal = vt[2]

    seed = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(normal, seed)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)

    x, y = centered @ e1, centered @ e2
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    radius = float(np.sqrt(sol[2] + cx ** 2 + cy ** 2))
    center_3d = centroid + cx * e1 + cy * e2

    return center_3d, radius, normal, e1, e2


def angle_around_circle(point, center, e1, e2):
    local = point - center
    return np.arctan2(local @ e2, local @ e1)


def rotate_about_axis(pcd, axis, center, angle_rad):
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle_rad)
    return pcd.rotate(R, center=center)


def _prep_for_refine(pcd):
    down = pcd.voxel_down_sample(REFINE_VOXEL)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=REFINE_VOXEL * 5, max_nn=30))
    return down


def overlap_residuals(clouds, max_dist=0.004, min_pts=200):
    """Median nearest-neighbor distance (mm) over each pair's overlapping
    points, keyed by (i, j). Used to report how well shots line up."""
    trees = [cKDTree(np.asarray(c.points)) for c in clouds]
    out = {}
    for i, a in enumerate(clouds):
        pts = np.asarray(a.points)
        for j in range(i + 1, len(clouds)):
            d, _ = trees[j].query(pts)
            near = d < max_dist
            if near.sum() >= min_pts:
                out[(i, j)] = float(np.median(d[near]) * 1000)
    return out


def refine_alignment(clouds, stems, axis, center, fixed_idx=0):
    """Refine each shot against the union of all the others (several
    Gauss-Seidel rounds: later shots see earlier shots' corrections) by
    least-squares point-to-plane distance over 4 parameters: a rotation
    about the turntable axis and a translation, both bounded. `fixed_idx`
    (the reference shot) never moves, otherwise the whole set could drift
    together. Modifies `clouds` in place."""
    totals = [np.zeros(4) for _ in clouds]
    lo = [-np.radians(REFINE_MAX_ANGLE_DEG)] + [-REFINE_MAX_SHIFT] * 3
    hi = [np.radians(REFINE_MAX_ANGLE_DEG)] + [REFINE_MAX_SHIFT] * 3

    for _ in range(REFINE_ROUNDS):
        for i, cloud in enumerate(clouds):
            if i == fixed_idx:
                continue
            others = o3d.geometry.PointCloud()
            for j, c in enumerate(clouds):
                if j != i:
                    others += c
            target = _prep_for_refine(others)
            tgt_pts, tgt_normals = np.asarray(target.points), np.asarray(target.normals)
            tree = cKDTree(tgt_pts)
            src = np.asarray(_prep_for_refine(cloud).points)

            def residuals(x):
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * x[0])
                moved = (src - center) @ R.T + center + x[1:]
                d, k = tree.query(moved)
                r = np.einsum("ij,ij->i", moved - tgt_pts[k], tgt_normals[k])
                return np.where(d < REFINE_MAX_DIST, r, 0.0)

            sol = least_squares(residuals, np.zeros(4), x_scale=[0.01, 0.001, 0.001, 0.001],
                                loss="soft_l1", f_scale=0.0007, bounds=(lo, hi))
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * sol.x[0])
            moved = (np.asarray(cloud.points) - center) @ R.T + center + sol.x[1:]
            cloud.points = o3d.utility.Vector3dVector(moved)
            totals[i] += sol.x

    for stem, x in zip(stems, totals):
        print(f"  {stem}: rotated {np.degrees(x[0]):+.2f} deg more, shifted {(x[1:] * 1000).round(2)} mm")


def _prune_against(candidate_pts, reference_pts, tomato_center, radius, cos_threshold):
    """Indices of `candidate_pts` that do NOT have a depth-duplicate in
    `reference_pts` -- close in distance *and* along essentially the same
    ray from `tomato_center` (see dedup_across_shots)."""
    if len(reference_pts) == 0:
        return np.arange(len(candidate_pts))

    tree = cKDTree(reference_pts)
    dists, idx = tree.query(candidate_pts, k=1)
    close = dists < radius

    new_dir = candidate_pts - tomato_center
    new_dir /= np.linalg.norm(new_dir, axis=1, keepdims=True)
    old_dir = reference_pts[idx] - tomato_center
    old_dir /= np.linalg.norm(old_dir, axis=1, keepdims=True)
    same_ray = np.sum(new_dir * old_dir, axis=1) > cos_threshold

    return np.where(~(close & same_ray))[0]


def dedup_across_shots(shot_clouds, tomato_center, radius=DEDUP_RADIUS, angle_deg=DEDUP_ANGLE_DEG):
    """Fold clouds (already in the desired circular merge order) into
    one. Each shot is pruned against exactly one neighbor -- the
    previous shot in the circle, already-pruned -- not against the whole
    accumulated history: 1 restricts 2, 2 restricts 3, ..., 4 restricts
    5, and finally 5 restricts 1 (closing the loop, using 5's own
    already-pruned points, not the other three shots as well).

    This was originally cascading (each shot checked against everything
    merged so far), which compounds: whichever shot ends up pruned last
    gets checked against every other shot combined, and lost far more
    real coverage than intended -- a large, genuine hole, not just the
    intended double-layer cleanup. Strictly pairwise avoids that.

    A point is dropped as a depth-duplicate if it's close in distance
    *and* sits along essentially the same ray from `tomato_center` as its
    match (see the module docstring for why direction, not just
    distance, is what separates that from a neighboring surface
    sample)."""
    cos_threshold = np.cos(np.radians(angle_deg))
    n = len(shot_clouds)

    kept_clouds = [shot_clouds[0]]
    for i in range(1, n):
        prev_pts = np.asarray(kept_clouds[i - 1].points)
        cur_pts = np.asarray(shot_clouds[i].points)
        keep = _prune_against(cur_pts, prev_pts, tomato_center, radius, cos_threshold)
        kept_clouds.append(shot_clouds[i].select_by_index(keep.tolist()))

    first_pts = np.asarray(kept_clouds[0].points)
    last_pts = np.asarray(kept_clouds[-1].points)
    keep_first = _prune_against(first_pts, last_pts, tomato_center, radius, cos_threshold)
    print(f"  closing the loop: shot 1 pruned from {len(first_pts)} to {len(keep_first)} points "
          f"against shot 5 alone")
    kept_clouds[0] = kept_clouds[0].select_by_index(keep_first.tolist())

    merged = o3d.geometry.PointCloud()
    for c in kept_clouds:
        merged += c
    return merged


def save_overlay(clouds_points, filepath, title, extra=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    for i, pts in enumerate(clouds_points):
        step = max(1, len(pts) // 3000)
        ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2],
                    c=[SHOT_COLORS[i % len(SHOT_COLORS)]], s=1.0, label=f"shot {i + 1}")
    if extra is not None:
        extra(ax)
    ax.set_title(title)
    ax.legend(markerscale=8)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}")


def save_circle_diagnostic(peaks, center, radius, normal, e1, e2, filepath):
    theta = np.linspace(0, 2 * np.pi, 200)
    circle_pts = center + radius * (np.outer(np.cos(theta), e1) + np.outer(np.sin(theta), e2))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(circle_pts[:, 0], circle_pts[:, 1], circle_pts[:, 2], c="black", lw=1, label="fitted circle")
    for i, p in enumerate(peaks):
        ax.scatter(*p, c=[SHOT_COLORS[i % len(SHOT_COLORS)]], s=80, marker="*", label=f"cone peak {i + 1}")
    ax.scatter(*center, c="black", s=40, marker="x", label="circle center")
    ax.set_title(f"Cone peaks and fitted turntable circle (radius {radius * 1000:.1f} mm)")
    ax.legend(markerscale=1, fontsize=7)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}")


def main():
    args = sys.argv[1:]
    directional_dedup = DIRECTIONAL_DEDUP_DEFAULT
    if "--no-dedup" in args:
        args.remove("--no-dedup")
        directional_dedup = False
    use_refine = REFINE_DEFAULT
    if "--no-refine" in args:
        args.remove("--no-refine")
        use_refine = False
    if "--dedup" in args:
        args.remove("--dedup")
        directional_dedup = True

    input_folder = Path(args[0]) if len(args) > 0 else Path(INPUT_FOLDER)
    output_folder = Path(args[1]) if len(args) > 1 else Path(OUTPUT_FOLDER)
    output_folder.mkdir(parents=True, exist_ok=True)

    stems = sorted({p.name[: -len("_tomato.ply")] for p in input_folder.glob("*_tomato.ply")})
    if len(stems) < 3:
        raise SystemExit(f"Expected at least 3 *_tomato.ply/_cone.ply pairs in {input_folder}, found {len(stems)}")

    shots = []
    for stem in stems:
        tomato = o3d.io.read_point_cloud(str(input_folder / f"{stem}_tomato.ply"))
        cone = o3d.io.read_point_cloud(str(input_folder / f"{stem}_cone.ply"))
        peak = cone_peak(cone)
        shots.append({"stem": stem, "tomato": tomato, "peak": peak})
        print(f"{stem}: {len(tomato.points)} tomato points, {len(cone.points)} cone points, peak {peak.round(4)}")

    peaks = np.array([s["peak"] for s in shots])
    center, radius, normal, e1, e2 = fit_circle_3d(peaks)

    plane_residual = (peaks - center) @ normal
    radial_residual = np.linalg.norm(
        np.stack([(peaks - center) @ e1, (peaks - center) @ e2], axis=1), axis=1
    ) - radius
    print(f"\nFitted circle: center {center.round(4)}, radius {radius * 1000:.2f} mm, axis {normal.round(4)}")
    print(f"  out-of-plane residuals (mm): {(plane_residual * 1000).round(3)}")
    print(f"  radial residuals (mm): {(radial_residual * 1000).round(3)}")

    save_circle_diagnostic(peaks, center, radius, normal, e1, e2, output_folder / "cone_peak_circle.png")

    raw_points = [np.asarray(s["tomato"].points) for s in shots]
    save_overlay(raw_points, output_folder / "before_registration.png", "Raw (no alignment)")

    angles = np.array([angle_around_circle(p, center, e1, e2) for p in peaks])
    ref_idx = 0
    ref_angle = angles[ref_idx]
    print(f"\nReference shot: {shots[ref_idx]['stem']}")

    aligned_tomato_points = []
    for s, angle in zip(shots, angles):
        delta = ref_angle - angle
        rotate_about_axis(s["tomato"], normal, center, delta)
        aligned_tomato_points.append(np.asarray(s["tomato"].points))
        print(f"{s['stem']}: rotated {np.degrees(delta):+.2f} deg about the turntable axis")

    save_overlay(aligned_tomato_points, output_folder / "after_registration.png",
                 "After circle-based turntable-angle alignment")

    def report(label):
        res = overlap_residuals([s["tomato"] for s in shots])
        print(f"Overlap gap {label} (median mm per overlapping pair): "
              + ", ".join(f"{i + 1}-{j + 1}: {v:.2f}" for (i, j), v in res.items()))

    report("after circle alignment")
    if use_refine:
        print("\nRefining shot poses against each other:")
        refine_alignment([s["tomato"] for s in shots], [s["stem"] for s in shots], normal, center, fixed_idx=ref_idx)
        report("after refinement")
        aligned_tomato_points = [np.asarray(s["tomato"].points) for s in shots]
        save_overlay(aligned_tomato_points, output_folder / "after_refinement.png", "After pose refinement")

    merge_order = np.argsort(angles)
    ordered_stems = [shots[i]["stem"] for i in merge_order]
    print(f"\nMerge order around the circle: {ordered_stems}")

    full = o3d.geometry.PointCloud()
    for i in merge_order:
        full += shots[i]["tomato"]

    full_path = output_folder / "tomato_merged_full.ply"
    o3d.io.write_point_cloud(str(full_path), full)
    print(f"Merged cloud (every aligned point, before any filtering): {len(full.points)} points -> {full_path}")

    if directional_dedup:
        tomato_center = np.vstack(aligned_tomato_points).mean(axis=0)
        merged = dedup_across_shots([shots[i]["tomato"] for i in merge_order], tomato_center)
        print(f"After cross-shot dedup: {len(merged.points)} points (from {len(full.points)})")
    else:
        merged = full

    merged = merged.voxel_down_sample(FINAL_VOXEL_SIZE)
    merged, _ = merged.remove_statistical_outlier(nb_neighbors=OUTLIER_NB_NEIGHBORS, std_ratio=OUTLIER_STD_RATIO)
    merged, _ = merged.remove_radius_outlier(nb_points=RADIUS_OUTLIER_NB_POINTS, radius=RADIUS_OUTLIER_RADIUS)

    out_path = output_folder / "tomato_merged.ply"
    o3d.io.write_point_cloud(str(out_path), merged)
    print(f"Merged cloud (downsampled + cleaned): {len(merged.points)} points -> {out_path}")


if __name__ == "__main__":
    main()
