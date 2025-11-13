"""
End-to-end RGB-D reconstruction pipeline that converts per-frame images into a
single registered point cloud, optionally aligns the estimated camera poses to
ground truth, and visualizes both trajectories. The script glues together depth
unprojection, feature-based global alignment, a custom ICP refinement loop, and
simple visualization utilities.
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import open3d as o3d

PRED_START_COLOR = (1.0, 0.85, 0.0)
PRED_END_COLOR = (1.0, 0.2, 0.6)
GT_START_COLOR = (0.2, 1.0, 0.8)
GT_END_COLOR = (0.2, 0.4, 1.0)
FOV_DEG = 90.0

def compute_alignment_transform(source, target):
    """
    Compute a similarity transform that best aligns `source` points to `target`
    points in the least-squares sense. This is used to roughly align predicted
    and ground-truth camera centers before reporting errors.
    """
    if len(source) == 0 or len(target) == 0:
        return np.eye(4), 1.0

    n = min(len(source), len(target))
    src = source[:n]
    tgt = target[:n]

    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)

    src_centered = src - src_mean
    tgt_centered = tgt - tgt_mean

    cov = (tgt_centered.T @ src_centered) / n
    U, _, Vt = np.linalg.svd(cov)

    reflection = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        reflection[-1, -1] = -1

    rotation = U @ reflection @ Vt

    scale = 1.0
    translation = tgt_mean - scale * rotation @ src_mean

    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform, scale


def apply_transform(points, transform):
    """Apply a homogeneous transformation matrix to a set of 3-D points."""
    ones = np.ones((points.shape[0], 1))
    hom = np.hstack((points, ones))
    transformed = (transform @ hom.T).T
    return transformed[:, :3]

def remove_ceiling_from_pcd(point_cloud, quantile=0.97):
    """
    Trim the tallest points from a cloud by keeping only the bottom `quantile`
    of the y-coordinate. Helps reduce ceiling clutter in the final visualization.
    """
    points = np.asarray(point_cloud.points)
    y_coords = points[:, 1]
    threshold = np.quantile(y_coords, quantile)
    keep_idx = np.where(y_coords <= threshold)[0]
    return point_cloud.select_by_index(keep_idx)


def create_marker(position, radius=0.05, color=(1.0, 0.0, 0.0)):
    """Build a colored sphere mesh to highlight interesting trajectory points."""
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    marker.compute_vertex_normals()
    marker.paint_uniform_color(np.asarray(color))
    marker.translate(position)
    return marker


def skew(vec):
    """Return the 3x3 skew-symmetric matrix associated with vector `vec`."""
    return np.array(
        [
            [0.0, -vec[2], vec[1]],
            [vec[2], 0.0, -vec[0]],
            [-vec[1], vec[0], 0.0],
        ],
        dtype=np.float64,
    )

def so3_exp(omega):
    """
    Convert an axis-angle rotation vector into a rotation matrix using the
    exponential map. Used to update ICP pose estimates with small increments.
    """
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        return np.eye(3) + skew(omega)

    K = skew(omega)
    theta_sq = theta * theta
    A = np.sin(theta) / theta
    B = (1.0 - np.cos(theta)) / theta_sq
    return np.eye(3) + A * K + B * (K @ K)


def depth_image_to_point_cloud(
    rgb,
    depth,
    storage_depth_range=10.0,
):
    """
    Convert an RGB-D pair into an Open3D point cloud. The pinhole intrinsics are
    derived from the image resolution and a fixed FOV, and depth values are
    scaled back into meters so every valid pixel turns into a 3-D colored point.
    """

    height, width = depth.shape

    fov_deg = FOV_DEG
    focal = 0.5 * width / np.tan(np.deg2rad(fov_deg / 2.0))
    fx = fy = focal
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    depth = depth.astype(np.float32)
    depth_m = depth / 255.0 * storage_depth_range

    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.stack((u[valid], v[valid], np.ones_like(u[valid])), axis=0) # homogenous coordinates
    z = depth_m[valid]

    K_inv = np.linalg.inv(K)
    cam_points = K_inv @ (pixels * z)
    points = cam_points.T

    colors = rgb[valid].astype(np.float32) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def preprocess_point_cloud(pcd, voxel_size):
    """
    Denoise, downsample, and enrich a point cloud with normals + FPFH features.
    The reduced representation makes RANSAC and ICP faster and more stable.
    """
    pcd_clean, ind = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0,
    )
    if len(ind) > 0:
        pcd = pcd_clean

    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    """
    Use feature matching + RANSAC to find a coarse alignment between successive
    frames. This protects the local ICP stage from large pose jumps.
    """
    # We allow a relatively loose correspondence distance because the point
    # clouds are still at voxel-level resolution. A larger threshold increases
    # robustness, but too large will admit many wrong matches.
    distance_threshold = voxel_size * 1.5
    logging.info(
        "Global registration: voxel_size=%.3f, distance_threshold=%.3f, source_pts=%d, target_pts=%d",
        voxel_size,
        distance_threshold,
        len(source_down.points),
        len(target_down.points),
    )
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    logging.info(
        "Global registration result: fitness=%.4f, inlier_rmse=%.4f",
        result.fitness,
        result.inlier_rmse,
    )
    return result  # standardized ICP result (pose, stats, correspondences)


def local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    """
    Standard Open3D multi-scale point-to-plane ICP refinement. We shrink the
    correspondence distance and iteration budget at each level to gradually
    tighten the alignment.
    """
    # Distance thresholds per pyramid level; coarse → fine allows large motions
    # initially and then locks the solution as the estimate improves.
    max_correspondence_distances = [voxel_size * 4.0, voxel_size * 2.0, voxel_size * 1.0]  # meters per level
    # Iteration budget per level; fewer passes are needed once we are close.
    max_iterations = [40, 30, 20]  # iterations per level (coarse → fine)

    # Copy the input transform so we can update it in place without surprising callers.
    current_transform = np.array(trans_init, dtype=np.float64)
    result = None

    # Run ICP for each pyramid level, feeding the refined transform into the next level.
    for level, (dist, iters) in enumerate(zip(max_correspondence_distances, max_iterations), start=1):
        logging.info(
            "Local ICP (Open3D) level %d: max_corr=%.3f, max_iter=%d",
            level,
            dist,
            iters,
        )  # track which schedule parameters are currently active
        # Core Open3D ICP call: uses point-to-plane residuals, the specified
        # correspondence gate, and convergence criteria limited by `iters`.
        result = o3d.pipelines.registration.registration_icp(
            source_down,
            target_down,
            dist,  # max correspondence distance for this stage
            current_transform,  # starting pose for refinement
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),  # residual model
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters),  # stop rule
        )
        current_transform = result.transformation  # feed the refined pose into the next level
        logging.info(
            "Local ICP (Open3D) level %d result: fitness=%.4f, inlier_rmse=%.4f",
            level,
            result.fitness,
            result.inlier_rmse,
        )  # record quality metrics for later debugging

    return result


def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    """
    Custom ICP loop that explicitly builds linearized point-to-plane constraints
    and solves for twist updates via least squares. The staged thresholds mimic
    a coarse-to-fine schedule similar to the Open3D helper above.
    """
    # Cache numpy arrays for faster math; Open3D containers are convenient but
    # slower to iterate over in tight loops.
    source_points = np.asarray(source_down.points)  # Nx3 source coordinates
    target_points = np.asarray(target_down.points)  # Mx3 target coordinates

    target_normals = np.asarray(target_down.normals)  # normals for point-to-plane constraints
    assert target_normals is not None  # downstream math assumes valid normals

    tree = o3d.geometry.KDTreeFlann(target_down)  # for fast NN queries
    T = np.array(trans_init, dtype=np.float64)  # running SE(3) estimate

    # Each stage uses progressively smaller correspondence thresholds and fewer
    # iterations. This mimics a pyramid without explicitly rebuilding the cloud.
    iteration_schedule = [30, 15, 10]  # iterations per level (coarse → fine)
    threshold_schedule = [4.0, 2.5, 1]  # scale factors for correspondence radius
    prev_error = np.inf  # track convergence between iterations
    correspondence_indices = []  # store successful matches for reporting

    homogeneous = np.hstack((source_points, np.ones((source_points.shape[0], 1))))  # allow single matmul for transforms

    converged = False  # once set, we drop out of the outer schedule loop
    for level, (stage_iters, threshold_factor) in enumerate(zip(iteration_schedule, threshold_schedule), start=1):  # outer coarse-to-fine loop
        base_threshold = voxel_size * threshold_factor  # base correspondence gate (squared later)
        logging.info(
            "Local ICP (custom) level %d: base_threshold=%.3f, iterations=%d",
            level,
            base_threshold,
            stage_iters,
        )  # emit schedule information so we can diagnose convergence

        for itr in range(stage_iters):  # Gauss-Newton iterations within current schedule level
            # Apply current SE(3) estimate to all source points.
            transformed = (T @ homogeneous.T).T[:, :3]  # apply current pose to all points

            # Shrink the acceptance radius over iterations so early steps
            # explore a broader basin and later steps focus on fine detail.
            adaptive_factor = max(0.3, 1.0 - itr / stage_iters)  # smoothly reduce acceptance radius
            current_threshold = base_threshold * adaptive_factor  # current linear threshold
            threshold_sq = current_threshold ** 2  # squared once so we compare directly to nn_dist

            # Accumulate linear system rows for A * xi = b where xi is the twist.
            A_rows = []  # rows of the Jacobian matrix
            b_rows = []  # RHS residuals
            correspondence_indices = []  # reset per iteration so fitness is accurate

            for idx, point in enumerate(transformed):  # visit each transformed source point
                # Find the closest target point; reject if beyond the gate.
                k, nn_idx, nn_dist = tree.search_knn_vector_3d(point, 1)  # 1-NN query in target cloud
                if k == 0 or nn_dist[0] > threshold_sq:
                    continue

                tgt_idx = nn_idx[0]
                correspondence_indices.append([idx, tgt_idx])  # record the index pairing for metrics

                normal = target_normals[tgt_idx]  # plane orientation at the match
                target_point = target_points[tgt_idx]  # matched target coordinate

                # The derivative of the point-to-plane error w.r.t. small pose
                # perturbations produces the cross product term (rotation) and
                # the normal itself (translation).
                cross = np.cross(point, normal)  # rotation Jacobian part
                A_rows.append(np.hstack((cross, normal)))  # full 1x6 Jacobian row
                b_rows.append(np.dot(normal, target_point - point))  # signed distance along normal

            # Solve the normal equations to obtain the incremental pose update.
            A = np.asarray(A_rows)  # Jacobian matrix
            b = np.asarray(b_rows)  # residual vector
            try:
                xi, *_ = np.linalg.lstsq(A, b, rcond=None)  # solve normal equations for twist
            except np.linalg.LinAlgError:
                converged = True  # ill-conditioned system; treat as convergence to avoid blowing up
                break

            rot_vec = xi[:3]  # incremental rotation (axis-angle)
            trans_vec = xi[3:]  # incremental translation

            # Convert the twist into an SE(3) matrix and left-multiply it so
            # updates compound correctly.
            delta_R = so3_exp(rot_vec)
            delta_T = np.eye(4)
            delta_T[:3, :3] = delta_R
            delta_T[:3, 3] = trans_vec

            T = delta_T @ T

            residual = b - A @ xi  # compute point-to-plane residuals after update
            mean_error = float(np.sqrt(np.mean(residual ** 2))) if residual.size else 0.0  # RMS error

            # Stop the loop if the update is tiny or the error stabilizes.
            if np.linalg.norm(xi) < 1e-5 or abs(prev_error - mean_error) < 1e-6:  # small pose delta OR error plateau
                prev_error = mean_error  # remember final RMS for reporting
                converged = True  # flag success so we can bail early
                logging.info(
                    "Local ICP (custom) level %d converged at iter %d with mean_error=%.6f",
                    level,
                    itr + 1,
                    mean_error,
                )
                break
            prev_error = mean_error
            # Loop continues with the updated pose and tightened threshold.

        if converged:
            logging.info("Local ICP (custom) exiting after level %d", level)  # no need to run remaining stages
            break
        else:
            logging.info("Local ICP (custom) level %d completed without convergence", level)  # move on to finer stage

    result = o3d.pipelines.registration.RegistrationResult()  # populate Open3D-style return value
    result.transformation = T  # final SE(3) pose
    result.fitness = len(correspondence_indices) / max(len(source_points), 1)  # match ratio
    result.inlier_rmse = prev_error if np.isfinite(prev_error) else 0.0  # last RMS error
    result.correspondence_set = o3d.utility.Vector2iVector(correspondence_indices)  # for debugging / visualization
    return result  # mimic Open3D API so caller can switch implementations easily



def reconstruct(args):
    """
    Main driver: load each RGB-D frame, register it to the running model, and
    accumulate all transformed points + camera poses. Returns the fused cloud,
    predicted camera centers, and per-frame transforms.
    """
    data_root = Path(args.data_root)
    rgb_dir = data_root / "rgb"
    depth_dir = data_root / "depth"

    frame_ids = sorted(int(p.stem) for p in rgb_dir.glob("*.png") if (depth_dir / p.name).exists())

    voxel_size = 0.125 #0.125

    def load_frame(frame_id):
        """Helper to load images and turn them into a point cloud for frame_id."""
        rgb_img = np.asarray(o3d.io.read_image(str(rgb_dir / f"{frame_id}.png")))
        depth_img = np.asarray(o3d.io.read_image(str(depth_dir / f"{frame_id}.png")))
        return depth_image_to_point_cloud(rgb_img, depth_img)

    result_points = []
    result_colors = []

    transforms = [np.eye(4)]
    pred_cam_pos = [np.zeros(3)]

    prev_pcd = load_frame(frame_ids[0])
    result_points.append(np.asarray(prev_pcd.points))
    result_colors.append(np.asarray(prev_pcd.colors))

    prev_down, prev_fpfh = preprocess_point_cloud(prev_pcd, voxel_size)

    for prev_idx, curr_idx in zip(frame_ids[:-1], frame_ids[1:]):
        # --- Frame loading and preprocessing ---------------------------------
        logging.info("Processing frame %d (previous frame %d).", curr_idx, prev_idx)
        source_pcd = load_frame(curr_idx)
        source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)

        trans_init = np.eye(4)
        if not args.disable_global_registration:
            # Global pass provides a decent seed; if it fails we fall back to
            # identity so the ICP stage can still attempt to register.
            logging.info("Running global registration between frames %d -> %d", curr_idx, prev_idx)
            global_result = execute_global_registration(
                source_down, prev_down, source_fpfh, prev_fpfh, voxel_size
            )
            trans_init = global_result.transformation
            if not np.all(np.isfinite(trans_init)) or global_result.fitness < 0.05:
                logging.warning(
                    "Global registration unstable (fitness=%.4f). Falling back to identity.",
                    global_result.fitness,
                )
                trans_init = np.eye(4)
        else:
            logging.info("Global registration disabled; using identity initialization.")

        if args.version == 'open3d':
            local_result = local_icp_algorithm(
                source_down,
                prev_down,
                trans_init,
                voxel_size=voxel_size,
            )
        else:
            local_result = my_local_icp_algorithm(
                source_down,
                prev_down,
                trans_init,
                voxel_size=voxel_size,
            )

        # Transform is expressed in the previous frame; accumulate to global
        # coordinates by chaining with the last world transform.
        transform = local_result.transformation
        world_transform = transforms[-1] @ transform
        transforms.append(world_transform)
        pred_cam_pos.append(world_transform[:3, 3].copy())

        homogeneous = np.concatenate(
            [np.asarray(source_pcd.points), np.ones((len(source_pcd.points), 1))],
            axis=1,
        )
        transformed_points = (world_transform @ homogeneous.T).T[:, :3]
        result_points.append(transformed_points)
        result_colors.append(np.asarray(source_pcd.colors))

        prev_pcd = source_pcd
        prev_down, prev_fpfh = source_down, source_fpfh

    result_pcd = o3d.geometry.PointCloud()
    if result_points:
        result_pcd.points = o3d.utility.Vector3dVector(np.vstack(result_points))
        result_pcd.colors = o3d.utility.Vector3dVector(np.vstack(result_colors))

    return result_pcd, np.asarray(pred_cam_pos), np.stack(transforms), frame_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='my_icp', help='open3d or my_icp')
    parser.add_argument('--data_root', type=str, default='data_collection/first_floor/')
    parser.add_argument('--output_pcd', type=str, default=None)
    parser.add_argument('--ceiling_quantile', type=float, default=0.6)
    parser.add_argument('--disable_global_registration', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.floor == 1:
        args.data_root = "data_collection/first_floor/"
    elif args.floor == 2:
        args.data_root = "data_collection/second_floor/"

    ceiling_quantile = float(np.clip(args.ceiling_quantile, 0.0, 1.0))

    result_pcd, pred_cam_pos, _, _ = reconstruct(args)

    gt_pose_path = Path(args.data_root) / 'GT_pose.npy'
    gt_cam_pos = np.zeros((0, 3))
    if gt_pose_path.exists():
        gt_raw = np.load(gt_pose_path)
        if gt_raw.ndim == 2 and gt_raw.shape[1] >= 3:
            gt_cam_pos = gt_raw[:, :3]

    pred_cam_pos_aligned = pred_cam_pos.copy()
    alignment_transform = np.eye(4)
    alignment_scale = 1.0

    if len(pred_cam_pos) and len(gt_cam_pos):
        limit = min(len(pred_cam_pos), len(gt_cam_pos))
        alignment_transform, alignment_scale = compute_alignment_transform(pred_cam_pos[:limit], gt_cam_pos[:limit])
        pred_cam_pos_aligned = apply_transform(pred_cam_pos, alignment_transform)
        result_pcd.transform(alignment_transform)
        l2_errors = np.linalg.norm(gt_cam_pos[:limit] - pred_cam_pos_aligned[:limit], axis=1)
        mean_l2 = float(np.mean(l2_errors))
    else:
        mean_l2 = float('nan')

    if 0.0 < ceiling_quantile < 1.0:
        result_pcd = remove_ceiling_from_pcd(result_pcd, ceiling_quantile)

    logging.info("Mean L2 distance: %.4f", mean_l2)

    if args.output_pcd:
        o3d.io.write_point_cloud(args.output_pcd, result_pcd)

    geometries = [result_pcd]

    if len(pred_cam_pos_aligned) > 1:
        pred_lines = [[i, i + 1] for i in range(len(pred_cam_pos_aligned) - 1)]
        pred_line_set = o3d.geometry.LineSet()
        pred_line_set.points = o3d.utility.Vector3dVector(pred_cam_pos_aligned)
        pred_line_set.lines = o3d.utility.Vector2iVector(pred_lines)
        pred_line_set.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[1.0, 0.0, 0.0]]), (len(pred_lines), 1))
        )
        geometries.append(pred_line_set)

    if len(gt_cam_pos) > 1:
        gt_lines = [[i, i + 1] for i in range(len(gt_cam_pos) - 1)]
        gt_line_set = o3d.geometry.LineSet()
        gt_line_set.points = o3d.utility.Vector3dVector(gt_cam_pos)
        gt_line_set.lines = o3d.utility.Vector2iVector(gt_lines)
        gt_line_set.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.0, 0.0, 0.0]]), (len(gt_lines), 1))
        )
        geometries.append(gt_line_set)

    if len(pred_cam_pos_aligned):
        geometries.append(
            create_marker(
                pred_cam_pos_aligned[0],
                radius=0.05,
                color=PRED_START_COLOR,
            )
        )
        geometries.append(
            create_marker(
                pred_cam_pos_aligned[-1],
                radius=0.05,
                color=PRED_END_COLOR,
            )
        )

    if len(gt_cam_pos):
        geometries.append(
            create_marker(
                gt_cam_pos[0],
                radius=0.05,
                color=GT_START_COLOR,
            )
        )
        geometries.append(
            create_marker(
                gt_cam_pos[-1],
                radius=0.05,
                color=GT_END_COLOR,
            )
        )

    o3d.visualization.draw_geometries(geometries)
