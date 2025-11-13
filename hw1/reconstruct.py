import argparse
import logging
from pathlib import Path
import numpy as np
import open3d as o3d


def _flatten_points_to_bev(points):
    flat = np.zeros_like(points)
    flat[:, 0] = points[:, 0]
    flat[:, 2] = points[:, 2]
    return flat


def _make_bev_point_cloud(pcd):
    bev_pcd = o3d.geometry.PointCloud()
    if not len(pcd.points):
        return bev_pcd
    points = np.asarray(pcd.points)
    bev_points = _flatten_points_to_bev(points)
    bev_pcd.points = o3d.utility.Vector3dVector(bev_points)
    if pcd.has_colors():
        bev_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
    return bev_pcd


def _flatten_trajectory(traj):
    if traj.size == 0:
        return traj
    flat = np.zeros_like(traj)
    flat[:, 0] = traj[:, 0]
    flat[:, 2] = traj[:, 2]
    return flat


def _create_line_set(points, color):
    if len(points) < 2:
        return None
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color, dtype=float), (len(lines), 1))
    )
    return line_set


def visualize_bev_projection(pcd, pred_traj, gt_traj):
    bev_geometries = []
    bev_pcd = _make_bev_point_cloud(pcd)
    if len(bev_pcd.points):
        bev_geometries.append(bev_pcd)

    pred_flat = _flatten_trajectory(pred_traj)
    gt_flat = _flatten_trajectory(gt_traj)

    pred_lines = _create_line_set(pred_flat, (1.0, 0.0, 0.0))
    if pred_lines is not None:
        bev_geometries.append(pred_lines)

    gt_lines = _create_line_set(gt_flat, (0.0, 0.0, 0.0))
    if gt_lines is not None:
        bev_geometries.append(gt_lines)

    if not bev_geometries:
        logging.warning("No geometries available for BEV visualization.")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="BEV Projection", width=960, height=720)
    render_opts = vis.get_render_option()
    if render_opts is not None:
        render_opts.point_size = 2.5
        render_opts.background_color = np.array([1.0, 1.0, 1.0])

    for geom in bev_geometries:
        vis.add_geometry(geom)

    all_points = []
    if len(bev_pcd.points):
        all_points.append(np.asarray(bev_pcd.points))
    if len(pred_flat):
        all_points.append(pred_flat)
    if len(gt_flat):
        all_points.append(gt_flat)

    if all_points:
        stacked = np.vstack(all_points)
        center = np.mean(stacked, axis=0)
    else:
        center = np.zeros(3)

    ctrl = vis.get_view_control()
    try:
        ctrl.set_lookat(center.tolist())
        ctrl.set_front([0.0, -1.0, 0.0])
        ctrl.set_up([0.0, 0.0, -1.0])
    except RuntimeError:
        pass

    vis.run()
    vis.destroy_window()


def compute_alignment_transform(source, target):
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
    ones = np.ones((points.shape[0], 1))
    hom = np.hstack((points, ones))
    transformed = (transform @ hom.T).T
    return transformed[:, :3]

def remove_ceiling_from_pcd(point_cloud, quantile=0.97):
    points = np.asarray(point_cloud.points)
    y_coords = points[:, 1]
    threshold = np.quantile(y_coords, quantile)
    keep_idx = np.where(y_coords <= threshold)[0]
    return point_cloud.select_by_index(keep_idx)

def create_marker(position, radius=0.05, color=(1.0, 0.0, 0.0)):
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    marker.compute_vertex_normals()
    marker.paint_uniform_color(np.asarray(color))
    marker.translate(position)
    return marker


def skew(vec):
    return np.array(
        [
            [0.0, -vec[2], vec[1]],
            [vec[2], 0.0, -vec[0]],
            [-vec[1], vec[0], 0.0],
        ],
        dtype=np.float64,
    )

def so3_exp(omega):
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
    Convert an RGB-D pair into an Open3D point cloud.
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
    colors = colors[:, [2, 1, 0]]  # BGR -> RGB for visualization

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def preprocess_point_cloud(pcd, voxel_size):
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
    return result


def local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    max_correspondence_distances = [voxel_size * 4.0, voxel_size * 2.0, voxel_size * 1.0]
    max_iterations = [40, 30, 20]

    current_transform = np.array(trans_init, dtype=np.float64)
    result = None

    for level, (dist, iters) in enumerate(zip(max_correspondence_distances, max_iterations), start=1):
        logging.info(
            "Local ICP (Open3D) level %d: max_corr=%.3f, max_iter=%d",
            level,
            dist,
            iters,
        )
        result = o3d.pipelines.registration.registration_icp(
            source_down,
            target_down,
            dist,
            current_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters),
        )
        current_transform = result.transformation
        logging.info(
            "Local ICP (Open3D) level %d result: fitness=%.4f, inlier_rmse=%.4f",
            level,
            result.fitness,
            result.inlier_rmse,
        )

    return result


def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    source_points = np.asarray(source_down.points)
    target_points = np.asarray(target_down.points)

    target_normals = np.asarray(target_down.normals) 
    assert target_normals is not None

    tree = o3d.geometry.KDTreeFlann(target_down)
    T = np.array(trans_init, dtype=np.float64)

    iteration_schedule = [30, 15, 10]
    threshold_schedule = [4.0, 2.5, 1]
    prev_error = np.inf
    correspondence_indices = []

    homogeneous = np.hstack((source_points, np.ones((source_points.shape[0], 1))))

    converged = False
    for level, (stage_iters, threshold_factor) in enumerate(zip(iteration_schedule, threshold_schedule), start=1):
        base_threshold = voxel_size * threshold_factor
        logging.info(
            "Local ICP (custom) level %d: base_threshold=%.3f, iterations=%d",
            level,
            base_threshold,
            stage_iters,
        )

        for itr in range(stage_iters):
            transformed = (T @ homogeneous.T).T[:, :3]

            adaptive_factor = max(0.3, 1.0 - itr / stage_iters)
            current_threshold = base_threshold * adaptive_factor
            threshold_sq = current_threshold ** 2

            A_rows = []
            b_rows = []
            correspondence_indices = []

            for idx, point in enumerate(transformed):
                k, nn_idx, nn_dist = tree.search_knn_vector_3d(point, 1)
                if k == 0 or nn_dist[0] > threshold_sq:
                    continue

                tgt_idx = nn_idx[0]
                correspondence_indices.append([idx, tgt_idx])

                normal = target_normals[tgt_idx]
                target_point = target_points[tgt_idx]

                cross = np.cross(point, normal)
                A_rows.append(np.hstack((cross, normal)))
                b_rows.append(np.dot(normal, target_point - point))

            A = np.asarray(A_rows)
            b = np.asarray(b_rows)
            try:
                xi, *_ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                converged = True
                break

            rot_vec = xi[:3]
            trans_vec = xi[3:]

            delta_R = so3_exp(rot_vec)
            delta_T = np.eye(4)
            delta_T[:3, :3] = delta_R
            delta_T[:3, 3] = trans_vec

            T = delta_T @ T

            residual = b - A @ xi
            mean_error = float(np.sqrt(np.mean(residual ** 2))) if residual.size else 0.0

            if np.linalg.norm(xi) < 1e-5 or abs(prev_error - mean_error) < 1e-6:
                prev_error = mean_error
                converged = True
                logging.info(
                    "Local ICP (custom) level %d converged at iter %d with mean_error=%.6f",
                    level,
                    itr + 1,
                    mean_error,
                )
                break
            prev_error = mean_error

        if converged:
            logging.info("Local ICP (custom) exiting after level %d", level)
            break
        else:
            logging.info("Local ICP (custom) level %d completed without convergence", level)

    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = T
    result.fitness = len(correspondence_indices) / max(len(source_points), 1)
    result.inlier_rmse = prev_error if np.isfinite(prev_error) else 0.0
    result.correspondence_set = o3d.utility.Vector2iVector(correspondence_indices)
    return result


FOV_DEG = 90.0

def reconstruct(args):
    data_root = Path(args.data_root)
    rgb_dir = data_root / "rgb"
    depth_dir = data_root / "depth"

    frame_ids = sorted(int(p.stem) for p in rgb_dir.glob("*.png") if (depth_dir / p.name).exists())

    voxel_size = 0.125 #0.125

    def load_frame(frame_id):
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
        logging.info("Processing frame %d (previous frame %d).", curr_idx, prev_idx)
        source_pcd = load_frame(curr_idx)
        source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)

        trans_init = np.eye(4)
        if not args.disable_global_registration:
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
    parser.add_argument(
        '--viewer',
        type=str,
        choices=['bev', '3d'],
        default='bev',
        help='Choose between top-down BEV or full 3D visualization.',
    )
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

    if args.viewer == '3d':
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
                    color=(0.0, 1.0, 0.0),
                )
            )
            geometries.append(
                create_marker(
                    pred_cam_pos_aligned[-1],
                    radius=0.05,
                    color=(1.0, 0.0, 0.0),
                )
            )

        if len(gt_cam_pos):
            geometries.append(
                create_marker(
                    gt_cam_pos[0],
                    radius=0.05,
                    color=(0.0, 1.0, 1.0),
                )
            )
            geometries.append(
                create_marker(
                    gt_cam_pos[-1],
                    radius=0.05,
                    color=(0.0, 0.0, 1.0),
                )
            )

        o3d.visualization.draw_geometries(geometries)
    else:
        visualize_bev_projection(result_pcd, pred_cam_pos_aligned, gt_cam_pos)
