import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib optional in runtime
    plt = None


# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value_lower = value.strip().lower()
    if value_lower in {"true", "1", "yes", "y"}:
        return True
    if value_lower in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret boolean value from '{value}'.")


# -----------------------------------------------------------------------------
# Intrinsics loading
# -----------------------------------------------------------------------------

DEFAULT_INTRINSICS_DATA: Dict[str, float] = {
    "width": 512,
    "height": 512,
    "fx": 256.0,
    "fy": 256.0,
    "cx": 256.0,
    "cy": 256.0,
}


def _intrinsics_from_mapping(
    data: Dict[str, float], origin: str
) -> Tuple[o3d.camera.PinholeCameraIntrinsic, int, int]:
    required = ["width", "height", "fx", "fy", "cx", "cy"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Intrinsics source '{origin}' missing keys: {missing}")

    try:
        width = int(float(data["width"]))
        height = int(float(data["height"]))
        fx = float(data["fx"])
        fy = float(data["fy"])
        cx = float(data["cx"])
        cy = float(data["cy"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid intrinsic value in '{origin}': {exc}") from exc

    intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    return intr, width, height


def load_intrinsics(path: Path) -> Tuple[o3d.camera.PinholeCameraIntrinsic, int, int]:
    """
    Reads JSON/YAML with keys: width, height, fx, fy, cx, cy.
    Returns (o3d.camera.PinholeCameraIntrinsic, width, height).
    """

    path = Path(path)
    data: Dict[str, float]
    origin: str

    if path.is_file():
        text = path.read_text()
        ext = path.suffix.lower()
        if ext == ".json":
            data = json.loads(text)
        elif ext in {".yaml", ".yml"}:
            data = _load_simple_yaml(text)
        else:
            raise ValueError("Intrinsics file must be JSON or YAML.")
        origin = str(path)
    elif path.exists():
        logging.warning(
            "Intrinsics path '%s' exists but is not a file; using embedded defaults.",
            path,
        )
        data = dict(DEFAULT_INTRINSICS_DATA)
        origin = "embedded defaults"
    elif path.name:
        logging.warning(
            "Intrinsics file '%s' not found; using embedded defaults.", path
        )
        data = dict(DEFAULT_INTRINSICS_DATA)
        origin = "embedded defaults"
    else:
        logging.info("No intrinsics path supplied; using embedded defaults.")
        data = dict(DEFAULT_INTRINSICS_DATA)
        origin = "embedded defaults"

    return _intrinsics_from_mapping(data, origin)


def _load_simple_yaml(text: str) -> Dict[str, float]:
    """Minimal YAML parser for simple key:value pairs."""
    result: Dict[str, float] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        try:
            result[key] = float(value)
        except ValueError:
            result[key] = value  # allow width/height to be integers later
    return result


# -----------------------------------------------------------------------------
# Depth preprocessing
# -----------------------------------------------------------------------------


def median_filter_depth(depth: np.ndarray, kernel: int = 3) -> np.ndarray:
    if kernel <= 1:
        return depth
    pad = kernel // 2
    padded = np.pad(depth, pad_width=pad, mode="edge")
    filtered = np.empty_like(depth)
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            window = padded[i : i + kernel, j : j + kernel]
            filtered[i, j] = np.median(window)
    return filtered


def preprocess_depth(depth: np.ndarray, depth_scale: float, max_depth: float) -> np.ndarray:
    if depth_scale <= 0.0:
        raise ValueError("Depth scale must be positive.")
    depth_float = depth.astype(np.float32) / depth_scale
    depth_float[np.isnan(depth_float)] = 0.0
    depth_float[np.isinf(depth_float)] = 0.0
    depth_float[depth_float < 0.0] = 0.0
    if max_depth > 0.0:
        depth_float[depth_float > max_depth] = 0.0
    filtered = median_filter_depth(depth_float, kernel=3)
    filtered[depth_float == 0.0] = 0.0
    return filtered


# -----------------------------------------------------------------------------
# Point cloud helpers
# -----------------------------------------------------------------------------


def rgbd_to_point_cloud(
    rgb: np.ndarray, depth_m: np.ndarray, intr: o3d.camera.PinholeCameraIntrinsic
) -> o3d.geometry.PointCloud:
    if depth_m.shape != rgb.shape[:2]:
        raise ValueError("RGB and depth resolution mismatch.")
    color_img = o3d.geometry.Image(rgb.astype(np.uint8))
    depth_img = o3d.geometry.Image(depth_m.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_img,
        depth_img,
        depth_scale=1.0,
        depth_trunc=float(np.max(depth_m) + 1.0) if np.any(depth_m) else 10.0,
        convert_rgb_to_intensity=False,
    )
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)


def preprocess_pcd(pcd: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    # voxel downsample, estimate normals, orient normals consistently
    if voxel <= 0.0:
        raise ValueError("Voxel size must be positive.")
    if len(pcd.points) == 0:
        return o3d.geometry.PointCloud()
    pcd_ds = pcd.voxel_down_sample(max(voxel, 1e-4))
    if len(pcd_ds.points) == 0:
        return pcd_ds
    radius_normal = voxel * 3.0
    pcd_ds.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=60)
    )
    pcd_ds.orient_normals_towards_camera_location(np.zeros(3))
    return pcd_ds


def compute_fpfh(pcd_ds: o3d.geometry.PointCloud, voxel: float):
    # FPFH with radius 5*voxel
    if len(pcd_ds.points) == 0:
        return None
    radius_feature = voxel * 10.0
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd_ds,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )


# -----------------------------------------------------------------------------
# Coarse registration
# -----------------------------------------------------------------------------


def coarse_register(
    src_ds: o3d.geometry.PointCloud,
    tgt_ds: o3d.geometry.PointCloud,
    src_fpfh,
    tgt_fpfh,
    voxel: float,
    use_fgr: bool = True,
    max_pairs: int = 20000,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Try Fast Global Registration (Open3D). If it fails (few inliers, bad fitness),
    fallback to RANSAC-based matching with distance and edge-length checks.
    Return 4x4 transform.
    """

    def limit_cloud_and_feature(
        cloud: o3d.geometry.PointCloud, feature, limit: int
    ) -> Tuple[o3d.geometry.PointCloud, Optional[o3d.pipelines.registration.Feature]]:
        if limit <= 0 or len(cloud.points) <= limit:
            return cloud, feature
        indices = np.random.choice(len(cloud.points), size=limit, replace=False)
        limited_cloud = cloud.select_by_index(indices)
        if feature is None:
            return limited_cloud, feature
        limited_feature = o3d.pipelines.registration.Feature()
        limited_feature.data = o3d.utility.MatrixXd(
            np.asarray(feature.data)[:, indices]
        )
        return limited_cloud, limited_feature

    src_use, src_feat_use = limit_cloud_and_feature(src_ds, src_fpfh, max_pairs)
    tgt_use, tgt_feat_use = limit_cloud_and_feature(tgt_ds, tgt_fpfh, max_pairs)

    threshold = voxel * 3.0
    best_trans = np.eye(4)
    best_fitness = 0.0
    best_rmse = float("inf")
    method_used = "none"

    if use_fgr:
        try:
            option = o3d.pipelines.registration.FastGlobalRegistrationOption()
            option.maximum_correspondence_distance = threshold
            option.iteration_number = 64
            option.decrease_mu = True
            fgr_result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                src_use, tgt_use, src_feat_use, tgt_feat_use, option
            )
            if fgr_result is not None:
                best_trans = fgr_result.transformation
                best_fitness = float(fgr_result.fitness)
                best_rmse = float(fgr_result.inlier_rmse)
                method_used = "fgr"
                logging.info(
                    "FGR coarse alignment fitness=%.3f rmse=%.4f",
                    best_fitness,
                    best_rmse,
                )
                if best_fitness < 0.97:
                    logging.info(
                        "FGR fitness %.3f < 0.97; falling back to RANSAC.", best_fitness
                    )
            else:
                logging.info("FGR returned None, falling back to RANSAC.")
        except RuntimeError as exc:
            logging.warning("FGR failed with %s. Falling back to RANSAC.", exc)

    if method_used != "fgr" or best_fitness < 0.97:
        distance_threshold = threshold
        ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_use,
            tgt_use,
            src_feat_use,
            tgt_feat_use,
            True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            4,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold
                ),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(40000, 500),
        )
        best_trans = ransac_result.transformation
        best_fitness = float(ransac_result.fitness)
        best_rmse = float(ransac_result.inlier_rmse)
        method_used = "ransac"
        logging.info(
            "RANSAC coarse alignment fitness=%.3f rmse=%.4f", best_fitness, best_rmse
        )
        if best_fitness < 0.05:
            logging.warning(
                "Coarse registration fitness %.3f is extremely low; downstream alignment may fail.",
                best_fitness,
            )
        assert (
            best_fitness >= 0.5
        ), f"Coarse registration fitness {best_fitness:.3f} below 0.2 threshold."

    return best_trans, {"fitness": best_fitness, "rmse": best_rmse, "method": method_used}


# -----------------------------------------------------------------------------
# Robust ICP ingredients
# -----------------------------------------------------------------------------


def huber_weight(residual: float, delta: float) -> float:
    a = abs(residual)
    return 1.0 if a <= delta else (delta / a)


def find_correspondences_knn(
    src_pts: np.ndarray,
    tgt_kdtree,
    tgt_points: np.ndarray,
    k: int,
    max_dist: float,
    tgt_normals: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    For each src point, query k-NN in target. Keep the neighbor with
    smallest point-to-plane residual (you’ll have target normals).
    Reject if Euclidean distance > max_dist.
    Return list of (i_src, i_tgt) indices.
    """

    correspondences: List[Tuple[int, int]] = []
    if len(src_pts) == 0 or tgt_normals.shape[0] == 0:
        return correspondences

    max_dist_sq = max_dist * max_dist
    effective_k = max(1, min(k, tgt_normals.shape[0]))
    for i, pt in enumerate(src_pts):
        k_count, idxs, dists = tgt_kdtree.search_knn_vector_3d(pt, effective_k)
        if k_count == 0:
            continue
        best_idx = -1
        best_residual = float("inf")
        for j in range(k_count):
            tgt_index = idxs[j]
            if dists[j] > max_dist_sq:
                continue
            normal = tgt_normals[tgt_index]
            norm = np.linalg.norm(normal)
            if norm < 1e-8:
                continue
            normal = normal / norm
            residual = float(np.dot(normal, pt - tgt_points[tgt_index]))
            if abs(residual) < abs(best_residual):
                best_residual = residual
                best_idx = tgt_index
        if best_idx >= 0:
            correspondences.append((i, best_idx))
    return correspondences


# -----------------------------------------------------------------------------
# Multi-scale ICP
# -----------------------------------------------------------------------------


def icp_point_to_plane_multiscale(
    src_raw: o3d.geometry.PointCloud,
    tgt_raw: o3d.geometry.PointCloud,
    T_init: np.ndarray,
    base_voxel: float,
    knn: int,
    iters_fine: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Custom multi-scale ICP that returns the refined transformation and metrics.
    """

    levels = [
        (base_voxel * 3.0, base_voxel * 8.0, 25),
        (base_voxel * 1.5, base_voxel * 4.5, 20),
        (base_voxel * 1.0, base_voxel * 3.0, max(40, iters_fine)),
    ]

    transform = np.array(T_init, dtype=np.float64)
    prev_level_rmse: Optional[float] = None
    level_summaries: List[Dict[str, float]] = []
    last_residuals: List[float] = []
    last_correspondences = 0
    last_source_size = 1

    for level_idx, (voxel, threshold, max_iter) in enumerate(levels):
        src = preprocess_pcd(src_raw, voxel)
        tgt = preprocess_pcd(tgt_raw, voxel)
        src_pts = np.asarray(src.points)
        tgt_pts = np.asarray(tgt.points)
        if src_pts.shape[0] < 200 or tgt_pts.shape[0] < 200:
            logging.warning(
                "Skipping ICP level %d due to insufficient points (src=%d, tgt=%d; need >=200).",
                level_idx,
                src_pts.shape[0],
                tgt_pts.shape[0],
            )
            continue
        tgt_normals = np.asarray(tgt.normals)
        kd_tree = o3d.geometry.KDTreeFlann(tgt)

        def compute_residuals(current_transform: np.ndarray) -> Tuple[List[float], int]:
            transformed = (current_transform[:3, :3] @ src_pts.T).T + current_transform[:3, 3]
            correspondences = find_correspondences_knn(
                transformed, kd_tree, tgt_pts, knn, threshold, tgt_normals
            )
            residuals_local: List[float] = []
            for i_src, i_tgt in correspondences:
                normal = tgt_normals[i_tgt]
                norm = np.linalg.norm(normal)
                if norm < 1e-8:
                    continue
                normal = normal / norm
                q = tgt_pts[i_tgt]
                residuals_local.append(float(np.dot(normal, transformed[i_src] - q)))
            return residuals_local, len(correspondences)

        residuals_init, corr_init = compute_residuals(transform)
        rmse_init = float(
            math.sqrt(np.mean(np.square(residuals_init))) if residuals_init else float("inf")
        )
        rmse_initial_level = rmse_init
        iterations_run = 0
        prev_rmse = rmse_init

        logging.info(
            "ICP level %d (voxel=%.4f) initial RMSE %.6f with %d correspondences.",
            level_idx,
            voxel,
            rmse_init,
            corr_init,
        )

        best_rmse_level = rmse_init
        best_transform_level = np.array(transform, copy=True)
        worsening_streak = 0

        for iteration in range(max_iter):
            transformed = (transform[:3, :3] @ src_pts.T).T + transform[:3, 3]
            correspondences = find_correspondences_knn(
                transformed, kd_tree, tgt_pts, knn, threshold, tgt_normals
            )
            if len(correspondences) < 6:
                logging.warning(
                    "ICP level %d iteration %d has too few correspondences (%d).",
                    level_idx,
                    iteration,
                    len(correspondences),
                )
                break

            JT_WJ = np.zeros((6, 6), dtype=np.float64)
            JT_Wr = np.zeros(6, dtype=np.float64)
            residuals = []

            for i_src, i_tgt in correspondences:
                p = transformed[i_src]
                q = tgt_pts[i_tgt]
                normal = tgt_normals[i_tgt]
                norm = np.linalg.norm(normal)
                if norm < 1e-8:
                    continue
                normal = normal / norm
                residual = float(np.dot(normal, p - q))
                weight = huber_weight(residual, threshold * 0.5)
                row = np.hstack((np.cross(p, normal), normal))
                JT_WJ += weight * np.outer(row, row)
                JT_Wr += weight * residual * row
                residuals.append(residual)

            if not residuals:
                break

            try:
                delta = np.linalg.solve(JT_WJ, -JT_Wr)
            except np.linalg.LinAlgError:
                delta, *_ = np.linalg.lstsq(JT_WJ, -JT_Wr, rcond=None)

            if np.linalg.norm(delta) < 1e-6:
                break

            transform = _se3_exp(delta) @ transform
            rmse_iter = float(math.sqrt(np.mean(np.square(residuals))))
            logging.debug(
                "ICP level %d iteration %d RMSE %.6f",
                level_idx,
                iteration,
                rmse_iter,
            )
            iterations_run = iteration + 1
            if rmse_iter + 1e-9 < best_rmse_level:
                best_rmse_level = rmse_iter
                best_transform_level = np.array(transform, copy=True)

            if rmse_iter > prev_rmse + 1e-6:
                worsening_streak += 1
            else:
                worsening_streak = 0

            if worsening_streak >= 2:
                logging.warning(
                    "ICP level %d RMSE worsened in two consecutive iterations; reverting to best estimate.",
                    level_idx,
                )
                transform = best_transform_level
                prev_rmse = best_rmse_level
                break

            if abs(rmse_iter - prev_rmse) < 1e-7:
                prev_rmse = rmse_iter
                break

            prev_rmse = rmse_iter

        residuals_final, corr_final = compute_residuals(transform)
        rmse_final = float(
            math.sqrt(np.mean(np.square(residuals_final))) if residuals_final else rmse_init
        )
        level_summaries.append(
            {"voxel": voxel, "rmse": rmse_final, "correspondences": float(corr_final)}
        )

        logging.info(
            "ICP level %d (voxel=%.4f) RMSE %.6f -> %.6f (%d iters).",
            level_idx,
            voxel,
            rmse_initial_level,
            rmse_final,
            iterations_run,
        )

        if prev_level_rmse is not None and rmse_final > prev_level_rmse + 1e-4:
            logging.warning(
                "ICP RMSE increased from %.6f to %.6f at level %d; continuing.",
                prev_level_rmse,
                rmse_final,
                level_idx,
            )
        prev_level_rmse = rmse_final
        last_residuals = residuals_final
        last_correspondences = corr_final
        last_source_size = src_pts.shape[0]

    fitness = last_correspondences / max(1, last_source_size)
    rmse_total = float(
        math.sqrt(np.mean(np.square(last_residuals))) if last_residuals else float("inf")
    )
    metrics = {
        "rmse": rmse_total,
        "fitness": fitness,
        "levels": level_summaries,
        "correspondences": last_correspondences,
    }
    return transform, metrics


def icp_point_to_plane_open3d(
    src_raw: o3d.geometry.PointCloud,
    tgt_raw: o3d.geometry.PointCloud,
    T_init: np.ndarray,
    base_voxel: float,
    knn: int,
    iters_fine: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Multi-scale wrapper around Open3D's built-in point-to-plane ICP.
    """

    levels = [
        (base_voxel * 3.0, base_voxel * 8.0, 25),
        (base_voxel * 1.5, base_voxel * 4.5, 20),
        (base_voxel * 1.0, base_voxel * 3.0, max(40, iters_fine)),
    ]

    transform = np.array(T_init, dtype=np.float64)
    level_summaries: List[Dict[str, float]] = []
    overall_fitness = 0.0
    overall_rmse = float("inf")
    overall_correspondences = 0.0

    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    for level_idx, (voxel, threshold, max_iter) in enumerate(levels):
        src = preprocess_pcd(src_raw, voxel)
        tgt = preprocess_pcd(tgt_raw, voxel)
        if len(src.points) < 200 or len(tgt.points) < 200:
            logging.warning(
                "Skipping Open3D ICP level %d due to insufficient points (src=%d, tgt=%d; need >=200).",
                level_idx,
                len(src.points),
                len(tgt.points),
            )
            continue

        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter
        )
        reg_result = o3d.pipelines.registration.registration_icp(
            src,
            tgt,
            threshold,
            transform,
            estimation,
            criteria,
        )
        transform = reg_result.transformation
        overall_fitness = float(reg_result.fitness)
        overall_rmse = float(reg_result.inlier_rmse)
        overall_correspondences = float(len(reg_result.correspondence_set))

        level_summaries.append(
            {"voxel": float(voxel), "rmse": overall_rmse, "correspondences": overall_correspondences}
        )
        logging.info(
            "Open3D ICP level %d (voxel=%.4f) fitness %.4f rmse %.6f (%d correspondences).",
            level_idx,
            voxel,
            reg_result.fitness,
            reg_result.inlier_rmse,
            len(reg_result.correspondence_set),
        )

    metrics = {
        "rmse": overall_rmse,
        "fitness": overall_fitness,
        "levels": level_summaries,
        "correspondences": overall_correspondences,
    }
    return transform, metrics


def _skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = vec
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _se3_exp(xi: np.ndarray) -> np.ndarray:
    omega = xi[:3]
    v = xi[3:]
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        R = np.eye(3) + _skew(omega)
        V = np.eye(3) + 0.5 * _skew(omega)
    else:
        omega_unit = omega / theta
        omega_hat = _skew(omega_unit)
        R = (
            np.eye(3)
            + math.sin(theta) * omega_hat
            + (1.0 - math.cos(theta)) * (omega_hat @ omega_hat)
        )
        V = (
            np.eye(3)
            + (1.0 - math.cos(theta)) / theta * omega_hat
            + (theta - math.sin(theta)) / (theta ** 2) * (omega_hat @ omega_hat)
        )
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = V @ v
    return transform


# -----------------------------------------------------------------------------
# Frame bookkeeping and utilities
# -----------------------------------------------------------------------------


@dataclass
class FrameRecord:
    frame_id: int
    rgb: np.ndarray
    depth: np.ndarray
    raw_pcd: o3d.geometry.PointCloud
    pose_world: np.ndarray
    coarse_pose_world: np.ndarray
    pair_metric: Optional[Dict[str, float]] = None


def _sorted_frame_ids(directory: Path) -> List[int]:
    ids = []
    for file in directory.glob("*.png"):
        try:
            ids.append(int(file.stem))
        except ValueError:
            continue
    return sorted(ids)


def accumulate_point_cloud(
    frames: Iterable[FrameRecord], pose_attr: str = "pose_world"
) -> o3d.geometry.PointCloud:
    accumulated = o3d.geometry.PointCloud()
    for frame in frames:
        if hasattr(frame.raw_pcd, "clone"):
            world_pcd = frame.raw_pcd.clone()
        else:
            world_pcd = o3d.geometry.PointCloud(frame.raw_pcd)
        pose = getattr(frame, pose_attr)
        world_pcd.transform(pose)
        accumulated += world_pcd
    return accumulated


# -----------------------------------------------------------------------------
# Debug exports
# -----------------------------------------------------------------------------


def _export_stage_point_clouds(
    pcd: o3d.geometry.PointCloud,
    output_dir: Path,
    file_basenames: Sequence[str],
    stage_description: str,
) -> List[Path]:
    """
    Write the same point cloud to multiple filenames to aid debugging.
    Returns the file paths that were written.
    """

    if len(pcd.points) == 0:
        logging.warning("Skipping %s export: point cloud is empty.", stage_description)
        return []

    written_paths: List[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for basename in file_basenames:
        path = output_dir / f"{basename}.ply"
        o3d.io.write_point_cloud(str(path), pcd)
        written_paths.append(path)

    logging.info(
        "Exported %s point cloud (%d pts) to %s.",
        stage_description,
        len(pcd.points),
        ", ".join(str(path) for path in written_paths),
    )
    return written_paths


def _umeyama_alignment(
    source: np.ndarray, target: np.ndarray, allow_scaling: bool = True
) -> Tuple[float, np.ndarray, np.ndarray]:
    if source.shape != target.shape:
        raise ValueError("Source and target must share the same shape.")
    n = source.shape[0]
    if n == 0:
        return 1.0, np.eye(3), np.zeros(3)

    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    src_demean = source - mu_s
    tgt_demean = target - mu_t

    cov = (tgt_demean.T @ src_demean) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt

    var_src = np.mean(np.sum(src_demean ** 2, axis=1))
    if allow_scaling and var_src > 1e-9:
        scale = np.sum(D * np.diag(S)) / var_src
    else:
        scale = 1.0

    t = mu_t - scale * (R @ mu_s)
    return float(scale), R, t


def _apply_similarity_transform(
    points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    if points.size == 0:
        return points.copy()
    return scale * (points @ rotation.T) + translation


def _transform_point_cloud_similarity(
    pcd: o3d.geometry.PointCloud, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> o3d.geometry.PointCloud:
    if len(pcd.points) == 0:
        return o3d.geometry.PointCloud()
    transformed = o3d.geometry.PointCloud()
    pts = np.asarray(pcd.points)
    transformed_points = _apply_similarity_transform(pts, scale, rotation, translation)
    transformed.points = o3d.utility.Vector3dVector(transformed_points)
    if pcd.has_colors():
        transformed.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        rotated_normals = (normals @ rotation.T)
        transformed.normals = o3d.utility.Vector3dVector(rotated_normals)
    return transformed


def _create_trajectory_line_set(
    trajectory: np.ndarray, color: Tuple[float, float, float]
) -> Optional[o3d.geometry.LineSet]:
    if trajectory.size == 0 or trajectory.shape[0] < 2:
        return None
    lines = [[i, i + 1] for i in range(trajectory.shape[0] - 1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(trajectory)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    colors = np.tile(np.asarray(color, dtype=float), (len(lines), 1))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


@dataclass
class PlaneModel:
    normal: np.ndarray
    offset: float
    inlier_count: int
    mean_height: float


def _fit_ceiling_plane(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.03,
    ransac_n: int = 3,
    num_iterations: int = 2000,
    top_percentile: float = 65.0,
    min_inlier_ratio: float = 0.05,
    max_planes: int = 4,
) -> Optional[PlaneModel]:
    if len(pcd.points) < ransac_n:
        return None
    points = np.asarray(pcd.points)
    if points.size == 0:
        return None
    vertical_coords = points[:, 1]
    percentile = float(np.clip(top_percentile, 0.0, 100.0))
    mask = vertical_coords >= np.percentile(vertical_coords, percentile) if percentile > 0 else np.ones_like(vertical_coords, dtype=bool)
    candidate_points = points[mask]
    if candidate_points.shape[0] < ransac_n:
        candidate_points = points
    candidate_pcd = o3d.geometry.PointCloud()
    candidate_pcd.points = o3d.utility.Vector3dVector(candidate_points)

    best: Optional[PlaneModel] = None
    best_score: Optional[Tuple[float, float, int]] = None
    remaining = candidate_pcd
    min_inliers = max(int(min_inlier_ratio * len(points)), ransac_n)

    for _ in range(max_planes):
        if len(remaining.points) < ransac_n:
            break
        plane = remaining.segment_plane(distance_threshold, ransac_n, num_iterations)
        if plane is None:
            break
        plane_model, inliers = plane
        if not inliers:
            break
        inliers_arr = np.asarray(remaining.points)[inliers]
        if inliers_arr.shape[0] < min_inliers:
            remaining = remaining.select_by_index(inliers, invert=True)
            continue
        normal = np.asarray(plane_model[:3], dtype=float)
        offset = float(plane_model[3])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            remaining = remaining.select_by_index(inliers, invert=True)
            continue
        normal /= norm
        offset /= norm
        if normal[1] < 0.0:
            normal = -normal
            offset = -offset
        mean_height = float(np.mean(inliers_arr[:, 1]))
        score = (float(abs(normal[1])), mean_height, len(inliers_arr))
        if best_score is None or score > best_score:
            best_score = score
            best = PlaneModel(normal=normal, offset=offset, inlier_count=len(inliers_arr), mean_height=mean_height)
        remaining = remaining.select_by_index(inliers, invert=True)
        if len(remaining.points) < min_inliers:
            break
    return best


def _remove_ceiling_points(
    pcd: o3d.geometry.PointCloud,
    plane_distance: float = 1.1,
    ransac_distance: float = 0.03,
    min_inlier_ratio: float = 0.2,
    fallback_percentile: float = 70.0,
) -> o3d.geometry.PointCloud:
    """Fit the dominant horizontal plane near the ceiling and remove its inliers."""
    if len(pcd.points) == 0:
        return o3d.geometry.PointCloud()
    plane = _fit_ceiling_plane(
        pcd,
        distance_threshold=ransac_distance,
        min_inlier_ratio=min_inlier_ratio,
    )
    coords = np.asarray(pcd.points)
    if plane is not None:
        distances = np.abs(coords @ plane.normal + plane.offset)
        mask = distances > plane_distance
        if np.any(mask):
            filtered = o3d.geometry.PointCloud()
            filtered.points = o3d.utility.Vector3dVector(coords[mask])
            if pcd.has_colors():
                filtered.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
            if pcd.has_normals():
                filtered.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[mask])
            logging.info(
                "Removed %d ceiling points using plane model (mean height %.3f m).",
                plane.inlier_count,
                plane.mean_height,
            )
            return filtered
        logging.warning(
            "Plane-based ceiling removal removed all points; falling back to percentile filter."
        )
    percentile = float(np.clip(fallback_percentile, 0.0, 100.0))
    if percentile <= 0.0:
        return pcd
    vertical = coords[:, 1]
    try:
        threshold = float(np.percentile(vertical, percentile))
    except IndexError:
        threshold = float(np.max(vertical))
    mask = vertical <= threshold
    if not np.any(mask):
        logging.warning("Ceiling removal kept no points; keeping original cloud.")
        return pcd
    filtered = o3d.geometry.PointCloud()
    filtered.points = o3d.utility.Vector3dVector(coords[mask])
    if pcd.has_colors():
        filtered.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    if pcd.has_normals():
        filtered.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[mask])
    return filtered


def _save_bev_plot(
    pcd: o3d.geometry.PointCloud,
    estimated_traj: np.ndarray,
    gt_traj: np.ndarray,
    output_path: Path,
) -> None:
    if plt is None:
        logging.warning("Matplotlib not available; skipping BEV plot generation.")
        return
    scene_points = np.asarray(pcd.points)
    if scene_points.size == 0:
        logging.warning("Aligned point cloud is empty; skipping BEV plot.")
        return
    point_colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#f9f9f9")

    x_coords = scene_points[:, 0]
    z_coords = scene_points[:, 2]
    mask = np.isfinite(x_coords) & np.isfinite(z_coords)
    x_coords = x_coords[mask]
    z_coords = z_coords[mask]
    if point_colors is not None and point_colors.shape[0] == scene_points.shape[0]:
        point_colors = point_colors[mask]
    else:
        point_colors = None

    if x_coords.size and z_coords.size:
        x_min, x_max = np.percentile(x_coords, [1.0, 99.0])
        z_min, z_max = np.percentile(z_coords, [1.0, 99.0])
        if not np.isfinite(x_min) or not np.isfinite(x_max):
            x_min, x_max = float(np.min(x_coords)), float(np.max(x_coords))
        if not np.isfinite(z_min) or not np.isfinite(z_max):
            z_min, z_max = float(np.min(z_coords)), float(np.max(z_coords))
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        if z_min == z_max:
            z_min -= 0.5
            z_max += 0.5
        margin_x = 0.05 * (x_max - x_min)
        margin_z = 0.05 * (z_max - z_min)
        hist_bins = int(np.clip(np.sqrt(x_coords.size), 64, 512))
        x_range = (x_min - margin_x, x_max + margin_x)
        z_range = (z_min - margin_z, z_max + margin_z)
        x_edges = np.linspace(x_range[0], x_range[1], hist_bins + 1, dtype=np.float32)
        z_edges = np.linspace(z_range[0], z_range[1], hist_bins + 1, dtype=np.float32)

        if point_colors is not None and point_colors.shape[0] == x_coords.shape[0]:
            face_rgb = np.array(ax.get_facecolor()[:3], dtype=np.float32)
            color_accum = np.zeros((hist_bins, hist_bins, 3), dtype=np.float32)
            counts = np.zeros((hist_bins, hist_bins), dtype=np.int32)

            x_idx = np.clip(np.digitize(x_coords, x_edges) - 1, 0, hist_bins - 1)
            z_idx = np.clip(np.digitize(z_coords, z_edges) - 1, 0, hist_bins - 1)
            np.add.at(color_accum, (z_idx, x_idx), point_colors.astype(np.float32))
            np.add.at(counts, (z_idx, x_idx), 1)

            valid = counts > 0
            color_grid = np.tile(face_rgb, (hist_bins, hist_bins, 1))
            color_grid[valid] = color_accum[valid] / counts[valid][..., None]
            ax.imshow(
                color_grid,
                extent=[x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]],
                origin="lower",
                interpolation="nearest",
            )
        else:
            density, _, _ = np.histogram2d(
                x_coords,
                z_coords,
                bins=hist_bins,
                range=[x_range, z_range],
            )
            density = np.log1p(density.T)
            ax.imshow(
                density,
                extent=[x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]],
                origin="lower",
                cmap="viridis",
                alpha=0.85,
            )

    if estimated_traj.size:
        (est_line,) = ax.plot(
            estimated_traj[:, 0],
            estimated_traj[:, 2],
            color="red",
            linewidth=2.0,
        )
        ax.scatter(
            estimated_traj[0, 0],
            estimated_traj[0, 2],
            color="red",
            marker="s",
            s=24,
            zorder=5,
            label=None,
        )
    if gt_traj.size:
        (gt_line,) = ax.plot(
            gt_traj[:, 0],
            gt_traj[:, 2],
            color="black",
            linewidth=2.0,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logging.info("Saved BEV visualization to %s.", output_path)


def _visualize_bev_scene(
    pcd: o3d.geometry.PointCloud,
    estimated_traj: np.ndarray,
    gt_traj: np.ndarray,
) -> None:
    geometries: List[o3d.geometry.Geometry] = []
    if len(pcd.points):
        geometries.append(pcd)
    est_lines = _create_trajectory_line_set(estimated_traj, (1.0, 0.0, 0.0))
    if est_lines is not None:
        geometries.append(est_lines)
    gt_lines = _create_trajectory_line_set(gt_traj, (0.0, 0.0, 0.0))
    if gt_lines is not None:
        geometries.append(gt_lines)
    if not geometries:
        logging.warning("No geometries available for visualization.")
        return
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Ceiling-Removed Reconstruction & Trajectories",
        width=960,
        height=720,
    )
    render_opts = vis.get_render_option()
    if render_opts is not None:
        render_opts.point_size = 2.5
        render_opts.background_color = np.array([1.0, 1.0, 1.0])
    for geom in geometries:
        vis.add_geometry(geom)
    all_points: List[np.ndarray] = []
    if len(pcd.points):
        all_points.append(np.asarray(pcd.points))
    if estimated_traj.size:
        all_points.append(np.asarray(estimated_traj, dtype=float))
    if gt_traj.size:
        all_points.append(np.asarray(gt_traj, dtype=float))
    if all_points:
        stacked = np.vstack(all_points)
        center = np.mean(stacked, axis=0)
        radius = float(np.max(np.linalg.norm(stacked - center, axis=1)))
    else:
        center = np.zeros(3)
        radius = 1.0
    ctrl = vis.get_view_control()
    try:
        ctrl.set_lookat(center.tolist())
        ctrl.set_up([0.0, 0.0, -1.0])
        ctrl.set_front([0.0, -1.0, 0.0])
        if hasattr(ctrl, "set_zoom"):
            zoom = float(np.clip(1.5 / max(radius, 1e-3), 0.15, 1.0))
            ctrl.set_zoom(zoom)
    except Exception:
        logging.debug("Open3D view control adjustments not supported; using defaults.")
    vis.run()
    vis.destroy_window()


def _save_camera_trajectory(
    output_path: Path,
    frame_ids: Sequence[int],
    estimated_traj: np.ndarray,
    aligned_traj: Optional[np.ndarray],
    gt_traj: Optional[np.ndarray],
    alignment: Optional[Dict[str, object]],
) -> None:
    trajectory_payload: Dict[str, object] = {
        "frame_ids": list(frame_ids),
        "estimated_world": np.asarray(estimated_traj, dtype=float).tolist(),
    }
    if aligned_traj is not None:
        trajectory_payload["estimated_aligned"] = np.asarray(aligned_traj, dtype=float).tolist()
    if gt_traj is not None:
        trajectory_payload["ground_truth"] = np.asarray(gt_traj, dtype=float).tolist()
    if alignment is not None:
        trajectory_payload["alignment"] = alignment
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trajectory_payload, indent=2))
    logging.info("Saved camera trajectory to %s.", output_path)


# -----------------------------------------------------------------------------
# Main reconstruction pipeline
# -----------------------------------------------------------------------------


def run_reconstruction(args) -> Dict[str, object]:
    intr, intr_width, intr_height = load_intrinsics(Path(args.intrinsics))
    data_root = Path(args.data_root)
    rgb_dir = data_root / "rgb"
    depth_dir = data_root / "depth"
    if not rgb_dir.exists() or not depth_dir.exists():
        fallback_root = None
        if data_root.exists():
            for candidate in sorted(data_root.iterdir()):
                if not candidate.is_dir():
                    continue
                candidate_rgb = candidate / "rgb"
                candidate_depth = candidate / "depth"
                if candidate_rgb.exists() and candidate_depth.exists():
                    fallback_root = candidate
                    break
        if fallback_root is None:
            raise FileNotFoundError(
                f"Could not locate 'rgb' ({rgb_dir}) and 'depth' ({depth_dir}) folders."
            )
        logging.info(
            "Using RGB-D data from %s as fallback for %s.", fallback_root, data_root
        )
        data_root = fallback_root
        rgb_dir = data_root / "rgb"
        depth_dir = data_root / "depth"

    frame_ids = sorted(set(_sorted_frame_ids(rgb_dir)) & set(_sorted_frame_ids(depth_dir)))
    if not frame_ids:
        raise RuntimeError("No overlapping RGB-D frames found.")

    base_voxel = float(args.voxel_size)
    frames: List[FrameRecord] = []
    pair_metrics: List[Dict[str, float]] = []

    prev_down = None
    prev_fpfh = None
    prev_pose_world = np.eye(4)

    max_depth = 15.0

    icp_solver = (
        icp_point_to_plane_multiscale
        if args.icp_mode == "custom"
        else icp_point_to_plane_open3d
    )

    if args.disable_global_registration:
        logging.info(
            "Global registration disabled; using identity as the initial transform for ICP."
        )

    for idx, frame_id in enumerate(frame_ids):
        rgb_path = rgb_dir / f"{frame_id}.png"
        depth_path = depth_dir / f"{frame_id}.png"

        rgb_img = np.asarray(o3d.io.read_image(str(rgb_path)))
        depth_img = np.asarray(o3d.io.read_image(str(depth_path)))
        if rgb_img.ndim == 2:
            rgb_img = np.repeat(rgb_img[..., None], 3, axis=2)
        depth_m = preprocess_depth(depth_img, args.depth_scale, max_depth=max_depth)

        if rgb_img.shape[1] != intr_width or rgb_img.shape[0] != intr_height:
            raise ValueError(
                f"Frame {frame_id} resolution {rgb_img.shape[1]}x{rgb_img.shape[0]} "
                f"does not match intrinsics ({intr_width}x{intr_height})."
            )

        raw_pcd = rgbd_to_point_cloud(rgb_img, depth_m, intr)
        if len(raw_pcd.points) == 0:
            logging.warning("Frame %d has no valid depth; skipping.", frame_id)
            continue

        down_pcd = preprocess_pcd(raw_pcd, base_voxel)
        fpfh = compute_fpfh(down_pcd, base_voxel)
        if fpfh is None:
            logging.warning("Frame %d: FPFH computation failed; skipping frame.", frame_id)
            continue

        if prev_down is None or prev_fpfh is None:
            pose_world = np.eye(4)
            frames.append(
                FrameRecord(
                    frame_id=frame_id,
                    rgb=rgb_img,
                    depth=depth_m,
                    raw_pcd=raw_pcd,
                    pose_world=pose_world,
                    coarse_pose_world=pose_world,
                )
            )
            prev_down = down_pcd
            prev_fpfh = fpfh
            prev_pose_world = pose_world
            logging.info("Initialized reconstruction with frame %d.", frame_id)
            continue

        if args.disable_global_registration:
            coarse_T = np.eye(4)
            coarse_stats = {"method": "disabled", "fitness": 0.0, "rmse": 0.0}
        else:
            coarse_T, coarse_stats = coarse_register(
                down_pcd,
                prev_down,
                fpfh,
                prev_fpfh,
                base_voxel,
                use_fgr=args.use_fgr,
                max_pairs=args.max_pairs,
            )

        coarse_pose_world = prev_pose_world @ coarse_T

        refined_T, icp_stats = icp_solver(
            down_pcd,
            prev_down,
            coarse_T,
            base_voxel,
            args.knn,
            args.icp_iters,
        )

        pose_world = prev_pose_world @ refined_T
        pair_metric = {
            "frame_id": frame_id,
            "fitness": float(icp_stats["fitness"]),
            "rmse": float(icp_stats["rmse"]),
            "correspondences": float(icp_stats["correspondences"]),
        }
        logging.info(
            "Frame %d registration: fitness=%.4f rmse=%.5f (coarse %s %.3f).",
            frame_id,
            pair_metric["fitness"],
            pair_metric["rmse"],
            coarse_stats["method"],
            coarse_stats["fitness"],
        )
        pair_metrics.append(pair_metric)

        frames.append(
            FrameRecord(
                frame_id=frame_id,
                rgb=rgb_img,
                depth=depth_m,
                raw_pcd=raw_pcd,
                pose_world=pose_world,
                coarse_pose_world=coarse_pose_world,
                pair_metric=pair_metric,
            )
        )

        prev_down = down_pcd
        prev_fpfh = fpfh
        prev_pose_world = pose_world

    raw_accumulated = accumulate_point_cloud(frames)
    coarse_accumulated = accumulate_point_cloud(frames, pose_attr="coarse_pose_world")
    optimized_accumulated = accumulate_point_cloud(frames)

    return {
        "frames": frames,
        "raw_frames": frames,
        "pair_metrics": pair_metrics,
        "raw_pcd": raw_accumulated,
        "coarse_pcd": coarse_accumulated,
        "optimized_pcd": optimized_accumulated,
        "data_root": data_root,
    }


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robust RGB-D reconstruction pipeline.")
    parser.add_argument("--intrinsics", type=str, default="intrinsics.yaml", help="Path to intrinsics JSON/YAML.")
    parser.add_argument(
        "--floor",
        type=str,
        choices=["first_floor", "second_floor"],
        default="first_floor",
        help="Select floor dataset under data_collection/.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override dataset directory (defaults to data_collection/<floor>).",
    )
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--knn", type=int, default=3)
    parser.add_argument("--icp-iters", type=int, default=20)
    parser.add_argument(
        "--icp-mode",
        type=str,
        choices=["custom", "open3d"],
        default="custom",
        help="Choose ICP backend: 'custom' multiscale implementation or Open3D local ICP.",
    )
    parser.add_argument("--use-fgr", type=str2bool, default=True)
    parser.add_argument(
        "--disable-global-registration",
        action="store_true",
        help="Skip the coarse global registration stage and initialize ICP with identity.",
    )
    parser.add_argument("--output-dir", type=str, default="./out")
    parser.add_argument("--max-pairs", type=int, default=20000)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    floor_name = args.floor
    if args.data_root is None:
        data_root_path = Path("data_collection") / floor_name
    else:
        data_root_path = Path(args.data_root)
    args.data_root = str(data_root_path)

    output_dir_path = Path(args.output_dir) / floor_name
    args.output_dir = str(output_dir_path)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info("Using dataset root %s.", args.data_root)
    logging.info("Saving outputs under %s.", args.output_dir)

    result = run_reconstruction(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_suffix = f"_{args.icp_mode}"

    raw_pcd: o3d.geometry.PointCloud = result["raw_pcd"]  # type: ignore[assignment]
    coarse_pcd: o3d.geometry.PointCloud = result["coarse_pcd"]  # type: ignore[assignment]
    optimized_pcd: o3d.geometry.PointCloud = result["optimized_pcd"]  # type: ignore[assignment]

    stage_suffix = output_suffix

    _export_stage_point_clouds(
        coarse_pcd,
        output_dir,
        [f"reconstruction_coarse{stage_suffix}"],
        "global registration (coarse alignment)",
    )

    frames: List[FrameRecord] = result["frames"]  # type: ignore[assignment]
    frame_ids = [frame.frame_id for frame in frames]
    estimated_positions = (
        np.array([frame.pose_world[:3, 3] for frame in frames], dtype=float)
        if frames
        else np.zeros((0, 3), dtype=float)
    )

    pair_metrics: List[Dict[str, float]] = result["pair_metrics"]  # type: ignore[assignment]
    if pair_metrics:
        avg_fitness = np.mean([m["fitness"] for m in pair_metrics])
        avg_rmse = np.mean([m["rmse"] for m in pair_metrics])
        logging.info(
            "Average pair metrics: fitness=%.4f rmse=%.5f over %d pairs.",
            avg_fitness,
            avg_rmse,
            len(pair_metrics),
        )

    data_root = Path(result.get("data_root", args.data_root))  # type: ignore[arg-type]
    gt_pose_path = data_root / "GT_pose.npy"
    gt_positions_arr: Optional[np.ndarray] = None
    aligned_positions_arr: Optional[np.ndarray] = None
    alignment_info: Optional[Dict[str, object]] = None

    if gt_pose_path.exists() and frames:
        gt_all = np.load(gt_pose_path)
        indices = np.clip(np.array(frame_ids) - 1, 0, gt_all.shape[0] - 1)
        gt_positions_arr = np.asarray(gt_all[indices, :3], dtype=float)
        scale, rotation, translation = _umeyama_alignment(
            estimated_positions, gt_positions_arr, allow_scaling=True
        )
        aligned_positions_arr = _apply_similarity_transform(
            estimated_positions, scale, rotation, translation
        )
        l2_errors = np.linalg.norm(aligned_positions_arr - gt_positions_arr, axis=1)
        mean_l2 = float(np.mean(l2_errors))
        max_l2 = float(np.max(l2_errors))
        logging.info("Mean L2 distance (aligned) = %.4f m.", mean_l2)
        logging.info("Max L2 distance (aligned) = %.4f m.", max_l2)
        alignment_info = {
            "scale": float(scale),
            "rotation": np.asarray(rotation, dtype=float).tolist(),
            "translation": np.asarray(translation, dtype=float).tolist(),
        }

        aligned_pcd = _transform_point_cloud_similarity(
            optimized_pcd, scale, rotation, translation
        )
        if len(aligned_pcd.points):
            aligned_path = output_dir / f"reconstruction_aligned{output_suffix}.ply"
            o3d.io.write_point_cloud(str(aligned_path), aligned_pcd)
            logging.info("Saved aligned reconstruction to %s.", aligned_path)
        else:
            aligned_pcd = o3d.geometry.PointCloud()

        filtered_pcd = _remove_ceiling_points(aligned_pcd)
        if len(filtered_pcd.points) == 0 and len(aligned_pcd.points):
            filtered_pcd = aligned_pcd

        bev_image_path = output_dir / f"bev_scene{output_suffix}.png"
        _save_bev_plot(filtered_pcd, aligned_positions_arr, gt_positions_arr, bev_image_path)
        try:
            _visualize_bev_scene(filtered_pcd, aligned_positions_arr, gt_positions_arr)
        except Exception as exc:
            logging.warning("Open3D visualization failed: %s", exc)
    else:
        if frames:
            logging.warning("Ground truth poses not found at %s; skipping alignment.", gt_pose_path)
        else:
            logging.warning("No frames available; skipping trajectory alignment and visualization.")

    if frames:
        _save_camera_trajectory(
            output_dir / f"camera_trajectory{output_suffix}.json",
            frame_ids,
            estimated_positions,
            aligned_positions_arr,
            gt_positions_arr,
            alignment_info,
        )


if __name__ == "__main__":
    main()
