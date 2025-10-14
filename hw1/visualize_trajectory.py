#!/usr/bin/env python3
"""
Visualize a reconstructed scene alongside a camera trajectory.

The script removes the ceiling from the input point cloud, renders an
interactive 3D view with the camera path, and produces a bird's-eye-view
projection of the scene with the same trajectory overlayed.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d

from reconstruct import (
    _apply_similarity_transform,
    _create_trajectory_line_set,
    _remove_ceiling_points,
    _save_bev_plot,
    _transform_point_cloud_similarity,
    _visualize_bev_scene,
)


def _empty_traj() -> np.ndarray:
    return np.zeros((0, 3), dtype=float)


@dataclass
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points.copy()
        return _apply_similarity_transform(points, self.scale, self.rotation, self.translation)

    def apply_point_cloud(self, cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        return _transform_point_cloud_similarity(cloud, self.scale, self.rotation, self.translation)

    def inverse_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points.copy()
        if abs(self.scale) < 1e-12:
            logging.warning("Alignment scale too small; skipping inverse scaling.")
            inv_scale = 1.0
        else:
            inv_scale = 1.0 / self.scale
        centered = (points - self.translation) * inv_scale
        return centered @ self.rotation


@dataclass
class TrajectoryBundle:
    estimated_world: np.ndarray
    estimated_aligned: np.ndarray
    ground_truth: np.ndarray
    frame_ids: Sequence[int]
    alignment: Optional[SimilarityTransform]

    def get(self, key: str) -> np.ndarray:
        lookup = {
            "estimated_world": self.estimated_world,
            "estimated_aligned": self.estimated_aligned,
            "ground_truth": self.ground_truth,
        }
        arr = lookup.get(key)
        if arr is None:
            return _empty_traj()
        return arr

    def has(self, key: str) -> bool:
        return self.get(key).size != 0


def _compute_extent(points: np.ndarray) -> float:
    if points.size == 0:
        return float("nan")
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    return float(np.linalg.norm(maxs - mins))


def _extent_diff(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("inf")
    return abs(a - b)


def select_coordinate_frame(
    point_cloud: o3d.geometry.PointCloud,
    bundle: TrajectoryBundle,
    preferred_key: Optional[str],
) -> Tuple[str, bool]:
    scene_points = np.asarray(point_cloud.points) if len(point_cloud.points) else np.empty((0, 3))
    scene_extent = _compute_extent(scene_points) if scene_points.size else float("nan")
    world_extent = _compute_extent(bundle.estimated_world)
    aligned_extent = _compute_extent(bundle.estimated_aligned)
    diff_world = _extent_diff(scene_extent, world_extent)
    diff_aligned = _extent_diff(scene_extent, aligned_extent)

    key_alias = {
        "ground_truth": "estimated_aligned",
        "groundtruth": "estimated_aligned",
    }
    if preferred_key:
        normalized = key_alias.get(preferred_key.lower(), preferred_key.lower())
        if normalized not in {"estimated_world", "estimated_aligned"}:
            raise ValueError(
                "Preferred trajectory key must be one of: estimated_world, estimated_aligned, ground_truth."
            )
        if not bundle.has(normalized):
            raise ValueError(f"Preferred trajectory key '{normalized}' not available in the trajectory file.")
        apply_alignment = (
            normalized == "estimated_aligned"
            and bundle.alignment is not None
            and diff_world < diff_aligned
        )
        return normalized, apply_alignment

    candidates: List[Tuple[str, float]] = []
    if bundle.has("estimated_aligned"):
        candidates.append(("estimated_aligned", diff_aligned))
    if bundle.has("estimated_world"):
        candidates.append(("estimated_world", diff_world))
    if not candidates:
        if bundle.has("ground_truth"):
            return "estimated_aligned", bundle.alignment is not None
        raise ValueError("No trajectory key found. Expected estimated_world or estimated_aligned.")

    # Prefer the candidate whose extent best matches the scene; tie-breaker favors aligned frame.
    candidates = sorted(
        candidates,
        key=lambda item: (item[1], 0 if item[0] == "estimated_aligned" else 1),
    )
    base_key = candidates[0][0]
    apply_alignment = base_key == "estimated_aligned" and bundle.alignment is not None and diff_world < diff_aligned
    return base_key, apply_alignment


def prepare_scene_and_trajectories(
    point_cloud: o3d.geometry.PointCloud,
    bundle: TrajectoryBundle,
    base_key: str,
    apply_alignment: bool,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray]:
    alignment = bundle.alignment
    scene = point_cloud
    estimated = bundle.get(base_key) if base_key != "estimated_world" else bundle.estimated_world
    gt = bundle.ground_truth if bundle.ground_truth.size else _empty_traj()

    if base_key == "estimated_world":
        if estimated.size == 0 and bundle.estimated_aligned.size and alignment is not None:
            logging.info("Reconstructing estimated_world poses from aligned trajectory via inverse alignment.")
            estimated = alignment.inverse_points(bundle.estimated_aligned)
        if gt.size and alignment is not None:
            gt = alignment.inverse_points(gt)
        elif gt.size and alignment is None:
            logging.warning(
                "Ground truth trajectory available but no alignment data; plotting ground truth without scale adjustment."
            )
        return scene, estimated, gt

    if base_key == "estimated_aligned":
        if estimated.size == 0 and bundle.estimated_world.size and alignment is not None:
            logging.info("Synthesizing aligned trajectory from estimated_world using saved alignment.")
            estimated = alignment.apply_points(bundle.estimated_world)
        elif estimated.size == 0:
            estimated = bundle.estimated_world
        if apply_alignment and alignment is not None:
            logging.info("Applying saved alignment to point cloud for visualization.")
            scene = alignment.apply_point_cloud(scene)
        return scene, estimated, gt

    raise ValueError(f"Unsupported base coordinate frame '{base_key}'.")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a point cloud with its camera trajectory in 3D and BEV."
    )
    parser.add_argument(
        "point_cloud",
        type=Path,
        help="Input scene point cloud (.ply).",
    )
    parser.add_argument(
        "trajectory",
        type=Path,
        help="Camera trajectory JSON produced by reconstruct.py.",
    )
    parser.add_argument(
        "--trajectory-key",
        type=str,
        default=None,
        help="Coordinate frame for visualization: estimated_world, estimated_aligned, or ground_truth.",
    )
    parser.add_argument(
        "--ceiling-percentile",
        type=float,
        default=63.0,
        help="Percentile of points to retain along Y (removes the highest Y values interpreted as ceiling).",
    )
    parser.add_argument(
        "--bev-output",
        type=Path,
        default=None,
        help="Output path for the BEV PNG (defaults to <point_cloud_stem>_bev.png beside the point cloud).",
    )
    parser.add_argument(
        "--skip-3d",
        action="store_true",
        help="Do not open the interactive 3D window.",
    )
    parser.add_argument(
        "--skip-bev-view",
        action="store_true",
        help="Do not open the interactive BEV window (still saves the PNG).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging verbosity (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def load_point_cloud(path: Path, ceiling_percentile: float) -> o3d.geometry.PointCloud:
    if not path.exists():
        raise FileNotFoundError(f"Point cloud '{path}' does not exist.")
    logging.info("Loading point cloud from %s", path)
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        logging.warning("Point cloud has no points.")
        return pcd
    ceiling_percentile = float(np.clip(ceiling_percentile, 0.0, 100.0))
    filtered = _remove_ceiling_points(pcd, percentile=ceiling_percentile)
    logging.info(
        "Retained %d of %d points after ceiling removal (percentile %.1f).",
        len(filtered.points),
        len(pcd.points),
        ceiling_percentile,
    )
    return filtered


def _parse_trajectory_array(payload: dict, key: str) -> np.ndarray:
    if key not in payload:
        return _empty_traj()
    arr = np.asarray(payload[key], dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Trajectory key '{key}' does not contain Nx3 positions.")
    return arr


def load_camera_trajectory(path: Path) -> TrajectoryBundle:
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file '{path}' does not exist.")
    logging.info("Loading camera trajectory from %s", path)
    payload = json.loads(path.read_text())
    frame_ids = payload.get("frame_ids", [])
    estimated_world = _parse_trajectory_array(payload, "estimated_world")
    estimated_aligned = _parse_trajectory_array(payload, "estimated_aligned")
    ground_truth = _parse_trajectory_array(payload, "ground_truth")

    alignment_data = payload.get("alignment")
    alignment: Optional[SimilarityTransform] = None
    if isinstance(alignment_data, dict):
        try:
            scale = float(alignment_data.get("scale", 1.0))
            rotation = np.asarray(alignment_data.get("rotation", np.eye(3)), dtype=float)
            translation = np.asarray(alignment_data.get("translation", np.zeros(3)), dtype=float)
            rotation = rotation.reshape(3, 3)
            translation = translation.reshape(3)
            alignment = SimilarityTransform(scale=scale, rotation=rotation, translation=translation)
        except (TypeError, ValueError) as exc:
            logging.warning("Invalid alignment data in %s: %s", path, exc)
            alignment = None

    logging.info(
        "Loaded trajectories: estimated_world=%d, estimated_aligned=%d, ground_truth=%d poses.",
        estimated_world.shape[0],
        estimated_aligned.shape[0],
        ground_truth.shape[0],
    )
    return TrajectoryBundle(
        estimated_world=estimated_world,
        estimated_aligned=estimated_aligned,
        ground_truth=ground_truth,
        frame_ids=frame_ids,
        alignment=alignment,
    )


def create_endpoint_cloud(
    trajectory: np.ndarray, start_color: Tuple[float, float, float], end_color: Tuple[float, float, float]
) -> Optional[o3d.geometry.PointCloud]:
    if trajectory.size == 0:
        return None
    endpoint_cloud = o3d.geometry.PointCloud()
    endpoint_cloud.points = o3d.utility.Vector3dVector(
        np.vstack([trajectory[0], trajectory[-1]])
    )
    endpoint_cloud.colors = o3d.utility.Vector3dVector(
        np.array([start_color, end_color], dtype=float)
    )
    return endpoint_cloud


def visualize_3d(
    pcd: o3d.geometry.PointCloud,
    estimated_traj: np.ndarray,
    gt_traj: np.ndarray,
) -> None:
    if len(pcd.points) == 0 and estimated_traj.size == 0 and gt_traj.size == 0:
        logging.warning("Nothing to visualize in 3D.")
        return
    geometries = []
    if len(pcd.points):
        geometries.append(pcd)
    est_lines = _create_trajectory_line_set(estimated_traj, (1.0, 0.0, 0.0))
    if est_lines is not None:
        geometries.append(est_lines)
        est_endpoints = create_endpoint_cloud(
            estimated_traj, (0.0, 0.7, 0.0), (1.0, 0.0, 0.0)
        )
        if est_endpoints is not None:
            geometries.append(est_endpoints)
    gt_lines = _create_trajectory_line_set(gt_traj, (0.0, 0.0, 0.0))
    if gt_lines is not None:
        geometries.append(gt_lines)
        gt_endpoints = create_endpoint_cloud(
            gt_traj, (0.2, 0.2, 0.2), (0.0, 0.0, 0.0)
        )
        if gt_endpoints is not None:
            geometries.append(gt_endpoints)
    logging.info("Opening 3D visualization window.")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Scene with Camera Trajectory",
        width=960,
        height=720,
    )


def save_bev_image(
    pcd: o3d.geometry.PointCloud,
    estimated_traj: np.ndarray,
    gt_traj: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Saving BEV projection to %s", output_path)
    _save_bev_plot(pcd, estimated_traj, gt_traj, output_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    pcd = load_point_cloud(args.point_cloud, args.ceiling_percentile)
    bundle = load_camera_trajectory(args.trajectory)
    frame_ids = bundle.frame_ids

    if frame_ids:
        logging.info("Frame ID range: %s -> %s", frame_ids[0], frame_ids[-1])

    base_key, apply_alignment = select_coordinate_frame(pcd, bundle, args.trajectory_key)
    logging.info(
        "Using '%s' coordinate frame for visualization (alignment %s).",
        base_key,
        "applied" if apply_alignment else "not applied",
    )
    scene_for_vis, estimated_traj, gt_traj = prepare_scene_and_trajectories(
        pcd, bundle, base_key, apply_alignment
    )

    if not args.skip_3d:
        visualize_3d(scene_for_vis, estimated_traj, gt_traj)

    bev_path = (
        args.bev_output
        if args.bev_output is not None
        else args.point_cloud.with_name(f"{args.point_cloud.stem}_bev.png")
    )
    save_bev_image(scene_for_vis, estimated_traj, gt_traj, bev_path)

    if not args.skip_bev_view:
        logging.info("Opening BEV visualization window.")
        _visualize_bev_scene(scene_for_vis, estimated_traj, gt_traj)


if __name__ == "__main__":
    main()
