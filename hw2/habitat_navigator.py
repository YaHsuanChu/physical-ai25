"""
Execute a pre-computed Habitat-space path with discrete actions and record video.

The script reads waypoints produced in Part 2, instantiates the Habitat simulator
with controllable step magnitudes, follows the path using move/turn actions,
overlays a semi-transparent mask for the requested target class, and saves both
an MP4 recording and the visited trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import habitat_sim

from habitat_sim.utils.common import quat_from_two_vectors, quat_rotate_vector

from load import make_env, transform_rgb_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Navigate along a Habitat-space path using discrete actions."
    )
    parser.add_argument("--target", required=True, help="Semantic target category.")
    parser.add_argument(
        "--path",
        required=True,
        type=Path,
        help="Path to the .npy file with (x_hab, z_hab) waypoints.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MP4 filepath. Defaults to results/<target>.mp4.",
    )
    parser.add_argument(
        "--traj_json",
        type=Path,
        default=None,
        help="Output JSON filepath. Defaults to results/nav_<target>_traj.json.",
    )
    parser.add_argument("--fps", type=int, default=60, help="Video frames per second.")
    parser.add_argument(
        "--pos_tolerance",
        type=float,
        default=0.05,
        help="Distance (m) to consider a waypoint reached.",
    )
    parser.add_argument(
        "--yaw_step_deg",
        type=float,
        default=5.0,
        help="Discrete yaw increment in degrees.",
    )
    parser.add_argument(
        "--forward_step_m",
        type=float,
        default=0.1,
        help="Forward step length in meters.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Optional override for the Habitat scene path.",
    )
    parser.add_argument(
        "--mask_png",
        type=Path,
        default=None,
        help="Optional PNG to blend with each frame (fallback visualization).",
    )
    parser.add_argument(
        "--mask_alpha",
        type=float,
        default=0.35,
        help="Alpha blend used for target mask overlays.",
    )
    parser.add_argument(
        "--mask_color_csv",
        type=Path,
        default=Path("semantic_segmentation_classes.csv"),
        help="CSV file mapping semantic classes to RGB colors.",
    )
    return parser.parse_args()


def parse_rgb_tuple(raw: str) -> Optional[Tuple[int, int, int]]:
    try:
        cleaned = raw.strip().strip("()")
        parts = [int(p.strip()) for p in cleaned.split(",")[:3]]
        if len(parts) != 3:
            return None
        return tuple(parts)
    except (ValueError, AttributeError):
        return None


def parse_hex_color(raw: str) -> Optional[Tuple[int, int, int]]:
    try:
        cleaned = raw.strip().lstrip("#")
        if len(cleaned) != 6:
            return None
        r = int(cleaned[0:2], 16)
        g = int(cleaned[2:4], 16)
        b = int(cleaned[4:6], 16)
        return (r, g, b)
    except (ValueError, AttributeError):
        return None


def load_target_color(
    target: str,
    csv_path: Path,
) -> Tuple[int, int, int]:
    """Return (R, G, B) color for the target class."""
    target_lower = target.lower()

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file with semantic colors not found at {csv_path}."
        )

    import csv

    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        name_columns = [
            col
            for col in fieldnames
            if col
            and "name" in col.lower()
            and not col.lower().startswith("unnamed")
        ]
        if not name_columns:
            name_columns = [
                col
                for col in fieldnames
                if col
                and any(
                    key in col.lower() for key in ("name", "class", "label", "category")
                )
            ]
        color_columns = [
            col
            for col in fieldnames
            if col and any(key in col.lower() for key in ("color", "rgb"))
        ]
        hex_columns = [
            col for col in fieldnames if col and "hex" in col.lower()
        ]

        for row in reader:
            name_val = next(
                (
                    str(row[col]).strip()
                    for col in name_columns
                    if row.get(col)
                ),
                None,
            )
            if not name_val or name_val.lower() != target_lower:
                continue

            rgb: Optional[Tuple[int, int, int]] = None
            color_candidates = [
                str(row[col]).strip()
                for col in color_columns
                if row.get(col)
            ]
            for raw_val in color_candidates:
                rgb = parse_rgb_tuple(raw_val)
                if rgb is not None:
                    return rgb
            for col in hex_columns:
                raw_val = row.get(col)
                if not raw_val:
                    continue
                rgb = parse_hex_color(str(raw_val))
                if rgb is not None:
                    return rgb
            for value in row.values():
                if not value:
                    continue
                rgb = parse_rgb_tuple(str(value))
                if rgb is not None:
                    return rgb

    raise ValueError(
        f"Unable to find RGB color for target '{target}' in {csv_path}."
    )


def load_waypoints(path_file: Path) -> np.ndarray:
    waypoints = np.load(path_file)
    if waypoints.ndim != 2 or waypoints.shape[1] != 2:
        raise ValueError(
            f"Expected waypoints of shape (N, 2); received {waypoints.shape}."
        )
    return waypoints.astype(np.float32)


def collect_target_semantic_ids(sim, target: str) -> Sequence[int]:
    scene = sim.semantic_scene
    if scene is None:
        return []

    target_lower = target.lower()
    instance_ids = []
    for obj in scene.objects:
        if obj is None or obj.semantic_id is None:
            continue
        category = obj.category
        if category is None:
            continue
        name = category.name()
        if name and name.lower() == target_lower:
            instance_ids.append(obj.semantic_id)
    return instance_ids


def compute_heading_deg(rotation) -> float:
    """Return heading in degrees with 0 pointing along -Z."""
    forward_world = quat_rotate_vector(rotation, np.array([0.0, 0.0, -1.0]))
    forward_world[1] = 0.0
    norm = np.linalg.norm(forward_world)
    if norm < 1e-6:
        return 0.0
    forward_world /= norm
    heading = math.degrees(math.atan2(forward_world[0], -forward_world[2]))
    return float((heading + 180.0) % 360.0 - 180.0)


def blend_mask(
    frame_bgr: np.ndarray,
    semantic_obs: Optional[np.ndarray],
    target_ids: Sequence[int],
    mask_color_bgr: Tuple[int, int, int],
    alpha: float,
    fallback_overlay: Optional[np.ndarray],
) -> Tuple[np.ndarray, bool]:
    blended = frame_bgr.copy()
    mask_applied = False

    if semantic_obs is not None and len(target_ids) > 0:
        mask = np.isin(semantic_obs, target_ids)
        if mask.any():
            color_arr = np.array(mask_color_bgr, dtype=np.float32)
            current_vals = blended[mask].astype(np.float32)
            blended[mask] = ((1.0 - alpha) * current_vals + alpha * color_arr).astype(
                np.uint8
            )
            mask_applied = True

    if fallback_overlay is not None:
        blended = cv2.addWeighted(blended, 1.0 - alpha, fallback_overlay, alpha, 0.0)
        mask_applied = True

    return blended, mask_applied


def prepare_mask_overlay(
    mask_path: Optional[Path], frame_size: Tuple[int, int]
) -> Optional[np.ndarray]:
    if mask_path is None:
        return None
    overlay = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if overlay is None:
        raise FileNotFoundError(f"Failed to read mask PNG at {mask_path}.")
    if overlay.shape[:2] != frame_size:
        overlay = cv2.resize(overlay, (frame_size[1], frame_size[0]))
    if overlay.shape[2] == 4:
        # Drop alpha; blending handles transparency.
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGRA2BGR)
    return overlay


def ensure_output_path(path: Optional[Path], default: Path) -> Path:
    result = path if path is not None else default
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def main() -> None:
    args = parse_args()

    waypoints = load_waypoints(args.path)
    if len(waypoints) == 0:
        raise ValueError("The provided waypoint file is empty.")

    output_path = ensure_output_path(
        args.output, Path("results") / f"{args.target}.mp4"
    )
    traj_path = ensure_output_path(
        args.traj_json, Path("results") / f"nav_{args.target}_traj.json"
    )

    rgb_color = load_target_color(args.target, args.mask_color_csv)
    mask_color_bgr = tuple(int(c) for c in rgb_color[::-1])

    override_settings: Dict[str, object] = {}
    if args.scene:
        override_settings["scene"] = args.scene

    poses = []
    frames: List[np.ndarray] = []
    mask_used = False
    sim = None
    video_writer = None
    display_window = "Habitat Navigation"
    display_enabled = False

    try:
        sim, agent, _ = make_env(
            override_settings or None,
            forward_step_m=args.forward_step_m,
            turn_step_deg=args.yaw_step_deg,
        )

        initial_state = agent.get_state()
        start_y = float(initial_state.position[1])
        start_state = habitat_sim.AgentState()
        start_state.position = np.array(
            [waypoints[0, 0], start_y, waypoints[0, 1]], dtype=np.float32
        )
        if len(waypoints) > 1:
            next_dir = np.array(
                [
                    waypoints[1, 0] - waypoints[0, 0],
                    0.0,
                    waypoints[1, 1] - waypoints[0, 1],
                ],
                dtype=np.float32,
            )
            norm = np.linalg.norm(next_dir)
            if norm > 1e-6:
                next_dir /= norm
                start_state.rotation = quat_from_two_vectors(
                    np.array([0.0, 0.0, -1.0], dtype=np.float32), next_dir
                )
        agent.set_state(start_state)

        observations = sim.get_sensor_observations()
        frame_bgr = transform_rgb_bgr(observations["color_sensor"])
        frame_h, frame_w = frame_bgr.shape[:2]
        fallback_overlay = prepare_mask_overlay(args.mask_png, (frame_h, frame_w))

        target_ids = collect_target_semantic_ids(sim, args.target)

        try:
            cv2.namedWindow(display_window, cv2.WINDOW_NORMAL)
            display_enabled = True
        except cv2.error:
            print("Warning: Failed to create display window; continuing without GUI.")

        def log_pose():
            state = agent.get_state()
            poses.append(
                {
                    "x": float(state.position[0]),
                    "y": float(state.position[1]),
                    "z": float(state.position[2]),
                    "heading": compute_heading_deg(state.rotation),
                }
            )

        blend_frame, applied = blend_mask(
            frame_bgr,
            observations.get("semantic_sensor"),
            target_ids,
            mask_color_bgr,
            args.mask_alpha,
            fallback_overlay,
        )
        mask_used |= applied
        frames.append(blend_frame)
        if display_enabled:
            cv2.imshow(display_window, blend_frame)
            cv2.waitKey(1)
        log_pose()

        forward_axis = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        for waypoint in waypoints[1:]:
            waypoint_vec = np.array(
                [waypoint[0], start_y, waypoint[1]], dtype=np.float32
            )
            while True:
                state = agent.get_state()
                current_pos = state.position
                delta = waypoint_vec - current_pos
                planar_delta = np.array([delta[0], 0.0, delta[2]], dtype=np.float32)
                distance = float(np.linalg.norm(planar_delta[[0, 2]]))
                if distance <= args.pos_tolerance:
                    break

                desired_dir = planar_delta.copy()
                norm = np.linalg.norm(desired_dir[[0, 2]])
                if norm < 1e-6:
                    break
                desired_dir /= norm

                forward_world = quat_rotate_vector(state.rotation, forward_axis)
                forward_world[1] = 0.0
                forward_norm = np.linalg.norm(forward_world[[0, 2]])
                if forward_norm < 1e-6:
                    forward_world = forward_axis.copy()
                else:
                    forward_world /= forward_norm

                cross = np.cross(forward_world, desired_dir)
                dot = float(np.clip(np.dot(forward_world, desired_dir), -1.0, 1.0))
                turn_angle = math.degrees(math.atan2(np.linalg.norm(cross), dot))

                if turn_angle > args.yaw_step_deg / 2.0:
                    action = "turn_left" if cross[1] >= 0 else "turn_right"
                else:
                    action = "move_forward"

                observations = sim.step(action)
                frame_bgr = transform_rgb_bgr(observations["color_sensor"])
                blend_frame, applied = blend_mask(
                    frame_bgr,
                    observations.get("semantic_sensor"),
                    target_ids,
                    mask_color_bgr,
                    args.mask_alpha,
                    fallback_overlay,
                )
                mask_used |= applied
                frames.append(blend_frame)
                if display_enabled:
                    cv2.imshow(display_window, blend_frame)
                    cv2.waitKey(1)
                log_pose()
        total_frames = len(frames)
        if total_frames == 0:
            raise RuntimeError("No frames captured for rendering.")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(output_path), fourcc, args.fps, (frame_w, frame_h)
        )
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer at {output_path}.")

        for idx, frame in enumerate(frames, start=1):
            video_writer.write(frame)
            print(
                f"\rRendering frame ({idx}/{total_frames})",
                end="",
                flush=True,
            )

        print()
        frames.clear()

        if display_enabled:
            cv2.destroyWindow(display_window)

        if sim is not None:
            sim.close()
            sim = None
        agent = None
    finally:
        if video_writer is not None:
            video_writer.release()
        if sim is not None:
            sim.close()

    with traj_path.open("w", encoding="utf-8") as handle:
        traj_dict = {str(i): pose for i, pose in enumerate(poses)}
        json.dump(traj_dict, handle, indent=2)

    if not mask_used:
        print(
            "Warning: no target mask was applied. Provide --mask_png if semantic IDs "
            "for the target are unavailable."
        )


if __name__ == "__main__":
    main()
