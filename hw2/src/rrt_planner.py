#!/usr/bin/env python3
"""Run an RRT planner on the 2D semantic map for Physical AI HW2."""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
from matplotlib import image as mpl_image
import numpy as np
from scipy import ndimage


LOGGER = logging.getLogger(Path(__file__).stem)

TARGET_CLASS = "chair"
MAP_IMAGE_PATH: Path = Path("results/map.png")
OBSTACLE_MASK_PATH: Path = Path("results/obstacle_mask.png")
META_PATH: Path = Path("results/map_meta.json")
CLASSES_CSV_PATH: Path = Path("semantic_segmentation_classes.csv")
OUTPUT_DIR: Path = Path("results")

CLICK_TO_SELECT_START = True
START_PIXEL: Optional[Tuple[int, int]] = None  # Used when CLICK_TO_SELECT_START is False.

# --------- RRT parameters ----------
RRT_STEP_SIZE = 15.0
GOAL_BIAS = 0.05
GOAL_RADIUS = 20.0
MAX_ITERATIONS = 20000
RNG_SEED: Optional[int] = None
# -----------------------------------

LOG_LEVEL = "INFO"


@dataclass
class Node:
    """Simple container for a tree node in pixel coordinates."""

    x: int
    y: int
    parent: Optional[int]


def configure_logging(level: str) -> None:
    """Configure logging output."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )


def get_pyplot(interactive: bool):
    """Import matplotlib.pyplot with an interactive or Agg backend."""
    if interactive:
        import matplotlib.pyplot as plt
    else:
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    return plt


def load_png(path: Path) -> np.ndarray:
    """Load a PNG image and return a uint8 array without alpha channel."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = mpl_image.imread(path)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype.kind == "f":
        image = np.round(image * 255.0).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    return image


def load_obstacle_mask(path: Path) -> np.ndarray:
    """Load and normalise the obstacle mask to binary uint8 values."""
    image = load_png(path)
    if image.ndim == 3:
        image = image[:, :, 0]
    mask = (image > 0).astype(np.uint8)
    return mask * 255


def load_class_colours(classes_path: Path) -> Dict[str, List[int]]:
    """Parse the CSV file into a name -> RGB mapping."""
    if not classes_path.exists():
        raise FileNotFoundError(f"Classes CSV file not found: {classes_path}")

    with classes_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        class_map: Dict[str, List[int]] = {}
        for row in reader:
            if not row:
                continue
            name = row.get("Name")
            colour_raw = row.get("Color_Code (R,G,B)")
            if not name or not colour_raw:
                continue
            colour = _parse_rgb_string(colour_raw)
            if colour is None:
                continue
            class_map[name] = colour
    if not class_map:
        raise ValueError(
            f"Failed to parse any class colours from {classes_path}. "
            "Ensure the CSV layout matches the expected template."
        )
    return class_map


def _parse_rgb_string(raw: str) -> List[int] | None:
    """Parse a '(R, G, B)' string into integer components."""
    cleaned = raw.strip().strip("()")
    parts = [part.strip() for part in cleaned.split(",")]
    if len(parts) != 3:
        return None
    try:
        values = [int(float(part)) for part in parts]
    except ValueError:
        return None
    return [int(np.clip(v, 0, 255)) for v in values]


def pick_target_anchor(
    map_image: np.ndarray,
    obstacle_mask: np.ndarray,
    target_colour: Sequence[int],
) -> Tuple[int, int]:
    """Select a free pixel directly in front of the target class blob."""
    target_rgb = np.array(target_colour, dtype=np.uint8)
    target_mask = np.all(map_image[:, :, :3] == target_rgb, axis=2)
    if not np.any(target_mask):
        raise ValueError("Target class pixels not found on the map.")

    free_mask = obstacle_mask == 0
    border_mask = ndimage.binary_dilation(target_mask, iterations=1) & ~target_mask
    candidate_mask = border_mask & free_mask
    if not np.any(candidate_mask):
        # Expand search slightly if direct neighbours are blocked.
        expanded_border = ndimage.binary_dilation(target_mask, iterations=2) & ~target_mask
        candidate_mask = expanded_border & free_mask
    if not np.any(candidate_mask):
        raise ValueError("Failed to locate a navigable pixel in front of the target blob.")

    distance_to_obstacles = ndimage.distance_transform_edt(free_mask)
    candidate_indices = np.column_stack(np.nonzero(candidate_mask))
    scores = distance_to_obstacles[candidate_mask]
    best_idx = int(np.argmax(scores))
    y, x = candidate_indices[best_idx]
    LOGGER.info(
        "Selected target anchor at pixel (%d, %d) with obstacle clearance %.2f px",
        x,
        y,
        scores[best_idx],
    )
    return int(x), int(y)


def pick_start_by_click(plt, map_image: np.ndarray) -> Tuple[int, int]:
    """Allow the user to click on the map to choose a start pixel."""
    coords: List[Tuple[int, int]] = []

    def onclick(event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        coords.append((int(round(event.xdata)), int(round(event.ydata))))
        plt.close(event.canvas.figure)

    fig, ax = plt.subplots()
    ax.imshow(map_image)
    ax.set_title("Click to choose the start position.")
    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    LOGGER.info("Awaiting user click for start position...")
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    plt.close(fig)
    if not coords:
        raise RuntimeError("No start position selected.")
    LOGGER.info("Selected start pixel (%d, %d)", coords[0][0], coords[0][1])
    return coords[0]


def in_bounds(point: Tuple[int, int], width: int, height: int) -> bool:
    """Check if a pixel is inside the map bounds."""
    x, y = point
    return 0 <= x < width and 0 <= y < height


def bresenham_line(p0: Tuple[int, int], p1: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Generate integer points along a line between two pixels."""
    x0, y0 = p0
    x1, y1 = p1
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    points: List[Tuple[int, int]] = []
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points


def segment_is_free(
    start: Tuple[int, int],
    end: Tuple[int, int],
    free_mask: np.ndarray,
) -> bool:
    """Return True if every pixel along the segment lies in free space."""
    for x, y in bresenham_line(start, end):
        if not in_bounds((x, y), free_mask.shape[1], free_mask.shape[0]):
            return False
        if not free_mask[y, x]:
            return False
    return True


def run_rrt(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    free_mask: np.ndarray,
    step_size: float,
    goal_bias: float,
    goal_radius: float,
    max_iter: int,
    rng: np.random.Generator,
) -> Tuple[List[Node], List[Tuple[Tuple[int, int], Tuple[int, int]]], int]:
    """Run a basic RRT and return the nodes, edges, and goal node index."""
    height, width = free_mask.shape
    if not free_mask[start[1], start[0]]:
        raise ValueError(f"Start pixel {start} is not in free space.")
    if not free_mask[goal[1], goal[0]]:
        raise ValueError(f"Goal pixel {goal} is not in free space.")

    free_coords = np.column_stack(np.nonzero(free_mask)).astype(np.int32)
    if free_coords.size == 0:
        raise ValueError("Free space is empty; cannot run RRT.")

    nodes: List[Node] = [Node(x=start[0], y=start[1], parent=None)]
    coords = np.array([[start[0], start[1]]], dtype=np.float32)
    edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

    goal_vec = np.array([goal[0], goal[1]], dtype=np.float32)

    for iteration in range(max_iter):
        if rng.random() < goal_bias:
            sample = goal_vec
        else:
            idx = rng.integers(0, free_coords.shape[0])
            # free_coords stores (y, x); flip to (x, y)
            y, x = free_coords[idx]
            sample = np.array([x, y], dtype=np.float32)

        dists = np.linalg.norm(coords - sample, axis=1)
        nearest_idx = int(np.argmin(dists))
        nearest_point = coords[nearest_idx]
        direction = sample - nearest_point
        distance = np.linalg.norm(direction)
        if distance == 0.0:
            continue
        direction /= distance
        step = direction * min(step_size, distance)
        new_point = nearest_point + step
        new_x = int(round(new_point[0]))
        new_y = int(round(new_point[1]))
        if not in_bounds((new_x, new_y), width, height):
            continue
        if not free_mask[new_y, new_x]:
            continue
        parent_pixel = (int(round(nearest_point[0])), int(round(nearest_point[1])))
        if not segment_is_free(parent_pixel, (new_x, new_y), free_mask):
            continue

        nodes.append(Node(x=new_x, y=new_y, parent=nearest_idx))
        coords = np.vstack([coords, [new_x, new_y]])
        edges.append((parent_pixel, (new_x, new_y)))

        dist_to_goal = math.hypot(new_x - goal[0], new_y - goal[1])
        if dist_to_goal <= goal_radius:
            if segment_is_free((new_x, new_y), goal, free_mask):
                goal_parent = len(nodes) - 1
                nodes.append(Node(x=goal[0], y=goal[1], parent=goal_parent))
                coords = np.vstack([coords, [goal[0], goal[1]]])
                edges.append(((new_x, new_y), goal))
                LOGGER.info("Reached goal after %d iterations.", iteration + 1)
                return nodes, edges, len(nodes) - 1
    raise RuntimeError("RRT failed to find a path within the iteration budget.")


def extract_path(nodes: List[Node], goal_index: int) -> List[Tuple[int, int]]:
    """Recover the path from start to goal by following parent links."""
    path: List[Tuple[int, int]] = []
    idx: Optional[int] = goal_index
    while idx is not None:
        node = nodes[idx]
        path.append((node.x, node.y))
        idx = node.parent
    path.reverse()
    return path


def pixels_to_habitat(path: np.ndarray, affine: Dict[str, float]) -> np.ndarray:
    """Convert path pixels to Habitat (x, z) coordinates using the affine inverse."""
    sx_inv = float(affine["sx_inv"])
    sz_inv = float(affine["sz_inv"])
    tx_inv = float(affine["tx_inv"])
    tz_inv = float(affine["tz_inv"])
    x_hab = sx_inv * path[:, 0] + tx_inv
    z_hab = sz_inv * path[:, 1] + tz_inv
    return np.stack([x_hab, z_hab], axis=1)


def save_path_arrays(
    output_dir: Path,
    target: str,
    path_pixels: np.ndarray,
    path_habitat: np.ndarray,
) -> Tuple[Path, Path]:
    """Persist path arrays to disk."""
    pixels_path = output_dir / f"path_{target}_pixels.npy"
    habitat_path = output_dir / f"path_{target}_habitat.npy"
    np.save(pixels_path, path_pixels)
    np.save(habitat_path, path_habitat)
    LOGGER.info("Saved pixel path to %s", pixels_path)
    LOGGER.info("Saved Habitat path to %s", habitat_path)
    return pixels_path, habitat_path


def save_overlay(
    plt,
    map_image: np.ndarray,
    edges: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]],
    path: Sequence[Tuple[int, int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    output_path: Path,
) -> None:
    """Render and save an overlay of the RRT exploration and final path."""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    ax.imshow(map_image)
    for (x0, y0), (x1, y1) in edges:
        ax.plot([x0, x1], [y0, y1], color="cyan", linewidth=0.5, alpha=0.4)
    if path:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color="yellow", linewidth=2.5, label="Path")
    ax.scatter([start[0]], [start[1]], c="lime", s=40, marker="o", label="Start")
    ax.scatter([goal[0]], [goal[1]], c="red", s=120, marker="*", label="Target anchor")
    ax.set_xlim(0, map_image.shape[1])
    ax.set_ylim(map_image.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right")
    fig.tight_layout(pad=0)
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info("Saved RRT overlay to %s", output_path)


def main() -> None:
    configure_logging(LOG_LEVEL)

    if CLICK_TO_SELECT_START and START_PIXEL is not None:
        raise ValueError("Set either CLICK_TO_SELECT_START or START_PIXEL, not both.")
    if not CLICK_TO_SELECT_START and START_PIXEL is None:
        raise ValueError("Provide START_PIXEL when CLICK_TO_SELECT_START is False.")

    rng = np.random.default_rng(RNG_SEED)
    output_dir: Path = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    map_image = load_png(MAP_IMAGE_PATH)
    obstacle_mask = load_obstacle_mask(OBSTACLE_MASK_PATH)
    if obstacle_mask.shape != map_image.shape[:2]:
        raise ValueError("Obstacle mask size does not match map image size.")

    class_colours = load_class_colours(CLASSES_CSV_PATH)
    if TARGET_CLASS not in class_colours:
        raise ValueError(f"Target class {TARGET_CLASS!r} not found in colour table.")

    target_colour = class_colours[TARGET_CLASS]
    target_anchor = pick_target_anchor(map_image, obstacle_mask, target_colour)

    plt = get_pyplot(CLICK_TO_SELECT_START)
    if CLICK_TO_SELECT_START:
        start_pixel = pick_start_by_click(plt, map_image)
    else:
        start_pixel = START_PIXEL
        if start_pixel is None:
            raise ValueError("START_PIXEL must be defined when CLICK_TO_SELECT_START is False.")
        start_pixel = (int(start_pixel[0]), int(start_pixel[1]))

    width = map_image.shape[1]
    height = map_image.shape[0]
    if not in_bounds(start_pixel, width, height):
        raise ValueError(f"Start pixel {start_pixel} lies outside the map bounds.")
    if not in_bounds(target_anchor, width, height):
        raise ValueError(f"Target anchor {target_anchor} lies outside the map bounds.")

    start_record = output_dir / "last_start.txt"
    start_record.write_text(f"{start_pixel[0]},{start_pixel[1]}\n")

    free_mask = (obstacle_mask == 0).astype(bool)
    nodes, edges, goal_idx = run_rrt(
        start=start_pixel,
        goal=target_anchor,
        free_mask=free_mask,
        step_size=float(RRT_STEP_SIZE),
        goal_bias=float(GOAL_BIAS),
        goal_radius=float(GOAL_RADIUS),
        max_iter=int(MAX_ITERATIONS),
        rng=rng,
    )
    path_pixels_list = extract_path(nodes, goal_idx)
    path_pixels = np.array(path_pixels_list, dtype=np.int32)

    meta = json.loads(META_PATH.read_text())
    affine = meta.get("affine")
    if affine is None:
        raise ValueError("Affine transform missing from map_meta.json.")
    path_habitat = pixels_to_habitat(path_pixels.astype(np.float64), affine)

    save_path_arrays(output_dir, TARGET_CLASS, path_pixels, path_habitat)

    overlay_path = output_dir / f"rrt_{TARGET_CLASS}_overlay.png"
    save_overlay(
        plt=plt,
        map_image=map_image,
        edges=edges,
        path=path_pixels_list,
        start=start_pixel,
        goal=target_anchor,
        output_path=overlay_path,
    )


if __name__ == "__main__":
    main()
