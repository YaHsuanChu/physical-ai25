#!/usr/bin/env python3
"""Construct a 2D semantic map from a 3D point cloud for Physical AI HW2.

This script filters out floor and ceiling points, projects the remaining
geometry onto the x–z plane, rasterises a semantic map, and exports the data
products required by downstream homework parts.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402


LOGGER = logging.getLogger(Path(__file__).stem)

POINTS_PATH: Path = Path("semantic_3d_pointcloud/point.npy")
COLOR01_PATH: Path | None = Path("semantic_3d_pointcloud/color01.npy")
COLOR0255_PATH: Path = Path("semantic_3d_pointcloud/color0255.npy")
CLASSES_CSV_PATH: Path = Path("semantic_segmentation_classes.csv")
MAP_SIZE = 2048
FLOOR_FRAC = 0.05
CEIL_FRAC = 0.05
OUTPUT_DIR: Path = Path("results")
LOG_LEVEL = "INFO"


@dataclass(frozen=True)
class AffineTransform:
    """Container for affine forward/backward mapping between Habitat and pixels."""

    sx: float
    sz: float
    tx: float
    tz: float

    @property
    def inverse(self) -> Dict[str, float]:
        """Return the inverse transform components."""
        sx_inv = 1.0 / self.sx if self.sx != 0.0 else 0.0
        sz_inv = 1.0 / self.sz if self.sz != 0.0 else 0.0
        tx_inv = -self.tx / self.sx if self.sx != 0.0 else 0.0
        tz_inv = -self.tz / self.sz if self.sz != 0.0 else 0.0
        return {
            "sx_inv": sx_inv,
            "sz_inv": sz_inv,
            "tx_inv": tx_inv,
            "tz_inv": tz_inv,
        }

    def as_dict(self) -> Dict[str, float]:
        """Return the forward transform components."""
        return {"sx": self.sx, "sz": self.sz, "tx": self.tx, "tz": self.tz}


def configure_logging(level: str) -> None:
    """Configure logging according to the supplied textual level."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )


def load_numpy_points(points_path: Path) -> np.ndarray:
    """Load and validate the point cloud."""
    if not points_path.exists():
        raise FileNotFoundError(f"Point cloud not found: {points_path}")
    points = np.load(points_path)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Expected point cloud shape (N,3); got {points.shape} from {points_path}"
        )
    return points.astype(np.float64)


def load_numpy_colours(color01_path: Path | None, color0255_path: Path) -> np.ndarray:
    """Load RGB colours, preferring the float [0,1] array if it exists."""
    rgb: np.ndarray | None = None
    if color01_path is not None and color01_path.exists():
        rgb = np.load(color01_path)
        if rgb.max() <= 1.0:
            rgb = np.round(np.clip(rgb * 255.0, 0.0, 255.0)).astype(np.uint8)
        else:
            rgb = np.round(np.clip(rgb, 0.0, 255.0)).astype(np.uint8)
        LOGGER.info("Loaded colours from %s", color01_path)
    if rgb is None:
        if not color0255_path.exists():
            raise FileNotFoundError(f"Colour array not found: {color0255_path}")
        rgb = np.load(color0255_path)
        if not np.issubdtype(rgb.dtype, np.integer):
            rgb = np.round(np.clip(rgb, 0.0, 255.0))
        rgb = rgb.astype(np.uint8)
        LOGGER.info("Loaded colours from %s", color0255_path)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(
            f"Expected colour array shape (N,3); got {rgb.shape} from colour files"
        )
    return rgb


def load_class_colours(classes_path: Path) -> Dict[str, List[int]]:
    """Parse the CSV file into a mapping of class names to RGB colours."""
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
    """Parse a '(R, G, B)' string into a list of ints."""
    cleaned = raw.strip().strip("()")
    parts = [part.strip() for part in cleaned.split(",")]
    if len(parts) != 3:
        return None
    try:
        values = [int(float(part)) for part in parts]
    except ValueError:
        return None
    return [int(np.clip(v, 0, 255)) for v in values]


def filter_floor_ceiling(
    points: np.ndarray,
    colours: np.ndarray,
    floor_frac: float,
    ceil_frac: float,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """Remove floor and ceiling points via quantile filtering on Y."""
    if not 0.0 <= floor_frac < 0.5 or not 0.0 <= ceil_frac < 0.5:
        raise ValueError("floor_frac and ceil_frac must lie in [0, 0.5).")
    y_vals = points[:, 1]
    lower = np.quantile(y_vals, floor_frac)
    upper = np.quantile(y_vals, 1.0 - ceil_frac)
    mask = (y_vals >= lower) & (y_vals <= upper)
    LOGGER.info(
        "Removing floor/ceiling via %.2f / %.2f quantiles -> kept %d / %d points",
        floor_frac,
        ceil_frac,
        int(mask.sum()),
        points.shape[0],
    )
    filtered_points = points[mask]
    filtered_colours = colours[mask]
    return filtered_points, filtered_colours, (float(lower), float(upper))


def remove_named_classes(
    points: np.ndarray,
    colours: np.ndarray,
    class_lookup: Dict[str, List[int]],
    names: Iterable[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove points whose colours correspond to the supplied class names."""
    exclude_colours = [
        np.array(class_lookup[name], dtype=np.uint8)
        for name in names
        if name in class_lookup
    ]
    if not exclude_colours:
        return points, colours
    mask = np.ones(colours.shape[0], dtype=bool)
    for colour in exclude_colours:
        mask &= ~np.all(colours == colour, axis=1)
    removed = colours.shape[0] - int(mask.sum())
    if removed > 0:
        LOGGER.info(
            "Removed %d points belonging to classes %s",
            removed,
            [name for name in names if name in class_lookup],
        )
    return points[mask], colours[mask]


def compute_affine(
    x: np.ndarray, z: np.ndarray, size: int
) -> Tuple[AffineTransform, np.ndarray, np.ndarray]:
    """Prepare affine transform and pixel coordinates."""
    xmin, xmax = float(x.min()), float(x.max())
    zmin, zmax = float(z.min()), float(z.max())
    if np.isclose(xmax, xmin):
        xmax = xmin + 1.0
    if np.isclose(zmax, zmin):
        zmax = zmin + 1.0
    sx = (size - 1) / (xmax - xmin)
    sz = (size - 1) / (zmax - zmin)
    tx = -sx * xmin
    tz = -sz * zmin
    affine = AffineTransform(sx=sx, sz=sz, tx=tx, tz=tz)
    px = np.floor(sx * x + tx + 1e-6).astype(np.int32)
    pz = np.floor(sz * z + tz + 1e-6).astype(np.int32)
    np.clip(px, 0, size - 1, out=px)
    np.clip(pz, 0, size - 1, out=pz)
    return affine, px, pz


def rasterise_map(
    width: int,
    height: int,
    px: np.ndarray,
    pz: np.ndarray,
    colours: np.ndarray,
) -> np.ndarray:
    """Rasterise the semantic map into an RGB image."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[pz, px] = colours
    return image


def create_obstacle_mask(image: np.ndarray) -> np.ndarray:
    """Generate a binary occupancy mask (255 obstacle, 0 free)."""
    occupied = np.any(image != 0, axis=2)
    closed = ndimage.binary_closing(occupied, structure=np.ones((3, 3), dtype=bool))
    mask = (closed.astype(np.uint8)) * 255
    return mask


def save_outputs(
    output_dir: Path,
    map_image: np.ndarray,
    obstacle_mask: np.ndarray,
    points_x: np.ndarray,
    points_z: np.ndarray,
    colours: np.ndarray,
) -> None:
    """Persist map image, occupancy mask, and filtered point data."""
    map_path = output_dir / "map.png"
    obstacle_path = output_dir / "obstacle_mask.png"
    npz_path = output_dir / "points_xz_color.npz"

    plt.imsave(str(map_path), map_image)
    plt.imsave(str(obstacle_path), obstacle_mask, cmap="gray", vmin=0, vmax=255)
    np.savez_compressed(npz_path, x=points_x, z=points_z, rgb_255=colours)
    LOGGER.info("Saved map to %s", map_path)
    LOGGER.info("Saved obstacle mask to %s", obstacle_path)
    LOGGER.info("Saved filtered points to %s", npz_path)


def main() -> None:
    configure_logging(LOG_LEVEL)
    output_dir: Path = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    class_colours = load_class_colours(CLASSES_CSV_PATH)

    points_raw = load_numpy_points(POINTS_PATH)
    colours_raw = load_numpy_colours(COLOR01_PATH, COLOR0255_PATH)
    if points_raw.shape[0] != colours_raw.shape[0]:
        raise ValueError(
            "Point cloud and colour arrays must have the same number of rows "
            f"(got {points_raw.shape[0]} vs {colours_raw.shape[0]})."
        )

    scale_factor = 10000.0 / 255.0
    points_habitat = points_raw * scale_factor

    filtered_points, filtered_colours, y_bounds = filter_floor_ceiling(
        points_habitat, colours_raw, FLOOR_FRAC, CEIL_FRAC
    )
    filtered_points, filtered_colours = remove_named_classes(
        filtered_points,
        filtered_colours,
        class_colours,
        names=("floor", "ceiling"),
    )
    x = filtered_points[:, 0]
    z = filtered_points[:, 2]

    affine, px, pz = compute_affine(x, z, MAP_SIZE)
    map_image = rasterise_map(MAP_SIZE, MAP_SIZE, px, pz, filtered_colours)
    obstacle_mask = create_obstacle_mask(map_image)

    save_outputs(output_dir, map_image, obstacle_mask, x, z, filtered_colours)

    bbox = {
        "xmin": float(x.min()),
        "xmax": float(x.max()),
        "zmin": float(z.min()),
        "zmax": float(z.max()),
    }

    meta_path = output_dir / "map_meta.json"
    meta = {
        "proj": "xz",
        "pixels_w": int(MAP_SIZE),
        "pixels_h": int(MAP_SIZE),
        "bbox_habitat": bbox,
        "affine": {
            **affine.as_dict(),
            **affine.inverse,
        },
        "floor_y_bounds": {"y_min": y_bounds[0], "y_max": y_bounds[1]},
        "class_colors_0255": class_colours,
        "scale_factor_apartment0": scale_factor,
        "num_points_raw": int(points_raw.shape[0]),
        "num_points_filtered": int(filtered_points.shape[0]),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    LOGGER.info("Wrote metadata to %s", meta_path)


if __name__ == "__main__":
    main()
