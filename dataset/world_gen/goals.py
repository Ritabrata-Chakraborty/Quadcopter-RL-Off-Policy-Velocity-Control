#!/usr/bin/env python3
"""Paint square goal markers on occupancy maps and write pose JSON (start + goals).

World frame: map centre at origin. Each output ``*.json`` contains ``start_pose``,
``goal_pose_list`` (metres), ``start_corner_pixels``, and ``goal_corner_pixels``
(pixel bounding boxes aligned with the PNG).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from utils import (
    COLORS_JSON,
    IMAGE_EXTS,
    MAP_CELL_SIZE,
    parse_extensions,
    read_colors_for_goals,
    resolve_colors_path,
)

GOAL_SIZE_PX = 4
GOAL_MARGIN_PX = 4


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------


@dataclass
class GoalConfig:
    """Colours, goal count range, cell size, and RNG for one batch run."""

    free_color: tuple[int, int, int]
    goal_color: tuple[int, int, int]
    start_color: tuple[int, int, int] | None
    min_goals: int
    max_goals: int
    cell_size: float
    rng: random.Random


# ------------------------------------------------------------------
# World ↔ pixel geometry
# ------------------------------------------------------------------


def pixel_center_to_world(
    row: float, col: float, w: int, h: int, cell: float
) -> tuple[float, float]:
    """World (x, y) at the centre of pixel (row, col)."""
    return (
        round((col + 0.5 - w / 2.0) * cell, 4),
        round((h / 2.0 - row - 0.5) * cell, 4),
    )


def block_anchor_to_world(
    row: int, col: int, w: int, h: int, cell: float
) -> tuple[float, float]:
    """World (x, y) at the centre of a GOAL_SIZE_PX×GOAL_SIZE_PX block (top-left row, col)."""
    g2 = GOAL_SIZE_PX / 2.0
    return (
        round((col + g2 - w / 2.0) * cell, 4),
        round((h / 2.0 - row - g2) * cell, 4),
    )


def find_start_pose(
    arr: np.ndarray,
    start_color: tuple[int, int, int],
    w: int,
    h: int,
    cell: float,
) -> tuple[float, float] | None:
    """Centroid of start-coloured pixels in world metres, or None if absent."""
    mask = np.all(arr == np.array(start_color, dtype=np.uint8), axis=2)
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None
    return pixel_center_to_world(float(rows.mean()), float(cols.mean()), w, h, cell)


def find_start_corner_pixels(
    arr: np.ndarray,
    start_color: tuple[int, int, int],
) -> dict[str, list[int] | int] | None:
    """Bounding box of start-coloured pixels (numpy row, col indexing)."""
    mask = np.all(arr == np.array(start_color, dtype=np.uint8), axis=2)
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    return {
        "top_left": [r0, c0],
        "height": r1 - r0 + 1,
        "width": c1 - c0 + 1,
    }


def goal_corner_dicts(anchors: list[tuple[int, int]]) -> list[dict[str, list[int] | int]]:
    """Corner records for each goal block (top-left row, col)."""
    g = GOAL_SIZE_PX
    return [
        {"top_left": [r, c], "height": g, "width": g}
        for r, c in anchors
    ]


def save_poses(
    path: Path,
    start_pose: tuple[float, float] | None,
    goal_poses: list[tuple[float, float]],
    start_corner_pixels: dict[str, list[int] | int] | None,
    goal_corner_pixels: list[dict[str, list[int] | int]],
) -> None:
    """Write goals JSON next to the output image."""
    payload = {
        "start_pose": [start_pose[0], start_pose[1]] if start_pose else None,
        "goal_pose_list": [[x, y] for x, y in goal_poses],
        "start_corner_pixels": start_corner_pixels,
        "goal_corner_pixels": goal_corner_pixels,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# ------------------------------------------------------------------
# Goal placement
# ------------------------------------------------------------------


def blocked_mask(arr: np.ndarray, free_color: tuple[int, int, int]) -> np.ndarray:
    """True where the pixel is not free space."""
    return np.any(arr != np.array(free_color, dtype=np.uint8), axis=2)


def candidate_anchors(blocked: np.ndarray) -> list[tuple[int, int]]:
    """Top-left (row, col) where a goal fits with margin (see GOAL_SIZE_PX, GOAL_MARGIN_PX)."""
    g, m = GOAL_SIZE_PX, GOAL_MARGIN_PX
    h, w = blocked.shape
    need = g + 2 * m
    if h < need or w < need:
        return []
    prefix = np.zeros((h + 1, w + 1), dtype=np.int32)
    prefix[1:, 1:] = np.cumsum(np.cumsum(blocked.astype(np.int32), axis=0), axis=1)
    r_idx = np.arange(m, h - g - m + 1)
    c_idx = np.arange(m, w - g - m + 1)
    if r_idx.size == 0 or c_idx.size == 0:
        return []
    sums = (
        prefix[np.ix_(r_idx + g + m, c_idx + g + m)]
        - prefix[np.ix_(r_idx - m, c_idx + g + m)]
        - prefix[np.ix_(r_idx + g + m, c_idx - m)]
        + prefix[np.ix_(r_idx - m, c_idx - m)]
    )
    vi, vj = np.where(sums == 0)
    return list(zip(r_idx[vi].tolist(), c_idx[vj].tolist()))


def place_goals(
    arr: np.ndarray,
    n_goals: int,
    free_color: tuple[int, int, int],
    goal_color: tuple[int, int, int],
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Paint up to ``n_goals`` square goals; returns image copy and anchor list."""
    g, m = GOAL_SIZE_PX, GOAL_MARGIN_PX
    result = arr.copy()
    blocked = blocked_mask(result, free_color)
    h_map, w_map = blocked.shape
    candidates = candidate_anchors(blocked)
    if not candidates:
        return result, []
    rng.shuffle(candidates)
    placed: list[tuple[int, int]] = []
    for r, c in candidates:
        if len(placed) >= n_goals:
            break
        r0, r1 = max(0, r - m), min(h_map, r + g + m)
        c0, c1 = max(0, c - m), min(w_map, c + g + m)
        if np.any(blocked[r0:r1, c0:c1]):
            continue
        result[r : r + g, c : c + g] = goal_color
        blocked[r0:r1, c0:c1] = True
        placed.append((r, c))
    return result, placed


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------


def process_image(
    image_path: Path,
    output_path: Path,
    config: GoalConfig,
) -> tuple[int, int, int]:
    """Add goals, save PNG + JSON; returns (width, height, goals_placed)."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    arr = np.array(img, dtype=np.uint8)
    start_pose = (
        find_start_pose(arr, config.start_color, w, h, config.cell_size)
        if config.start_color
        else None
    )
    start_corner_pixels = (
        find_start_corner_pixels(arr, config.start_color)
        if config.start_color
        else None
    )
    n = config.rng.randint(config.min_goals, config.max_goals)
    result_arr, anchors = place_goals(arr, n, config.free_color, config.goal_color, config.rng)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result_arr).save(output_path)
    poses = [block_anchor_to_world(r, c, w, h, config.cell_size) for r, c in anchors]
    goal_corner_pixels = goal_corner_dicts(anchors)
    save_poses(
        output_path.with_suffix(".json"),
        start_pose,
        poses,
        start_corner_pixels,
        goal_corner_pixels,
    )
    return w, h, len(anchors)


def main() -> None:
    p = argparse.ArgumentParser(description="Add goal pixels and JSON poses to occupancy maps.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", metavar="FILE", help="One occupancy image.")
    g.add_argument("--input-dir", metavar="DIR", help="Directory of images.")
    p.add_argument("--output", metavar="FILE", help="Output image (single-file mode).")
    p.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Batch output dir (default: <input-dir>_goals).",
    )
    p.add_argument("--colors", default="", help=f"colors JSON (default: {COLORS_JSON}).")
    p.add_argument("--min-goals", type=int, default=3)
    p.add_argument("--max-goals", type=int, default=10)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--extensions", default=",".join(sorted(IMAGE_EXTS)))
    args = p.parse_args()
    if args.min_goals < 1 or args.max_goals < args.min_goals:
        sys.exit("Invalid --min-goals / --max-goals.")

    colors_path = resolve_colors_path(args.colors or None)
    if not colors_path.is_file():
        sys.exit(f"colors file not found: {colors_path}")

    free_color, goal_color, start_color = read_colors_for_goals(colors_path)
    rng = random.Random(args.seed)
    exts = parse_extensions(args.extensions)
    cfg = GoalConfig(
        free_color=free_color,
        goal_color=goal_color,
        start_color=start_color,
        min_goals=args.min_goals,
        max_goals=args.max_goals,
        cell_size=MAP_CELL_SIZE,
        rng=rng,
    )

    print(f"Goal colour  : {cfg.goal_color}")
    print(f"Start colour : {cfg.start_color}")
    print(f"Goal count   : [{cfg.min_goals}, {cfg.max_goals}]")
    print(f"Cell size    : {cfg.cell_size} m/px")
    print(f"Seed         : {args.seed}")

    if args.input:
        img_path = Path(args.input).resolve()
        if not img_path.is_file():
            sys.exit(f"image not found: {img_path}")
        out = (
            Path(args.output).resolve()
            if args.output
            else img_path.parent / f"{img_path.stem}_goals{img_path.suffix}"
        )
        w, h, placed = process_image(img_path, out, cfg)
        print(f"Image        : {w}x{h}  goals placed={placed}")
        print(f"Saved image  : {out}")
        return

    in_dir = Path(args.input_dir).resolve()
    if not in_dir.is_dir():
        sys.exit(f"input directory not found: {in_dir}")
    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else in_dir.parent / f"{in_dir.name}_goals"
    )
    images = sorted(x for x in in_dir.iterdir() if x.is_file() and x.suffix.lower() in exts)
    if not images:
        sys.exit(f"no images in {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input dir    : {in_dir}  ({len(images)} images)")
    print(f"Output dir   : {out_dir}\n")
    total = 0
    for img_path in images:
        w, h, placed = process_image(img_path, out_dir / img_path.name, cfg)
        total += placed
        print(f"  {img_path.name}: {w}x{h}  goals={placed}")
    print(f"\nDone. {len(images)} image(s), {total} goals total.")


if __name__ == "__main__":
    main()
