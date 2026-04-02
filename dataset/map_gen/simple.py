#!/usr/bin/env python3
"""Generate simple occupancy maps: bordered free region with random obstacles and start patch."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from utils import COLORS_JSON, DATASET_DIR, MAP_PIXELS, read_colors_for_simple

# Dataset generation config
NUM_TRAIN = 2000
NUM_TEST = 50
SEED = 42

TRAIN_DIR = DATASET_DIR / "maps_train" / "simple"
TEST_DIR = DATASET_DIR / "maps_eval" / "simple"


@dataclass
class MapConfig:
    """Configuration for simple map generation."""
    canvas_size: int        # Total image width/height in pixels
    inner_size: int         # Free region width/height in pixels
    start_patch_size: int   # Start marker width/height in pixels
    edge_clearance: int     # Min distance from border to start/obstacles
    obs_min_side: int       # Obstacle min width/height
    obs_max_side: int       # Obstacle max width/height
    obs_min_density: float  # Min fraction of free area covered by obstacles
    obs_max_density: float  # Max fraction of free area covered by obstacles


DEFAULT_CONFIG = MapConfig(
    canvas_size=MAP_PIXELS,
    inner_size=200,
    start_patch_size=10,
    edge_clearance=10,
    obs_min_side=5,
    obs_max_side=10,
    obs_min_density=0.01,
    obs_max_density=0.05,
)


class MapGenerator:
    """Generates occupancy maps with obstacles and start positions."""

    def __init__(self, config: MapConfig = DEFAULT_CONFIG):
        self.config = config
        self.validate_config()

    def validate_config(self) -> None:
        cfg = self.config
        if cfg.inner_size > cfg.canvas_size:
            raise ValueError(f"inner_size ({cfg.inner_size}) > canvas_size ({cfg.canvas_size})")
        border = cfg.canvas_size - cfg.inner_size
        if border % 2 != 0:
            raise ValueError(f"(canvas_size - inner_size) must be even, got {border}")
        min_space = cfg.start_patch_size + 2 * cfg.edge_clearance
        if cfg.inner_size < min_space:
            raise ValueError(
                f"inner_size ({cfg.inner_size}) too small for "
                f"start_patch ({cfg.start_patch_size}) + clearance ({cfg.edge_clearance})"
            )
        if cfg.obs_min_side > cfg.obs_max_side:
            raise ValueError(f"obs_min_side ({cfg.obs_min_side}) > obs_max_side ({cfg.obs_max_side})")

    def generate(self, rng: random.Random, colors: dict[str, tuple[int, int, int]]) -> np.ndarray:
        """Generate one RGB occupancy map."""
        cfg = self.config
        free = np.array(colors["free"], dtype=np.uint8)
        occ = np.array(colors["occupied"], dtype=np.uint8)
        start = np.array(colors["start"], dtype=np.uint8)

        margin = (cfg.canvas_size - cfg.inner_size) // 2
        img = np.tile(occ, (cfg.canvas_size, cfg.canvas_size, 1))
        img[margin : margin + cfg.inner_size, margin : margin + cfg.inner_size] = free

        start_pos = self.reserve_start_position(rng)
        start_zone = self.get_reserved_zone(start_pos)
        self.place_obstacles(rng, img, free, occ, start_zone, margin)
        self.place_start(img, start_pos, start)

        return img

    def reserve_start_position(self, rng: random.Random) -> tuple[int, int]:
        """Pick a random valid position for the start patch (respecting edge clearance)."""
        cfg = self.config
        margin = (cfg.canvas_size - cfg.inner_size) // 2
        r_min = margin + cfg.edge_clearance
        r_max = margin + cfg.inner_size - cfg.edge_clearance - cfg.start_patch_size
        c_min = margin + cfg.edge_clearance
        c_max = margin + cfg.inner_size - cfg.edge_clearance - cfg.start_patch_size

        if r_min > r_max or c_min > c_max:
            raise RuntimeError("Invalid start position bounds")

        return (rng.randint(r_min, r_max + 1), rng.randint(c_min, c_max + 1))

    def get_reserved_zone(self, start_pos: tuple[int, int]) -> tuple[int, int, int, int]:
        """Return bounding box of reserved zone (start patch + clearance)."""
        r, c = start_pos
        cfg = self.config
        return (
            max(0, r - cfg.edge_clearance),
            min(cfg.canvas_size, r + cfg.start_patch_size + cfg.edge_clearance),
            max(0, c - cfg.edge_clearance),
            min(cfg.canvas_size, c + cfg.start_patch_size + cfg.edge_clearance),
        )

    def place_obstacles(
        self,
        rng: random.Random,
        img: np.ndarray,
        free: np.ndarray,
        occ: np.ndarray,
        start_zone: tuple[int, int, int, int],
        margin: int,
    ) -> None:
        """Place random square obstacles with clearance, avoiding the start zone."""
        cfg = self.config
        target_area = int(
            cfg.inner_size * cfg.inner_size
            * rng.uniform(cfg.obs_min_density, cfg.obs_max_density)
        )
        placed_area = 0
        max_attempts = 100

        while placed_area < target_area:
            side = rng.randint(cfg.obs_min_side, cfg.obs_max_side + 1)
            for _ in range(max_attempts):
                r = rng.randint(margin, margin + cfg.inner_size - side)
                c = rng.randint(margin, margin + cfg.inner_size - side)
                if self.can_place_obstacle(img, r, c, side, free, start_zone):
                    img[r : r + side, c : c + side] = occ
                    placed_area += side * side
                    break

    def can_place_obstacle(
        self,
        img: np.ndarray,
        r: int,
        c: int,
        side: int,
        free: np.ndarray,
        start_zone: tuple[int, int, int, int],
    ) -> bool:
        """Return True if an obstacle can be placed: clearance zone free and no start overlap."""
        cfg = self.config
        check_r_min = max(0, r - cfg.edge_clearance)
        check_r_max = min(cfg.canvas_size, r + side + cfg.edge_clearance)
        check_c_min = max(0, c - cfg.edge_clearance)
        check_c_max = min(cfg.canvas_size, c + side + cfg.edge_clearance)

        region = img[check_r_min:check_r_max, check_c_min:check_c_max]
        is_clearance_free = np.all(region == free)
        no_start_overlap = not rects_overlap(
            check_r_min, check_r_max, check_c_min, check_c_max, start_zone
        )
        return is_clearance_free and no_start_overlap

    def place_start(self, img: np.ndarray, pos: tuple[int, int], start: np.ndarray) -> None:
        """Place start patch at reserved position."""
        r, c = pos
        cfg = self.config
        img[r : r + cfg.start_patch_size, c : c + cfg.start_patch_size] = start


def rects_overlap(
    r_min1: int, r_max1: int, c_min1: int, c_max1: int,
    zone: tuple[int, int, int, int],
) -> bool:
    """Return True if rectangle (r_min1..r_max1, c_min1..c_max1) overlaps zone."""
    r_min2, r_max2, c_min2, c_max2 = zone
    return not (r_max1 <= r_min2 or r_min1 >= r_max2 or c_max1 <= c_min2 or c_min1 >= c_max2)


def generate_dataset(
    train_count: int,
    test_count: int,
    train_dir: Path,
    test_dir: Path,
    seed: int = SEED,
    config: MapConfig = DEFAULT_CONFIG,
) -> None:
    """Generate and save train/test maps."""
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    colors = read_colors_for_simple(COLORS_JSON)
    generator = MapGenerator(config)

    for i in range(1, train_count + 1):
        rng = random.Random(seed + i)
        arr = generator.generate(rng, colors)
        Image.fromarray(arr).save(train_dir / f"simple_{i}.png")

    for i in range(1, test_count + 1):
        rng = random.Random(seed + 10_000 + i)
        arr = generator.generate(rng, colors)
        Image.fromarray(arr).save(test_dir / f"simple_{i}.png")

    margin = (config.canvas_size - config.inner_size) // 2
    print(f"Generated {train_count} training maps -> {train_dir}")
    print(f"Generated {test_count} test maps -> {test_dir}")
    print(
        f"  Canvas: {config.canvas_size}x{config.canvas_size} | "
        f"Free: {config.inner_size}x{config.inner_size} | "
        f"Margin: {margin}px"
    )
    print(
        f"  Start: {config.start_patch_size}x{config.start_patch_size} | "
        f"Clearance: {config.edge_clearance}px | "
        f"Obstacles: {config.obs_min_density:.0%}-{config.obs_max_density:.0%}"
    )


def main() -> None:
    generate_dataset(NUM_TRAIN, NUM_TEST, TRAIN_DIR, TEST_DIR)


if __name__ == "__main__":
    main()
