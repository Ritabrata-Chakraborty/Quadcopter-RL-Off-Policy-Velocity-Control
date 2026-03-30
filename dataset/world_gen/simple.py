#!/usr/bin/env python3
"""Generate empty-room occupancy PNGs: gray frame, white interior, random start patch."""

from __future__ import annotations

import random

import numpy as np
from PIL import Image

from utils import COLORS_JSON, DATASET_DIR, MAP_PIXELS, read_colors_for_simple

# ------------------------------------------------------------------
# Defaults (fixed dataset layout)
# ------------------------------------------------------------------

START_PATCH = 10
NUM_TRAIN = 2000
NUM_TEST = 50
INNER = 200
SEED = 42
CLEARANCE = 10

TRAIN_DIR = DATASET_DIR / "maps_train" / "simple"
TEST_DIR = DATASET_DIR / "maps_eval" / "simple"


def generate_map(
    rng: random.Random,
    canvas: int,
    inner: int,
    colors: dict[str, tuple[int, int, int]],
    clearance: int,
) -> np.ndarray:
    """One RGB occupancy map with random 10×10 start patch inside the free region."""
    free = np.array(colors["free"], dtype=np.uint8)
    occ = np.array(colors["occupied"], dtype=np.uint8)
    start = np.array(colors["start"], dtype=np.uint8)

    if inner > canvas:
        raise ValueError("inner > canvas")
    border = canvas - inner
    if border % 2 != 0:
        raise ValueError("(canvas - inner) must be even")
    margin = border // 2
    if inner < START_PATCH + 2 * clearance:
        raise ValueError("inner too small for clearance + start patch")

    img = np.tile(occ, (canvas, canvas, 1))
    img[margin : margin + inner, margin : margin + inner] = free

    r_lo = margin + clearance
    r_hi = margin + inner - clearance - START_PATCH
    c_lo = margin + clearance
    c_hi = margin + inner - clearance - START_PATCH
    if r_lo > r_hi or c_lo > c_hi:
        raise RuntimeError("no valid start placement")

    r = rng.randint(r_lo, r_hi + 1)
    c = rng.randint(c_lo, c_hi + 1)
    img[r : r + START_PATCH, c : c + START_PATCH] = start
    return img


def main() -> None:
    colors = read_colors_for_simple(COLORS_JSON)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(1, NUM_TRAIN + 1):
        rng = random.Random(SEED + i)
        arr = generate_map(rng, MAP_PIXELS, INNER, colors, CLEARANCE)
        Image.fromarray(arr).save(TRAIN_DIR / f"simple_{i}.png")

    for i in range(1, NUM_TEST + 1):
        rng = random.Random(SEED + 10_000 + i)
        arr = generate_map(rng, MAP_PIXELS, INNER, colors, CLEARANCE)
        Image.fromarray(arr).save(TEST_DIR / f"simple_{i}.png")

    m = (MAP_PIXELS - INNER) // 2
    print(f"Wrote {NUM_TRAIN} maps to {TRAIN_DIR}")
    print(f"Wrote {NUM_TEST} maps to {TEST_DIR}")
    print(
        f"{MAP_PIXELS}x{MAP_PIXELS}, white {INNER}x{INNER}, margin {m}px, "
        f"start {START_PATCH}px, clearance {CLEARANCE}px, {COLORS_JSON}"
    )


if __name__ == "__main__":
    main()
