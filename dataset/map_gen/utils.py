"""Shared paths and color JSON helpers for map_gen scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

Color = tuple[int, int, int]

IMAGE_EXTS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})

DATASET_DIR = Path(__file__).resolve().parent.parent
COLORS_JSON = DATASET_DIR / "colors.json"

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from parameter import MAP_CELL_SIZE, MAP_PIXELS


def load_rgb(data: dict[str, Any], key: str) -> Color:
    """Parse one ``[R,G,B]`` list; values in 0..255."""
    v = data[key]
    if not isinstance(v, list) or len(v) != 3:
        raise ValueError(f"'{key}' must be [R,G,B], got {v!r}")
    rgb = tuple(int(c) for c in v)
    if not all(0 <= x <= 255 for x in rgb):
        raise ValueError(f"'{key}' out of range: {rgb}")
    return rgb  # type: ignore[return-value]


def parse_extensions(s: str) -> set[str]:
    """Comma-separated extensions → ``{'.png', ...}``."""
    out: set[str] = set()
    for e in s.split(","):
        e = e.strip().lower()
        if e:
            out.add(e if e.startswith(".") else f".{e}")
    return out


def read_colors_for_goals(path: Path | None = None) -> tuple[Color, Color, Color | None]:
    """Load free, goal, and optional start RGB from ``colors.json``."""
    p = path or COLORS_JSON
    data = json.loads(p.read_text(encoding="utf-8"))
    free = load_rgb(data, "free")
    goal = load_rgb(data, "goal") if "goal" in data else (0, 0, 0)
    start = load_rgb(data, "start") if "start" in data else None
    return free, goal, start


def read_colors_for_simple(path: Path | None = None) -> dict[str, Color]:
    """Load free, occupied, and start RGB for simple map generation."""
    p = path or COLORS_JSON
    data = json.loads(p.read_text(encoding="utf-8"))
    return {
        "free": load_rgb(data, "free"),
        "occupied": load_rgb(data, "occupied"),
        "start": load_rgb(data, "start"),
    }


def resolve_colors_path(arg: str | None) -> Path:
    """Resolve CLI path to colors file; default is ``dataset/colors.json``."""
    if not arg:
        return COLORS_JSON
    return Path(arg).resolve()
