"""Interactive CLI: label RGB bands in a map image and write dataset/colors.json."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Install Pillow: pip install Pillow") from exc

_DATASET = Path(__file__).resolve().parent
sys.path.insert(0, str(_DATASET / "world_gen"))
from utils import IMAGE_EXTS

CONFIG = _DATASET / "colors.json"
LABELS = ("free", "occupied", "start")


def first_image(path: Path) -> Path:
    """Return ``path`` if it is a file, else the first image file in ``path``."""
    if path.is_file():
        return path
    for candidate in sorted(path.iterdir()):
        if candidate.suffix.lower() in IMAGE_EXTS:
            return candidate
    raise FileNotFoundError(f"No image in {path}")


def ask_index(prompt: str, n: int) -> int:
    """Read 1-based index in ``[1, n]`` from stdin."""
    while True:
        raw = input(prompt).strip()
        if raw.isdigit() and 1 <= int(raw) <= n:
            return int(raw) - 1
        print(f"  Enter 1–{n}.")


def ask_rgb(prompt: str, default: tuple[int, int, int]) -> list[int]:
    """Parse three 0–255 ints or return ``default``."""
    raw = input(prompt).strip()
    if not raw:
        return list(default)
    try:
        parts = [int(x) for x in raw.replace(",", " ").split()]
        if len(parts) == 3 and all(0 <= v <= 255 for v in parts):
            return parts
    except ValueError:
        pass
    print(f"  Using default {list(default)}.")
    return list(default)


def main() -> None:
    ap = argparse.ArgumentParser(description="Map RGB labels → colors.json")
    ap.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=_DATASET / "maps_train" / "outdoor",
        help="Image file or directory (first image if dir).",
    )
    args = ap.parse_args()

    img_path = first_image(args.path.resolve())
    print(f"Using: {img_path}\n")

    pixels = list(Image.open(img_path).convert("RGB").getdata())
    counts = Counter(pixels)
    found = sorted(counts.keys())
    total = len(pixels)

    print("Colors found:")
    for i, rgb in enumerate(found, 1):
        print(f"  {i}: {rgb}  ({100 * counts[rgb] / total:.1f}%)")
    print()

    out: dict[str, list[int]] = {}
    for label in LABELS:
        idx = ask_index(f"Which number is '{label}'? (1-{len(found)}): ", len(found))
        out[label] = list(found[idx])

    out["goal"] = ask_rgb(
        "\nGoal marker RGB [default 0,0,0]: ",
        (0, 0, 0),
    )

    CONFIG.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {CONFIG}")


if __name__ == "__main__":
    main()
