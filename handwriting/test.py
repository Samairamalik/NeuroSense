"""Standalone handwriting prediction — CLI wrapper around src.handwriting_model.

Usage:
    python handwriting/test.py                       # uses the bundled samples
    python handwriting/test.py spiral.png wave.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fusion import label_for  # noqa: E402
from src.handwriting_model import HandwritingModel  # noqa: E402


def main(argv: list[str]) -> int:
    here = Path(__file__).resolve().parent
    spiral_path = Path(argv[1]) if len(argv) > 1 else here / "spiral.png"
    wave_path = Path(argv[2]) if len(argv) > 2 else here / "wave.png"

    for path in (spiral_path, wave_path):
        if not path.exists():
            print(f"Image not found: {path}")
            return 1

    results = {}
    for kind, path in (("spiral", spiral_path), ("wave", wave_path)):
        label, probability = HandwritingModel(kind).predict(path)
        results[kind] = probability
        print(f"{kind.capitalize():8s} {path.name:20s} {label:10s} {probability * 100:6.2f}%")

    combined = sum(results.values()) / len(results)
    print(f"\nHandwriting risk: {combined * 100:.2f}% — {label_for(combined)}")
    print("Screening indicator only — not a diagnosis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
