"""Standalone MRI prediction — CLI wrapper around src.mri_model.

Usage:
    python mri/mri_test.py                # uses the bundled sample
    python mri/mri_test.py scan.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fusion import label_for  # noqa: E402
from src.mri_model import MRIModel  # noqa: E402


def main(argv: list[str]) -> int:
    image_path = Path(argv[1]) if len(argv) > 1 else Path(__file__).parent / "sample.png"

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return 1

    label, probability = MRIModel().predict(image_path)

    print(f"\nMRI scan: {image_path}")
    print(f"Prediction:        {label}")
    print(f"Parkinson's prob.: {probability * 100:.2f}%")
    print(f"Assessment:        {label_for(probability)}")
    print("\nScreening indicator only — not a diagnosis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
