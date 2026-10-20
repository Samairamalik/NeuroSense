"""Standalone voice prediction — CLI wrapper around src.audio_model.

Feature extraction lives in src/voice_features.py. This file deliberately
contains no extraction logic of its own: the previous version had its own copy,
which drifted from both the app and the training code (see docs/VOICE_BUG.md).

Usage:
    python sound/test_sound.py sound/sample.wav
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio_model import VoiceModel  # noqa: E402
from src.fusion import label_for  # noqa: E402


def main(argv: list[str]) -> int:
    audio_path = Path(argv[1]) if len(argv) > 1 else Path(__file__).parent / "sample.wav"

    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        return 1

    model = VoiceModel()
    label, probability, features = model.predict(audio_path)

    print(f"\nAudio: {audio_path}")
    print(f"Prediction:        {label}")
    print(f"Parkinson's prob.: {probability * 100:.2f}%")
    print(f"Assessment:        {label_for(probability)}")

    print("\nFeatures consumed by the model (in training order):")
    for name in model.expected_features:
        print(f"  {name:24s} {features[name]:12.6f}")

    print("\nClinical acoustic measures:")
    for name in ("f0_mean", "f0_std", "jitter_local", "jitter_rap", "shimmer_local", "hnr"):
        unit = " dB" if name == "hnr" else (" Hz" if name.startswith("f0") else "")
        print(f"  {name:24s} {features[name]:12.6f}{unit}")

    print("\nScreening indicator only — not a diagnosis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
