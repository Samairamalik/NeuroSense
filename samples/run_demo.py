#!/usr/bin/env python3
"""Run every bundled sample through the models and print a results table.

    python samples/run_demo.py

Prints one row per sample with the true label, the model's prediction and its
probability, then a per-modality summary. Nothing here is cherry-picked: the
MRI samples were drawn at random (seed 42) from the dataset and every one that
was drawn is included, correct or not.

Read samples/README.md before presenting these numbers. The MRI figures in
particular are not what they appear to be.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

from src.fusion import label_for  # noqa: E402
from src.handwriting_model import HandwritingModel  # noqa: E402
from src.mri_model import MRIModel  # noqa: E402

SAMPLES = ROOT / "samples"
RULE = "─" * 78


def header(title: str) -> None:
    print(f"\n{RULE}\n  {title}\n{RULE}")


def run_mri() -> tuple[int, int]:
    header("MRI  ·  RBF SVM on 128×128 flattened pixels")
    model = MRIModel()
    files = sorted((SAMPLES / "mri").glob("*.png"))
    if not files:
        print("  (no MRI samples found)")
        return 0, 0

    print(f"  {'true':10s} {'predicted':10s} {'P(parkinson)':>13s}   {'sequence':28s}")
    print(f"  {'-'*10} {'-'*10} {'-'*13}   {'-'*28}")

    correct = 0
    for path in files:
        true = path.name.split("_")[0]
        sequence = path.name.split("__", 1)[1].rsplit("_", 1)[0] if "__" in path.name else "?"
        label, prob = model.predict(path)
        hit = label == true
        correct += hit
        mark = " " if hit else "  <-- wrong"
        print(f"  {true:10s} {label:10s} {prob*100:12.2f}%   {sequence[:28]:28s}{mark}")

    print(f"\n  {correct}/{len(files)} correct")
    print("  NOTE: this number is not evidence the model detects Parkinson's.")
    print("        See 'The MRI result is a scanner artefact' in samples/README.md.")
    return correct, len(files)


def run_handwriting() -> None:
    header("Handwriting  ·  HOG features + RBF SVM")
    spiral = SAMPLES / "handwriting" / "spiral_01.png"
    wave = SAMPLES / "handwriting" / "wave_01.png"
    if not (spiral.exists() and wave.exists()):
        print("  (handwriting samples missing)")
        return

    results = {}
    for kind, path in (("spiral", spiral), ("wave", wave)):
        label, prob = HandwritingModel(kind).predict(path)
        results[kind] = prob
        print(f"  {kind:10s} {path.name:22s} {label:10s} {prob*100:7.2f}%")

    combined = sum(results.values()) / len(results)
    print(f"\n  combined: {combined*100:.2f}%  —  {label_for(combined)}")
    print("  NOTE: true label unknown. These two drawings ship with the repo and")
    print("        carry no ground-truth annotation, so correctness cannot be")
    print("        checked — only that the pipeline runs end to end.")


def run_voice() -> None:
    header("Voice  ·  acoustic features + random forest")
    files = sorted((SAMPLES / "voice").glob("*.wav"))
    if not files:
        print("  (no voice samples found)")
        return

    try:
        from src.audio_model import VoiceModel
    except ImportError as exc:
        print(f"  Could not load the voice model: {exc}")
        print("  Install the audio dependencies:  pip install librosa soundfile")
        return

    model = VoiceModel()
    for path in files:
        label, prob, feats = model.predict(path)
        print(f"  {path.name:30s} {label:10s} {prob*100:7.2f}%  —  {label_for(prob)}")
        print(f"     f0 {feats['f0_mean']:.1f} Hz · jitter {feats['jitter_local']:.4f} · "
              f"shimmer {feats['shimmer_local']:.4f} · HNR {feats['hnr']:.1f} dB")

    print("\n  NOTE: only one voice sample exists in this project and its true")
    print("        label is unknown, so no accuracy can be reported.")


def main() -> int:
    print(f"\n{RULE}\n  NeuroSense — sample run")
    print("  Screening indicators only. Not a diagnosis.")
    print(RULE)

    run_mri()
    run_handwriting()
    run_voice()

    print(f"\n{RULE}")
    print("  Read samples/README.md for what these numbers do and do not show.")
    print(f"{RULE}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
