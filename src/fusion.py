"""Combine per-modality probabilities into one risk score.

A weighted mean over whichever modalities were actually supplied. Weights live
in src/config.py and are equal by default — we have no validated per-modality
AUCs yet, and inventing weights without them would be worse than averaging.

Caveat worth stating in any write-up: these probabilities are not calibrated
against each other. An SVM's Platt-scaled 0.7 and a Random Forest's 0.7 do not
carry the same meaning, so the fused number is indicative, not probabilistic.
Running CalibratedClassifierCV on each modality would fix this.
"""

from __future__ import annotations

from .config import DECISION_THRESHOLD, FUSION_WEIGHTS


def label_for(probability: float) -> str:
    return "Likely Parkinson's" if probability > DECISION_THRESHOLD else "Likely Healthy"


def fuse(modality_probabilities: dict[str, float]) -> dict | None:
    """Fuse `{modality: probability}` into a single assessment.

    Returns None if nothing was supplied. Unknown modality names raise, so a
    typo surfaces immediately instead of being silently dropped.
    """
    supplied = {k: v for k, v in modality_probabilities.items() if v is not None}
    if not supplied:
        return None

    unknown = set(supplied) - set(FUSION_WEIGHTS)
    if unknown:
        raise ValueError(f"No fusion weight configured for: {sorted(unknown)}")

    total_weight = sum(FUSION_WEIGHTS[k] for k in supplied)
    combined = sum(FUSION_WEIGHTS[k] * v for k, v in supplied.items()) / total_weight

    return {
        "probability": round(combined * 100, 2),
        "label": label_for(combined),
        "modalities": {k: round(v * 100, 2) for k, v in sorted(supplied.items())},
        "weights": {k: FUSION_WEIGHTS[k] for k in sorted(supplied)},
        "n_modalities": len(supplied),
    }
