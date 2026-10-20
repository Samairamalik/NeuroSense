"""Handwriting (spiral / wave) analysis: HOG features + RBF SVM.

Parkinsonian tremor shows up as jagged, inconsistent stroke directions. HOG
captures exactly that — per-cell histograms of edge orientation — so a shaky
spiral produces a visibly different descriptor from a smooth one. With only a
few hundred training images an RBF SVM generalises better here than a CNN.
"""

from __future__ import annotations

import cv2
import joblib
import numpy as np
from skimage import feature

from .config import (
    HOG_BLOCK_NORM,
    HOG_CELLS_PER_BLOCK,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    HW_IMG_SIZE,
    MODEL_DIR,
)


def quantify_image(image: np.ndarray) -> np.ndarray:
    """HOG descriptor for a pre-thresholded grayscale drawing."""
    return feature.hog(
        image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        transform_sqrt=True,
        block_norm=HOG_BLOCK_NORM,
    )


def preprocess(image_path) -> np.ndarray:
    """Grayscale -> resize -> inverted Otsu threshold (strokes become white)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(
            f"Could not read '{image_path}' as an image. Upload a PNG/JPEG drawing."
        )

    img = cv2.resize(img, HW_IMG_SIZE)
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


class HandwritingModel:
    """One drawing type (spiral or wave): scaler + SVM + label encoder."""

    def __init__(self, kind: str):
        self.kind = kind
        self.model = joblib.load(MODEL_DIR / f"parkinsons_{kind}_model.pkl")
        self.scaler = joblib.load(MODEL_DIR / f"scaler_{kind}.pkl")
        self.encoder = joblib.load(MODEL_DIR / f"label_encoder_{kind}.pkl")
        self._parkinson_index = list(self.encoder.classes_).index("parkinson")

    def predict(self, image_path) -> tuple[str, float]:
        """Return (label, probability of Parkinson's)."""
        feats = quantify_image(preprocess(image_path)).reshape(1, -1)

        if feats.shape[1] != self.model.n_features_in_:
            raise RuntimeError(
                f"{self.kind}: produced {feats.shape[1]} HOG features but the model "
                f"expects {self.model.n_features_in_}. Pre-processing constants in "
                "src/config.py no longer match the trained model."
            )

        feats = self.scaler.transform(feats)
        prediction = self.model.predict(feats)[0]
        probability = self.model.predict_proba(feats)[0][self._parkinson_index]

        label = self.encoder.inverse_transform([prediction])[0]
        return str(label), float(probability)
