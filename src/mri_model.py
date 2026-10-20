"""MRI analysis: normalised pixel intensities + RBF SVM.

Note the limitation: flattening discards all spatial structure, so the model is
sensitive to slice position, orientation and contrast. It works because the
dataset is small (an SVM depends only on its support vectors, so it tolerates
p >> n), but extracting features first — HOG, or a pretrained CNN embedding —
is the clearest available accuracy upgrade.

Note the saved model is SVC(kernel="rbf", C=1, gamma="scale") with NO class
weighting, despite the dataset being imbalanced 2.76:1 toward the normal class.
The training notebook fits two SVCs and the second, unweighted one is what was
persisted.
"""

from __future__ import annotations

import cv2
import joblib
import numpy as np

from .config import MODEL_DIR, MRI_IMG_SIZE


class MRIModel:
    def __init__(self):
        self.model = joblib.load(MODEL_DIR / "parkinsons_mri_model.pkl")

    @staticmethod
    def preprocess(image_path) -> np.ndarray:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(
                f"Could not read '{image_path}' as an image. Upload a PNG/JPEG scan."
            )

        img = cv2.resize(img, MRI_IMG_SIZE) / 255.0
        return img.reshape(1, -1)

    def predict(self, image_path) -> tuple[str, float]:
        """Return (label, probability of Parkinson's)."""
        x = self.preprocess(image_path)

        if x.shape[1] != self.model.n_features_in_:
            raise RuntimeError(
                f"MRI: produced {x.shape[1]} pixel features but the model expects "
                f"{self.model.n_features_in_}. MRI_IMG_SIZE in src/config.py no "
                "longer matches the trained model."
            )

        prediction = self.model.predict(x)[0]
        # classes_ is [0, 1]; index 1 is the Parkinson's class.
        probability = self.model.predict_proba(x)[0][1]

        label = "parkinson" if prediction == 1 else "normal"
        return label, float(probability)
