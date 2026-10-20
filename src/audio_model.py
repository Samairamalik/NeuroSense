"""Voice analysis: acoustic features + tuned Random Forest.

Random Forest suits this well — the features are tabular and on wildly
different scales (Hz, dB, dimensionless ratios), need no standardisation, and
the forest is robust to the noisier estimates among them.

Feature selection is driven by the model's own `feature_names_in_` rather than a
hand-written list, so the input can never silently drift out of alignment with
training again.
"""

from __future__ import annotations

import joblib

from .config import MODEL_DIR
from .voice_features import build_feature_frame, extract_voice_features, model_feature_names


class VoiceModel:
    def __init__(self):
        self.model = joblib.load(MODEL_DIR / "random_forest_tuned.pkl")

    @property
    def expected_features(self) -> list[str]:
        return model_feature_names(self.model)

    def predict(self, audio_path) -> tuple[str, float, dict[str, float]]:
        """Return (label, probability of Parkinson's, all extracted features)."""
        features = extract_voice_features(audio_path)
        frame = build_feature_frame(features, self.model)

        prediction = self.model.predict(frame)[0]
        probability = self.model.predict_proba(frame)[0][1]

        label = "parkinson" if prediction == 1 else "healthy"
        return label, float(probability), features
