"""Canonical voice feature extraction.

THIS IS THE ONLY PLACE voice features are computed. The Flask app and the
standalone script both import from here. Previously each had its own copy and
they had silently drifted apart (unnormalised jitter, HNR hardcoded to zero),
which meant the web app fed the classifier values on the wrong scale.

The full dictionary of features is computed, then the model's own
`feature_names_in_` attribute decides which ones are passed to it and in what
order. That removes the hand-maintained ordering list that caused the original
bug: if the model is retrained on a different feature set, this code follows
automatically.
"""

from __future__ import annotations

import librosa
import numpy as np
import pandas as pd

from .config import AUDIO_SR, AUDIO_TRIM_TOP_DB, N_MFCC

_FMIN = librosa.note_to_hz("C2")
_FMAX = librosa.note_to_hz("C7")


def load_audio(path, target_sr: int = AUDIO_SR):
    """Load audio as mono at `target_sr`, with leading/trailing silence trimmed."""
    y, sr = librosa.load(path, sr=None, mono=False)

    if y.ndim > 1:
        y = librosa.to_mono(y)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    y, _ = librosa.effects.trim(y, top_db=AUDIO_TRIM_TOP_DB)

    if y.size == 0:
        raise ValueError("Audio is empty after silence trimming — recording too quiet?")

    return y, target_sr


def _f0_track(y, sr):
    """Voiced fundamental-frequency track, NaNs removed."""
    f0, _, _ = librosa.pyin(y, fmin=_FMIN, fmax=_FMAX, sr=sr)
    return f0[~np.isnan(f0)]


def _f0_stats(f0):
    if f0.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(f0)), float(np.std(f0)), float(np.median(f0))


def _jitter(f0):
    """Cycle-to-cycle period variation, normalised by the mean period.

    Normalisation matters: without it the values come out ~100x smaller and no
    longer resemble anything the model was trained on.
    """
    if f0.size < 2:
        return 0.0, 0.0

    periods = 1.0 / (f0 + 1e-8)
    mean_period = np.mean(periods) + 1e-8

    jitter_local = float(np.mean(np.abs(np.diff(periods))) / mean_period)

    if periods.size >= 3:
        # Relative Average Perturbation: each period against its neighbours' mean.
        neighbour_mean = (periods[:-2] + periods[2:]) / 2.0
        rap = float(np.mean(np.abs(periods[1:-1] - neighbour_mean)) / mean_period)
    else:
        rap = jitter_local

    return jitter_local, rap


def _shimmer(y, sr):
    """Amplitude perturbation over 25 ms frames, normalised by mean amplitude."""
    frame_length = int(sr * 0.025)
    hop_length = int(sr * 0.010)
    amplitude = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    if amplitude.size < 2:
        return 0.0

    return float(np.mean(np.abs(np.diff(amplitude))) / (np.mean(amplitude) + 1e-8))


def _hnr(y, sr, f0_mean):
    """Harmonics-to-noise ratio in dB.

    Real implementation — the web app used to hardcode this to 0, destroying one
    of the most clinically meaningful features for dysphonia.
    """
    if f0_mean <= 0:
        return 0.0

    power = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)

    harmonic_indices = []
    for h in range(1, 6):  # first five harmonics
        idx = int(np.argmin(np.abs(freqs - h * f0_mean)))
        if idx < power.size:
            harmonic_indices.append(idx)

    if not harmonic_indices:
        return 0.0

    harmonic_power = float(np.sum(power[harmonic_indices]))
    noise_power = float(np.sum(power)) - harmonic_power

    if noise_power <= 0 or harmonic_power <= 0:
        return 0.0

    return float(10.0 * np.log10(harmonic_power / noise_power))


def extract_voice_features(path) -> dict[str, float]:
    """Compute the full voice feature dictionary for one recording.

    Returns every feature the project knows how to compute (pitch, jitter,
    shimmer, HNR, 20 MFCCs with deltas, energy, ZCR). Selecting a subset is the
    caller's job — see `build_feature_frame`.
    """
    y, sr = load_audio(path)

    f0 = _f0_track(y, sr)
    f0_mean, f0_std, f0_median = _f0_stats(f0)
    jitter_local, jitter_rap = _jitter(f0)

    features: dict[str, float] = {
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_median": f0_median,
        "jitter_local": jitter_local,
        "jitter_rap": jitter_rap,
        "shimmer_local": _shimmer(y, sr),
        "hnr": _hnr(y, sr, f0_mean),
    }

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

    for i in range(N_MFCC):
        features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
        features[f"mfcc_{i}_std"] = float(np.std(mfccs[i]))
        features[f"mfcc_delta_{i}_mean"] = float(np.mean(mfcc_delta[i]))
        features[f"mfcc_delta2_{i}_mean"] = float(np.mean(mfcc_delta2[i]))

    energy = librosa.feature.rms(y=y)[0]
    features["energy_mean"] = float(np.mean(energy))
    features["energy_std"] = float(np.std(energy))

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    return features


def model_feature_names(model) -> list[str]:
    """The feature names a fitted estimator was trained on.

    sklearn records these when the model is fitted on a DataFrame, which is how
    we recover the ground truth instead of guessing at an ordering.
    """
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        raise RuntimeError(
            "The voice model does not record feature names, so the correct input "
            "ordering cannot be recovered. Retrain it on a pandas DataFrame, or "
            "pin the ordering explicitly here with a comment citing the training code."
        )
    return [str(n) for n in names]


def build_feature_frame(features: dict[str, float], model) -> pd.DataFrame:
    """Arrange `features` into the exact columns the model expects."""
    names = model_feature_names(model)

    missing = [n for n in names if n not in features]
    if missing:
        raise ValueError(
            "Cannot build model input — extraction did not produce these "
            f"features the model requires: {missing}"
        )

    return pd.DataFrame([[features[n] for n in names]], columns=names)
