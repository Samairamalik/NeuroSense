# 🧠 NeuroSense — Multimodal Parkinson's Screening

A machine-learning screening application that analyses **handwriting drawings,
MRI brain scans and voice recordings** to produce a probabilistic, non-invasive
risk indicator for Parkinson's disease.

> ⚠️ **Not a diagnostic tool and not a medical device.** It has not been
> clinically validated. Output is a risk indicator intended to support clinical
> judgement, never to replace it.

---

## Why multimodal

Parkinson's affects motor control, neurological structure and speech. Any single
signal is noisy, so three complementary modalities are scored independently and
then combined.

## System architecture

### 1. Handwriting (spiral & wave)

| | |
|---|---|
| **Input** | Spiral and wave drawings (one image each) |
| **Pre-processing** | Grayscale → resize 200×200 → inverted Otsu threshold |
| **Features** | HOG — 12 orientations, 8×8 cells, 2×2 blocks, L2-Hys (**27,648-dim**) |
| **Model** | `SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")` |
| **Output** | P(Parkinson's) per drawing, averaged into one handwriting score |

Tremor and micrographia show up as jagged, inconsistent stroke directions, which
is precisely what HOG encodes. With only a few hundred training images an RBF SVM
generalises better than a CNN (a CNN was prototyped in
`handwriting/spiral_test.ipynb` and not adopted).

### 2. MRI scan

| | |
|---|---|
| **Input** | Brain MRI slice |
| **Pre-processing** | Grayscale → resize 128×128 → scale to [0,1] → flatten (**16,384-dim**) |
| **Model** | `SVC(kernel="rbf", class_weight="balanced")` |
| **Output** | P(Parkinson's) |

An SVM depends only on its support vectors, so it tolerates p ≫ n on a small
dataset. **Known limitation:** flattening discards all spatial structure, making
this the weakest of the three designs — see [Roadmap](#roadmap).

### 3. Voice

| | |
|---|---|
| **Input** | Short recording (sustained "ahh" works best) |
| **Pre-processing** | Mono, 16 kHz, silence trimmed |
| **Features** | 15 selected features, mostly MFCC standard deviations |
| **Model** | `RandomForestClassifier(n_estimators=100, max_depth=8)` |
| **Output** | P(Parkinson's) |

Random Forest suits tabular features on heterogeneous scales (Hz, dB,
dimensionless ratios) with no standardisation needed, and is robust to the
noisier estimates among them.

The feature set is read from the model's own `feature_names_in_` rather than a
hand-written list. This matters: a hardcoded list previously fed the model 15
values in the wrong order — see **[docs/VOICE_BUG.md](docs/VOICE_BUG.md)**.

### 4. Fusion

A weighted mean over whichever modalities were supplied (`src/fusion.py`).
Weights are equal by default and configurable in `src/config.py`.

**Caveat:** per-modality probabilities are **not calibrated against each other**.
An SVM's Platt-scaled 0.7 and a Random Forest's 0.7 do not mean the same thing,
so the fused score is indicative rather than a true probability. Calibration is
on the roadmap.

---

## Project structure

```
NeuroSense/
├── app.py                  # Flask app: routes + upload handling only
├── requirements.txt
├── src/
│   ├── config.py           # Paths, pre-processing constants, fusion weights
│   ├── handwriting_model.py
│   ├── mri_model.py
│   ├── audio_model.py
│   ├── voice_features.py   # Canonical voice feature extraction
│   └── fusion.py
├── models/                 # Trained .pkl models (committed)
├── templates/index.html
├── handwriting/            # Training notebooks + CLI test + samples
├── mri/
├── sound/
└── docs/VOICE_BUG.md
```

All model logic lives in `src/`. The Flask app and the CLI scripts are thin
wrappers over it — duplicated feature code is what caused the voice bug.

---

## Setup

```bash
git clone https://github.com/Samairamalik/NeuroSense.git
cd NeuroSense
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`scikit-learn` is **pinned to 1.6.1** because the committed models were fitted
with that version; other versions raise `InconsistentVersionWarning` and may
change predictions.

## Running

```bash
python app.py
```

Then open http://127.0.0.1:5000. For debug mode: `FLASK_DEBUG=1 python app.py`.

### Command line

```bash
python handwriting/test.py [spiral.png wave.png]
python mri/mri_test.py [scan.png]
python sound/test_sound.py [recording.wav]
```

### JSON API

```bash
curl -F spiral=@handwriting/spiral.png \
     -F wave=@handwriting/wave.png \
     -F mri=@mri/sample.png \
     -F audio=@sound/sample.wav \
     http://127.0.0.1:5000/api/predict
```

Any subset of inputs is accepted; fusion covers whatever was supplied.

---

## Roadmap

Highest value first:

1. **Evaluation harness** — no metrics are currently saved anywhere. Needs
   accuracy, sensitivity, specificity, AUC and confusion matrices per modality,
   committed as a results file.
2. **Patient-level splits for MRI** — `mri/mri_scan.ipynb` uses a plain
   `train_test_split`. If several slices come from one patient and land in both
   splits, the reported accuracy is inflated. Use `GroupKFold` on patient ID.
3. **Probability calibration** — `CalibratedClassifierCV` per modality plus
   reliability curves, so fusion combines comparable numbers.
4. **AUC-weighted fusion** — replace equal weights once (1) exists.
5. **Better MRI features** — replace raw flattened pixels with HOG or a
   pretrained CNN embedding; the single largest accuracy gain available.
6. **Commit the training code** — the voice model has no training script in the
   repo, so it is not reproducible. The datasets are also absent.
7. **Explainability** — HOG importance overlays or Grad-CAM-style heatmaps
   showing *where* tremor was detected.
8. **Data augmentation** for handwriting (small rotations, shifts, elastic
   distortion) to expand a ~100-image dataset.

## Technologies

Python 3.9+ · Flask · OpenCV · scikit-image · scikit-learn · librosa · NumPy ·
pandas · joblib
