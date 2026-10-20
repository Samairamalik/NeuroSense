"""Shared paths and tunable constants.

Keeping these in one place means the Flask app, the standalone test scripts and
any future training code all agree on where models live and how inputs are
pre-processed. Divergence between those three is what broke the voice pipeline
(see docs/VOICE_BUG.md).
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"

# ---- Handwriting -------------------------------------------------------
# These MUST match the training notebooks. The resulting HOG vector is
# 27,648-dimensional, which is what the saved SVCs expect (n_features_in_).
HW_IMG_SIZE = (200, 200)
HOG_ORIENTATIONS = 12
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"

# ---- MRI ---------------------------------------------------------------
# 128 x 128 grayscale, flattened to 16,384 features.
MRI_IMG_SIZE = (128, 128)

# ---- Voice -------------------------------------------------------------
AUDIO_SR = 16000
AUDIO_TRIM_TOP_DB = 20
N_MFCC = 20

# ---- Decision threshold & fusion --------------------------------------
DECISION_THRESHOLD = 0.5

# Fusion weights per modality. Equal by default: we have no validated
# per-modality AUCs to justify anything else yet. Once evaluation exists,
# set these to the relative AUCs and document the source.
FUSION_WEIGHTS = {
    "handwriting": 1.0,
    "mri": 1.0,
    "voice": 1.0,
}

# ---- Upload safety ----------------------------------------------------
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB
