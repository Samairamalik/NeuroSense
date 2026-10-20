"""NeuroSense — multimodal Parkinson's screening web app.

Not a diagnostic tool. It produces risk indicators intended to support, never
replace, clinical judgement.

All model and pre-processing logic lives in src/ so that the web app, the
standalone test scripts and any future training code share one implementation.
"""

from __future__ import annotations

import os
import re
import uuid
import warnings

from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.utils import secure_filename

from src.audio_model import VoiceModel
from src.config import (
    ALLOWED_AUDIO_EXTENSIONS,
    ALLOWED_IMAGE_EXTENSIONS,
    BASE_DIR,
    MAX_UPLOAD_BYTES,
    UPLOAD_DIR,
)
from src.fusion import fuse, label_for
from src.handwriting_model import HandwritingModel
from src.mri_model import MRIModel

warnings.filterwarnings("ignore")

app = Flask(__name__)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# Loaded once at import — these pickles are large (44 MB for MRI alone).
spiral_model = HandwritingModel("spiral")
wave_model = HandwritingModel("wave")
mri_model = MRIModel()
voice_model = VoiceModel()


# ======================================================
# ASSETS
# ======================================================

@app.context_processor
def inject_asset_helper():
    """Stamp static URLs with the file's mtime.

    Without this, browsers keep serving a cached stylesheet or script after a
    deploy, so a user can end up with new markup and old CSS.
    """
    static_root = BASE_DIR / "static"

    def asset(filename):
        try:
            stamp = int((static_root / filename).stat().st_mtime)
        except OSError:
            stamp = 0
        return url_for("static", filename=filename, v=stamp)

    return {"asset": asset}


# ======================================================
# UPLOAD HANDLING
# ======================================================

def _extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def save_upload(file_storage, allowed_extensions: set[str], field: str) -> str:
    """Validate and save one upload, returning its stored filename.

    The stored name is prefixed with a random token so that two users uploading
    'spiral.png' cannot overwrite each other's scans.
    """
    original = secure_filename(file_storage.filename or "")
    if not original:
        raise ValueError(f"No file supplied for '{field}'.")

    ext = _extension(original)
    if ext not in allowed_extensions:
        raise ValueError(
            f"'{original}' has an unsupported type for {field}. "
            f"Allowed: {', '.join(sorted(allowed_extensions))}."
        )

    stored = f"{uuid.uuid4().hex[:12]}_{original}"
    file_storage.save(os.path.join(app.config["UPLOAD_FOLDER"], stored))
    return stored


def _path(stored_name: str) -> str:
    return os.path.join(app.config["UPLOAD_FOLDER"], stored_name)


def _sanitise(message: str) -> str:
    """Strip server filesystem paths out of a message before showing it.

    Underlying libraries put the full path of the saved upload into their
    errors. Echoing that to the browser discloses the server's directory
    layout and tells the user nothing useful, so paths are reduced to the
    bare filename.
    """
    upload_dir = str(UPLOAD_DIR)
    message = message.replace(upload_dir + os.sep, "").replace(upload_dir, "")
    # Any remaining absolute path -> its final component.
    return re.sub(r"(/[\w.\-]+)+/([\w.\-]+)", r"\2", message)


# ======================================================
# PER-MODALITY ANALYSIS
# ======================================================

def analyse_handwriting(spiral_file, wave_file) -> dict:
    spiral_name = save_upload(spiral_file, ALLOWED_IMAGE_EXTENSIONS, "spiral image")
    wave_name = save_upload(wave_file, ALLOWED_IMAGE_EXTENSIONS, "wave image")

    spiral_label, spiral_prob = spiral_model.predict(_path(spiral_name))
    wave_label, wave_prob = wave_model.predict(_path(wave_name))

    # The two drawings are two views of the same modality, so they are averaged
    # here and the result enters tri-modal fusion as a single 'handwriting' vote.
    combined = (spiral_prob + wave_prob) / 2

    return {
        "spiral_img": spiral_name,
        "wave_img": wave_name,
        "spiral_label": spiral_label,
        "spiral_prob": round(spiral_prob * 100, 2),
        "wave_label": wave_label,
        "wave_prob": round(wave_prob * 100, 2),
        "final": label_for(combined),
        "final_prob": round(combined * 100, 2),
        "_raw": combined,
    }


def analyse_mri(mri_file) -> dict:
    name = save_upload(mri_file, ALLOWED_IMAGE_EXTENSIONS, "MRI scan")
    label, prob = mri_model.predict(_path(name))

    return {
        "mri_img": name,
        "label": label,
        "prob": round(prob * 100, 2),
        "final": label_for(prob),
        "_raw": prob,
    }


def analyse_voice(audio_file) -> dict:
    name = save_upload(audio_file, ALLOWED_AUDIO_EXTENSIONS, "voice recording")
    label, prob, features = voice_model.predict(_path(name))

    # Show the features the model actually consumed first, then the rest.
    used = voice_model.expected_features

    return {
        "audio_file": name,
        "label": label,
        "prob": round(prob * 100, 2),
        "final": label_for(prob),
        "model_features": {k: round(features[k], 4) for k in used},
        "features": {
            k.replace("_", " ").title(): round(v, 4)
            for k, v in features.items()
            if k in {
                "f0_mean", "f0_std", "f0_median", "jitter_local", "jitter_rap",
                "shimmer_local", "hnr", "energy_mean", "energy_std",
                "zcr_mean", "zcr_std",
            }
        },
        "_raw": prob,
    }


# ======================================================
# ROUTES
# ======================================================

# Each modality renders its own page. The POST contract is unchanged: every
# form still posts to "/" with a test_type, and the same handler runs the same
# analysis functions — only the template chosen for the response is new.
MODALITY_PAGES = {
    "handwriting": ("handwriting.html", "handwriting"),
    "mri": ("mri.html", "mri"),
    "voice": ("voice.html", "voice"),
    "combined": ("combined.html", "combined"),
}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", active="home")

    handwriting_result = mri_result = voice_result = fusion_result = None
    error = None
    test = request.form.get("test_type")

    try:
        spiral = request.files.get("spiral")
        wave = request.files.get("wave")
        mri_file = request.files.get("mri")
        audio = request.files.get("audio")

        def has(f):
            return f is not None and f.filename

        if test == "handwriting":
            if not (has(spiral) and has(wave)):
                raise ValueError("Both a spiral and a wave image are required.")
            handwriting_result = analyse_handwriting(spiral, wave)

        elif test == "mri":
            if not has(mri_file):
                raise ValueError("An MRI scan is required.")
            mri_result = analyse_mri(mri_file)

        elif test == "voice":
            if not has(audio):
                raise ValueError("A voice recording is required.")
            voice_result = analyse_voice(audio)

        elif test == "combined":
            # Tri-modal screening: use whatever the user supplied.
            if has(spiral) and has(wave):
                handwriting_result = analyse_handwriting(spiral, wave)
            if has(mri_file):
                mri_result = analyse_mri(mri_file)
            if has(audio):
                voice_result = analyse_voice(audio)

            if not any([handwriting_result, mri_result, voice_result]):
                raise ValueError(
                    "Supply at least one of: both handwriting drawings, an MRI "
                    "scan, or a voice recording."
                )
        else:
            raise ValueError(f"Unknown test type: {test!r}")

        fusion_result = fuse({
            "handwriting": handwriting_result["_raw"] if handwriting_result else None,
            "mri": mri_result["_raw"] if mri_result else None,
            "voice": voice_result["_raw"] if voice_result else None,
        })

    except ValueError as exc:
        error = _sanitise(str(exc))
    except Exception as exc:  # model/IO failure — show it rather than a 500
        error = _sanitise(f"Analysis failed: {exc}")

    template, active = MODALITY_PAGES.get(test, ("index.html", "home"))
    return render_template(
        template,
        handwriting_result=handwriting_result,
        mri_result=mri_result,
        voice_result=voice_result,
        fusion_result=fusion_result,
        error=error,
        active=active,
    )


# ---- Modality pages (GET only; they post back to "/") ----------------

@app.route("/handwriting")
def spiral_page():
    return render_template("handwriting.html", active="handwriting")


@app.route("/mri")
def mri_page():
    return render_template("mri.html", active="mri")


@app.route("/voice")
def voice_page():
    return render_template("voice.html", active="voice")


@app.route("/combined")
def combined_page():
    return render_template("combined.html", active="combined")


# ---- Error pages ------------------------------------------------------

@app.errorhandler(404)
def not_found(_exc):
    return render_template(
        "error.html",
        code=404,
        heading="That page does not exist",
        detail="The address may be mistyped, or the page may have moved.",
        retry_url=None,
        active=None,
    ), 404


@app.errorhandler(413)
def too_large(_exc):
    limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    return render_template(
        "error.html",
        code=413,
        heading="That file is too large",
        detail=(
            f"Uploads are limited to {limit_mb} MB. Try exporting the scan or "
            "recording at a smaller size, then upload it again."
        ),
        retry_url=url_for("combined_page"),
        active=None,
    ), 413


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON endpoint. Accepts any subset of: spiral+wave, mri, audio."""
    results: dict[str, object] = {}

    try:
        spiral, wave = request.files.get("spiral"), request.files.get("wave")
        mri_file, audio = request.files.get("mri"), request.files.get("audio")

        if spiral and wave:
            hw = analyse_handwriting(spiral, wave)
            results["handwriting"] = {k: v for k, v in hw.items() if not k.startswith("_")}
        if mri_file:
            m = analyse_mri(mri_file)
            results["mri"] = {k: v for k, v in m.items() if not k.startswith("_")}
        if audio:
            v = analyse_voice(audio)
            results["voice"] = {k: val for k, val in v.items() if not k.startswith("_")}

        if not results:
            return jsonify({"error": "No inputs supplied."}), 400

        results["fusion"] = fuse({
            "handwriting": results.get("handwriting", {}).get("final_prob", 0) / 100
            if "handwriting" in results else None,
            "mri": results.get("mri", {}).get("prob", 0) / 100 if "mri" in results else None,
            "voice": results.get("voice", {}).get("prob", 0) / 100 if "voice" in results else None,
        })
        results["disclaimer"] = (
            "Screening indicator only. Not a diagnosis. Consult a clinician."
        )
        return jsonify(results)

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Analysis failed: {exc}"}), 500


if __name__ == "__main__":
    # debug is opt-in: FLASK_DEBUG=1 python app.py
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
