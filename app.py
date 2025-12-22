from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import joblib
from skimage import feature
import librosa
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ALLOWED_AUDIO_EXTENSIONS"] = {"wav", "mp3", "flac"}

# ======================================================
# LOAD MODELS
# ======================================================

spiral_model = joblib.load("models/parkinsons_spiral_model.pkl")
wave_model = joblib.load("models/parkinsons_wave_model.pkl")

spiral_scaler = joblib.load("models/scaler_spiral.pkl")
wave_scaler = joblib.load("models/scaler_wave.pkl")

spiral_le = joblib.load("models/label_encoder_spiral.pkl")
wave_le = joblib.load("models/label_encoder_wave.pkl")

mri_model = joblib.load("models/parkinsons_mri_model.pkl")  # Use the correct model
audio_model = joblib.load("models/random_forest_tuned.pkl")

# ======================================================
# HANDWRITING
# ======================================================

def quantify_image(image):
    return feature.hog(
        image,
        orientations=12,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm="L2-Hys",
    )

def handwriting_predict(image_path, model, scaler, le):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (200, 200))
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    feats = quantify_image(img)
    feats = scaler.transform([feats])

    pred = model.predict(feats)[0]
    prob = model.predict_proba(feats)[0]
    label = le.inverse_transform([pred])[0]

    idx = list(le.classes_).index("parkinson")
    return label, prob[idx]

# ======================================================
# MRI
# ======================================================

def mri_predict(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid MRI image")
    
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = img.reshape(1, -1)

    pred = mri_model.predict(img)[0]
    probs = mri_model.predict_proba(img)[0]
    
    parkinson_prob = probs[1]  # Parkinson class probability

    label = "parkinson" if pred == 1 else "normal"
    return label, parkinson_prob

# ======================================================
# VOICE FEATURE EXTRACTION
# ======================================================

def load_audio(path, sr=16000):
    y, _ = librosa.load(path, sr=sr)
    y, _ = librosa.effects.trim(y)
    return y, sr

def extract_voice_features(path):
    y, sr = load_audio(path)

    features = {}

    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr
    )
    f0 = f0[~np.isnan(f0)]
    features["f0_mean"] = np.mean(f0) if len(f0) else 0
    features["f0_std"] = np.std(f0) if len(f0) else 0
    features["f0_median"] = np.median(f0) if len(f0) else 0

    periods = 1 / (f0 + 1e-8) if len(f0) else np.array([0])
    features["jitter_local"] = np.mean(np.abs(np.diff(periods))) if len(periods) > 1 else 0
    features["jitter_rap"] = features["jitter_local"]

    rms = librosa.feature.rms(y=y)[0]
    features["shimmer_local"] = np.mean(np.abs(np.diff(rms))) if len(rms) > 1 else 0
    features["energy_mean"] = np.mean(rms)
    features["energy_std"] = np.std(rms)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features["zcr_mean"] = np.mean(zcr)
    features["zcr_std"] = np.std(zcr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=2)
    features["mfcc_0_mean"] = np.mean(mfcc[0])
    features["mfcc_1_mean"] = np.mean(mfcc[1])
    features["mfcc_0_std"] = np.std(mfcc[0])
    features["mfcc_1_std"] = np.std(mfcc[1])

    features["hnr"] = 0  # simplified for stability

    return features

def voice_predict(path):
    feats = extract_voice_features(path)

    order = [
        "f0_mean","f0_std","f0_median",
        "jitter_local","jitter_rap",
        "shimmer_local","hnr",
        "mfcc_0_mean","mfcc_1_mean",
        "mfcc_0_std","mfcc_1_std",
        "energy_mean","energy_std",
        "zcr_mean","zcr_std"
    ]

    X = np.array([feats[k] for k in order]).reshape(1, -1)
    pred = audio_model.predict(X)[0]
    prob = audio_model.predict_proba(X)[0][1]

    label = "parkinson" if pred == 1 else "healthy"
    return label, prob, feats

# ======================================================
# ROUTE
# ======================================================

@app.route("/", methods=["GET", "POST"])
def index():
    handwriting_result = voice_result = mri_result = None

    if request.method == "POST":
        test = request.form.get("test_type")

        # HANDWRITING TEST
        if test == "handwriting":
            spiral = request.files.get("spiral")
            wave = request.files.get("wave")

            if spiral and wave:
                spiral_name = secure_filename(spiral.filename)
                wave_name = secure_filename(wave.filename)

                spiral_path = os.path.join(UPLOAD_FOLDER, spiral_name)
                wave_path = os.path.join(UPLOAD_FOLDER, wave_name)

                spiral.save(spiral_path)
                wave.save(wave_path)

                spiral_label, spiral_prob = handwriting_predict(
                    spiral_path, spiral_model, spiral_scaler, spiral_le
                )
                wave_label, wave_prob = handwriting_predict(
                    wave_path, wave_model, wave_scaler, wave_le
                )

                avg_prob = (spiral_prob + wave_prob) / 2
                final_label = "Likely Parkinson's" if avg_prob > 0.5 else "Likely Healthy"

                handwriting_result = {
                    "spiral_img": spiral_name,
                    "wave_img": wave_name,
                    "spiral_label": spiral_label,
                    "spiral_prob": round(spiral_prob * 100, 2),
                    "wave_label": wave_label,
                    "wave_prob": round(wave_prob * 100, 2),
                    "final": final_label,
                    "final_prob": round(avg_prob * 100, 2),
                }

        # MRI TEST
        elif test == "mri":
            mri_file = request.files.get("mri")

            if mri_file:
                mri_name = secure_filename(mri_file.filename)
                mri_path = os.path.join(UPLOAD_FOLDER, mri_name)
                mri_file.save(mri_path)

                try:
                    label, prob = mri_predict(mri_path)
                    final_label = "Likely Parkinson's" if prob > 0.5 else "Likely Healthy"

                    mri_result = {
                        "mri_img": mri_name,
                        "label": label,
                        "prob": round(prob * 100, 2),
                        "final": final_label,
                    }
                except Exception as e:
                    return f"Error processing MRI: {str(e)}", 500

        # VOICE TEST
        elif test == "voice":
            audio = request.files.get("audio")

            if audio:
                name = secure_filename(audio.filename)
                path = os.path.join(UPLOAD_FOLDER, name)
                audio.save(path)

                try:
                    label, prob, feats = voice_predict(path)

                    voice_result = {
                        "audio_file": name,
                        "label": label,
                        "prob": round(prob * 100, 2),
                        "final": "Likely Parkinson's" if prob > 0.5 else "Likely Healthy",
                        "features": {k.replace("_"," ").title(): round(float(v), 2) for k,v in feats.items()}
                    }
                except Exception as e:
                    return f"Error processing audio: {str(e)}", 500

    return render_template(
        "index.html",
        handwriting_result=handwriting_result,
        voice_result=voice_result,
        mri_result=mri_result
    )

if __name__ == "__main__":
    app.run(debug=True)