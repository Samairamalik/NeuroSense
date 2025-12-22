import cv2
import numpy as np
import joblib
from skimage import feature

#Generating numeric data from image
def quantify_image(image):
    return feature.hog(
        image,
        orientations=12,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm="L2-Hys",
    )

# Generic prediction function
def predict_image(image_path, model_path, scaler_path, encoder_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(encoder_path)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    features = quantify_image(image)
    features = scaler.transform([features])

    # Prediction + probability
    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0]

    label = le.inverse_transform([pred])[0]
    parkinson_index = list(le.classes_).index("parkinson")
    parkinson_prob = probs[parkinson_index]

    return label, parkinson_prob


#Test

spiral_label, spiral_prob = predict_image(
    image_path="handwriting/spiral.png",
    model_path="models/parkinsons_spiral_model.pkl",
    scaler_path="models/scaler_spiral.pkl",
    encoder_path="models/label_encoder_spiral.pkl",
)

wave_label, wave_prob = predict_image(
    image_path="handwriting/wave.png",
    model_path="models/parkinsons_wave_model.pkl",
    scaler_path="models/scaler_wave.pkl",
    encoder_path="models/label_encoder_wave.pkl",
)

#Final result

avg_prob = (spiral_prob + wave_prob) / 2

print("Spiral Prediction:", spiral_label)
print(f"Spiral Parkinson's likelihood: {(spiral_prob)*100:.3f}%\n")

print("Wave Prediction:", wave_label)
print(f"Wave Parkinson's likelihood: {(wave_prob)*100:.3f}%\n")

print("Final Assessment")
print("----------------")
print(f"Average Parkinson's likelihood: {avg_prob*100:.3f}%")

if avg_prob > 0.5:
    print("Conclusion: Likely Parkinson's")
else:
    print("Conclusion: Likely Healthy")
