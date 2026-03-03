import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib

# -----------------------------
# Initialize Flask App
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# Load Models
# -----------------------------
cnn_model = tf.keras.models.load_model("model/brain_tumor_model.h5")
svm_model = joblib.load("model/svm_model.pkl")

# -----------------------------
# MRI Validation Function
# -----------------------------
def is_valid_mri(image):
    image_np = np.array(image)

    # If image has 3 channels (RGB)
    if len(image_np.shape) == 3:
        r = image_np[:, :, 0]
        g = image_np[:, :, 1]
        b = image_np[:, :, 2]

        # Measure how different channels are
        diff_rg = np.mean(np.abs(r - g))
        diff_gb = np.mean(np.abs(g - b))
        diff_rb = np.mean(np.abs(r - b))

        total_diff = diff_rg + diff_gb + diff_rb

        # If too colorful → not MRI
        if total_diff > 25:   # threshold (can tune)
            return False

    return True

# -----------------------------
# Home Route
# -----------------------------
@app.route("/")
def home():
    return "Brain Tumor Detection API is running!"

# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        model_type = request.form.get("model")

        if not model_type:
            return jsonify({"error": "Model type not provided"}), 400

        model_type = model_type.lower()
        print("Model type received:", model_type)

        image = Image.open(file).convert("RGB")
        # Validate MRI image (grayscale check)
        if not is_valid_mri(image):
            return jsonify({
                "prediction": "Invalid Image - Please Upload Brain MRI",
                "confidence": 0,
                "model_used": model_type.upper(),
                "time_ms": 0
    })

        start_time = time.time()

        # ---------------- CNN ----------------
        if model_type == "cnn":
            image_resized = image.resize((150, 150))
            img_array = np.array(image_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = cnn_model.predict(img_array)[0][0]

            if prediction > 0.5:
                result = "Tumor Detected"
                confidence = prediction
            else:
                result = "No Tumor"
                confidence = 1 - prediction

        # ---------------- SVM ----------------
        elif model_type == "svm":
            import cv2
            from skimage.feature import hog

            image_gray = image.convert("L")
            image_array = np.array(image_gray)

            image_array = cv2.resize(image_array, (128, 128))

            features = hog(
                image_array,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                visualize=False
            )

            features = features.reshape(1, -1)

            prediction = svm_model.predict(features)[0]
            prob = svm_model.predict_proba(features)[0][prediction]

            result = "Tumor Detected" if prediction == 1 else "No Tumor"
            confidence = prob

        else:
            return jsonify({"error": "Invalid model selection"}), 400

        end_time = time.time()
        prediction_time = round((end_time - start_time) * 1000, 2)

        return jsonify({
            "prediction": result,
            "confidence": round(float(confidence) * 100, 2),
            "model_used": model_type.upper(),
            "time_ms": prediction_time
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)