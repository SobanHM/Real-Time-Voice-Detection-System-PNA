import os
import joblib
import numpy as np
import json

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from utils.feature_extraction import extract_features

app = Flask(__name__, 
            template_folder="../frontend/templates",
            static_folder="../frontend/static")

CORS(app)  # enable cross-origin

# Load trained artifacts
MODEL_PATH = "model/speaker_recognition_model.pkl"
ENCODER_PATH = "model/label_encoder.pkl"

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    with open("metrics/metrics.json") as f:
        metrics = json.load(f)
    return render_template("dashboard.html", metrics=metrics)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = "temp.wav"
    file.save(file_path)

    try:
        features = extract_features(file_path)  # extract MFCC
        probs = model.predict_proba(features)  # get probabilities
        pred_index = np.argmax(probs)
        confidence = float(np.max(probs))

        speaker = encoder.inverse_transform([pred_index])[0]

        os.remove(file_path)

        return jsonify({
            "speaker": speaker,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
