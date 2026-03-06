from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
import tempfile

from features.extract_logmel import extract_log_mel

app = Flask(__name__)

# Labels
labels = ["yes", "no", "stop", "go", "hello", "how", "are", "you",
          "he", "is", "good", "right", "down", "left", "up"]

CONFIDENCE_THRESHOLD = 0.7

# Load model
model = tf.keras.models.load_model("speech_cnn_model.h5")

@app.route("/")
def home():
    return "Speech Recognition Backend Running"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Save temp wav file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file.save(tmp.name)
        temp_path = tmp.name

    try:
        features = extract_log_mel(temp_path)
        features = features[np.newaxis, ..., np.newaxis]

        prediction = model.predict(features, verbose=0)[0]
        idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({"word": None, "confidence": confidence})

        return jsonify({
            "word": labels[idx],
            "confidence": confidence
        })

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    # IMPORTANT: avoid Windows reloader resets
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)