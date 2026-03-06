import os
import numpy as np
import tensorflow as tf
from features.extract_logmel import extract_log_mel

# ------------------------------
# Labels used for evaluation (14 labels)
# ------------------------------
LABELS = ["yes", "no", "stop", "go", "hello", "how", "are", "you",
          "he", "is", "good", "right", "down", "left", "up"]

# ------------------------------
# Load trained model
# ------------------------------
model = tf.keras.models.load_model("speech_cnn_model.h5", compile=False)
NUM_CLASSES = model.output_shape[-1]
print(f"✅ Model output classes: {NUM_CLASSES}")

# ------------------------------
# Dataset path (same as used for training)
# ------------------------------
DATASET_DIR = "dataset"

correct = 0
total = 0

for label in LABELS:
    folder = os.path.join(DATASET_DIR, label)
    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        # Extract log-Mel features
        features = extract_log_mel(path)
        features = features[np.newaxis, ..., np.newaxis]

        # Predict
        pred = model.predict(features, verbose=0)

        # Ignore last output unit (15th) if exists
        pred = pred[:, :len(LABELS)]  # take only first 14 outputs

        predicted_label = LABELS[np.argmax(pred)]

        if predicted_label == label:
            correct += 1
        total += 1

accuracy = (correct / total) * 100
print(f"✅ Word-level Accuracy: {accuracy:.2f}%")
