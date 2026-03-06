import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tensorflow as tf
from features.extract_logmel import extract_log_mel

# Load trained model
model = tf.keras.models.load_model("speech_cnn_model.h5")

labels = ["yes", "no", "stop", "go", "hello", "how", "are", "you", "he", "is", "good", "right", "down", "good", "left"]
SAMPLE_RATE = 16000
DURATION = 2  # Increased to 2 seconds

def record_audio(filename="temp.wav"):
    print("🎙 Speak now...")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='int16'
    )
    sd.wait()
    wav.write(filename, SAMPLE_RATE, audio)
    print("✅ Recording finished")

def predict_audio(filename="temp.wav"):
    features = extract_log_mel(filename)
    features = features[np.newaxis, ..., np.newaxis]

    prediction = model.predict(features)
    confidence = np.max(prediction)
    predicted_label = labels[np.argmax(prediction)]

    if confidence < 0.75:
        print("⚠️ Unclear speech, please repeat")
    else:
        print(f"📝 Recognized word: {predicted_label} (confidence: {confidence:.2f})")

if __name__ == "__main__":
    record_audio()
    predict_audio()