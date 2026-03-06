import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tensorflow as tf
import time
from features.extract_logmel import extract_log_mel

# ------------------------------
# Load trained model
# ------------------------------
model = tf.keras.models.load_model("speech_cnn_model.h5")

# ------------------------------
# Configuration
# ------------------------------
labels = ["yes", "no", "stop", "go", "hello", "how", "are", "you", "he", "is", "good", "right", "down", "left"]
SAMPLE_RATE = 16000
CHUNK_DURATION = 2.0          # seconds per audio frame
CONFIDENCE_THRESHOLD = 0.7
SILENCE_THRESHOLD = 0.002      # RMS threshold for silence
SILENCE_CHUNKS_FOR_EMPTY = 2

sentence_queue = []
last_word = None
silence_count = 0

# ------------------------------
# Helper functions
# ------------------------------
def rms_energy(signal):
    signal = signal.astype(np.float32) / 32768.0
    return np.sqrt(np.mean(signal**2))

def record_chunk():
    frames = int(CHUNK_DURATION * SAMPLE_RATE)
    audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    return audio.flatten()

def predict_word(audio):
    audio_norm = audio.astype(np.float32) / 32768.0
    audio_norm = np.clip(audio_norm, -1.0, 1.0)

    energy = rms_energy(audio)
    print(f"🔊 Energy: {energy:.6f}")

    if energy < SILENCE_THRESHOLD:
        return None, 0.0

    wav.write("temp.wav", SAMPLE_RATE, (audio_norm * 32768).astype(np.int16))
    features = extract_log_mel("temp.wav")[np.newaxis, ..., np.newaxis]

    pred = model.predict(features, verbose=0)
    return labels[np.argmax(pred)], np.max(pred)

# ------------------------------
# Main loop
# ------------------------------
def main():
    global last_word, silence_count
    print("🎙 Continuous live recording started (Ctrl+C to stop)\n")

    try:
        while True:
            audio = record_chunk()
            word, conf = predict_word(audio)

            if word and conf >= CONFIDENCE_THRESHOLD:
                silence_count = 0
                if word != last_word:
                    sentence_queue.append(word)
                    print(f"✅ Recognized word: {word} ({conf:.2f})")
                    last_word = word

            else:
                silence_count += 1
                if silence_count >= SILENCE_CHUNKS_FOR_EMPTY:
                    if last_word != "_":
                        sentence_queue.append("_")
                        print("⚠️ Unclear speech added as '_'")
                        last_word = "_"
                    silence_count = 0

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n🛑 Recording stopped")
        final_sentence = " ".join(sentence_queue)
        print("📝 Final sentence:")
        print("👉", final_sentence if final_sentence else "No clear speech")

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    main()
