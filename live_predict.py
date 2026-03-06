import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tensorflow as tf
import time

from features.extract_logmel import extract_log_mel


# --------------------------------
# Load trained model
# --------------------------------
model = tf.keras.models.load_model("speech_cnn_model.h5")


# --------------------------------
# Configuration
# --------------------------------
labels = [
    "yes","no","stop","hello","how","are","you",
    "he","is","good","right","down","left","up"
]

SAMPLE_RATE = 16000
CHUNK_DURATION = 2.0      # seconds per recording
CONFIDENCE_THRESHOLD = 0.7
SILENCE_THRESHOLD = 0.005

sentence_queue = []


# --------------------------------
# Helper functions
# --------------------------------
def rms_energy(signal):
    signal = signal.astype(np.float32) / 32768.0
    return np.sqrt(np.mean(signal ** 2))


def record_chunk():
    frames = int(CHUNK_DURATION * SAMPLE_RATE)

    audio = sd.rec(
        frames,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    return audio.flatten()


def predict_word(audio):

    audio_norm = audio.astype(np.float32) / 32768.0
    audio_norm = np.clip(audio_norm, -1.0, 1.0)

    # Silence detection
    energy = rms_energy(audio)
    if energy < SILENCE_THRESHOLD:
        return None, 0.0

    # Save temporary wav file
    wav.write("temp.wav", SAMPLE_RATE, (audio_norm * 32768).astype(np.int16))

    # Extract features
    features = extract_log_mel("temp.wav")[np.newaxis, ...]

    # Prediction
    pred = model.predict(features, verbose=0)

    word = labels[np.argmax(pred)]
    confidence = np.max(pred)

    return word, confidence


# --------------------------------
# Main system
# --------------------------------
def main():

    print("\n🎙 Speech Recognition System Started")
    print("Speak one word at a time.")
    print("Press Ctrl+C to stop.\n")

    try:

        while True:

            input("🟢 Press ENTER and speak a word...")

            print("🎙 Recording...")
            audio = record_chunk()

            word, conf = predict_word(audio)

            if word and conf >= CONFIDENCE_THRESHOLD:

                print(f"🔊 Recognized: {word}  (confidence {conf:.2f})")

                accept = input("Accept this word? (Enter = yes / n = no): ").strip().lower()

                if accept == "" or accept == "y":

                    sentence_queue.append(word)

                    print("\n📝 Current Sentence:")
                    print(" ".join(sentence_queue))
                    print()

                else:
                    print("❌ Word discarded\n")

            else:
                print("⚠️ No clear word detected. Try again.\n")

            time.sleep(0.3)

    except KeyboardInterrupt:

        print("\n🛑 Session stopped")

        final_sentence = " ".join(sentence_queue)

        print("\n📝 Final Sentence:")
        print(final_sentence if final_sentence else "No words accepted")


# --------------------------------
# Run program
# --------------------------------
if __name__ == "__main__":
    main()