import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tensorflow as tf
import os

from features.extract_logmel import extract_log_mel


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("speech_cnn_model.h5")

model = load_model()


# -----------------------------
# Config
# -----------------------------
labels = [
    "yes","no","stop","hello","how","are","you",
    "he","is","good","right","down","left","up"
]

SAMPLE_RATE = 16000
CHUNK_DURATION = 2.0
CONFIDENCE_THRESHOLD = 0.7
SILENCE_THRESHOLD = 0.005


# -----------------------------
# Helper functions
# -----------------------------
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

    energy = rms_energy(audio)

    if energy < SILENCE_THRESHOLD:
        return None, 0.0

    wav.write("temp.wav", SAMPLE_RATE, (audio_norm * 32768).astype(np.int16))

    features = extract_log_mel("temp.wav")

    os.remove("temp.wav")

    # add batch + channel dimension for CNN
    features = features[np.newaxis, ..., np.newaxis]

    pred = model.predict(features, verbose=0)

    word = labels[np.argmax(pred)]
    confidence = np.max(pred)

    return word, confidence


# -----------------------------
# Streamlit UI
# -----------------------------
def speech_interface():

    st.header("🎙 Speech Recognition")

    if "sentence_queue" not in st.session_state:
        st.session_state.sentence_queue = []

    if "last_word" not in st.session_state:
        st.session_state.last_word = None

    if "last_conf" not in st.session_state:
        st.session_state.last_conf = 0.0


    st.subheader("📝 Current Sentence")

    sentence = " ".join(st.session_state.sentence_queue)

    st.text_area("Sentence", value=sentence, height=70, disabled=True)


    if st.button("🎤 Record Speech"):

        st.info("Recording... Speak now")

        audio = record_chunk()

        word, conf = predict_word(audio)

        st.session_state.last_word = word
        st.session_state.last_conf = conf


    if st.session_state.last_word:

        st.success(
            f"Predicted word: {st.session_state.last_word} (confidence {st.session_state.last_conf:.2f})"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Accept"):
                st.session_state.sentence_queue.append(st.session_state.last_word)
                st.session_state.last_word = None

        with col2:
            if st.button("❌ Reject"):
                st.session_state.last_word = None


    if st.button("🧹 Clear Sentence"):
        st.session_state.sentence_queue = []