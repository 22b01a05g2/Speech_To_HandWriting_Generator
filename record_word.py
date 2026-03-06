import os
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

SAMPLE_RATE = 16000
DURATION = 1.5  # seconds per recording
CHANNELS = 1

def record_word(word, num_samples=20):
    folder = f"dataset/{word}"
    os.makedirs(folder, exist_ok=True)

    # Determine starting index for file naming
    existing_files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    start_index = len(existing_files) + 1

    print(f"🎙 Recording {num_samples} samples for '{word}'...")

    for i in range(num_samples):
        print(f"\nSample {i+1}/{num_samples}: Speak now!")
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
        sd.wait()

        max_amp = np.max(np.abs(audio))
        if max_amp < 500:  # very low volume
            print("⚠️ Warning: very low volume, try again")
            continue

        filename = f"{folder}/{word}_{start_index + i:02d}.wav"
        wav.write(filename, SAMPLE_RATE, audio)
        print(f"✅ Saved: {filename} (Max amplitude: {max_amp})")

    print(f"\n🎉 Finished recording {num_samples} samples for '{word}'")

if __name__ == "__main__":
    word = input("Enter the word you want to record: ").strip()
    num_samples = int(input("Enter number of samples to record: "))
    record_word(word, num_samples)
