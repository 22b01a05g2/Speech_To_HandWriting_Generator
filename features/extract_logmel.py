import librosa
import numpy as np

def extract_log_mel(file_path, max_len=64):
    """
    Extract log-mel spectrogram + delta + delta-delta
    Output shape: (40, 64, 3)
    """

    # Load audio
    audio, sr = librosa.load(file_path, sr=16000)

    # Remove silence
    audio, _ = librosa.effects.trim(audio, top_db=20)

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=40,
        n_fft=512,
        hop_length=160
    )

    log_mel = librosa.power_to_db(mel)

    # Determine safe delta width
    frames = log_mel.shape[1]

    width = min(9, frames)

    # ensure width is odd
    if width % 2 == 0:
        width -= 1

    # ensure width >=3
    if width < 3:
        width = 3

    # delta features
    delta = librosa.feature.delta(log_mel, width=width)
    delta2 = librosa.feature.delta(log_mel, order=2, width=width)

    # stack features
    features = np.stack([log_mel, delta, delta2], axis=-1)

    # pad / trim
    if features.shape[1] < max_len:
        pad = max_len - features.shape[1]
        features = np.pad(features, ((0,0),(0,pad),(0,0)))
    else:
        features = features[:, :max_len, :]

    return features