import io
from pathlib import Path
from typing import Any, Literal

import librosa
import numpy as np
from tensorflow.keras.models import load_model

__version__ = "0.1.0"
name_cnn = "Trained_model_mel-"
name_mlp = "Trained_model_mfcc-"

BASE_DIR = Path(__file__).resolve().parent
full_path = BASE_DIR / (name_mlp + __version__ + ".keras")

_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _model_filename(
    feature_type: Literal["mfcc", "mel"], variant: Literal["standard", "robust"]
) -> Path:
    base = name_mlp if feature_type == "mfcc" else name_cnn
    suffix = "-noise" if variant == "robust" else ""
    return BASE_DIR / f"{base}{suffix}{_version_}.keras"


def get_model(
    feature_type: Literal["mfcc", "mel"], variant: Literal["standard", "robust"]
) -> Any:
    key = (feature_type, variant)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    path = _model_filename(feature_type, variant)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    mdl = load_model(path)
    _MODEL_CACHE[key] = mdl
    return mdl

model: Any = load_model(full_path)

classes = ["go", "left", "no", "on", "right", "stop", "up", "yes"]
MAX_FRAMES = 87


def trasfrom_blob_to_wav(aduio_bytes: bytes):
    pass


def load_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, float]:
    return librosa.load(io.BytesIO(audio_bytes), sr=44100)


def normalize_data(x: np.ndarray, kind: Literal["mfcc", "mel"]) -> np.ndarray:
    if kind == "mfcc":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
    else:
        x = librosa.power_to_db(x, ref=np.max)
        mean = x.mean(axis=0)
        std = x.std(axis=0)

    std = np.where(std == 0, 1.0, std)  # safety check
    return (x - mean) / std


def pad_frames(x: np.ndarray, max_len: int = MAX_FRAMES) -> np.ndarray:
    if x.shape[0] >= max_len:
        return x[:max_len]
    padding = max_len - x.shape[0]
    return np.pad(x, ((0, padding), (0, 0)), mode="constant")


def extract_features(
    audio: np.ndarray, sr: float, kind: Literal["mfcc", "mel"] = "mfcc"
) -> np.ndarray:
    if kind == "mfcc":
        x = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=30)
    else:
        x = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)

    x = x.T
    x = normalize_data(x, kind)
    x = pad_frames(x, MAX_FRAMES)

    if kind == "mel":
        x = x[..., np.newaxis]

    return np.expand_dims(x, 0)


def prediction_pipeline(
    audio_bytes: bytes, feature_type: Literal["mfcc", "mel"] = "mfcc"
) -> dict[str, object]:
    audio, sr = load_audio_bytes(audio_bytes)
    features = extract_features(audio, sr, kind=feature_type)
    features = np.array(features, dtype=np.float32)
    predictions = model.predict(features, verbose=0)
    probabilities = predictions[0].tolist()
    label_index = int(np.argmax(predictions, axis=1)[0])

    return {
        "label": classes[label_index],
        "confidence": probabilities[label_index] * 100,
        "scores": probabilities,
    }
