from pathlib import Path
from typing import Callable, Sequence

import librosa
import numpy as np
from sklearn.model_selection import train_test_split


def feature_extrac_mfcc(audio: np.ndarray, sr: float) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=30)
    return mfcc


def feature_extrac_mel(audio: np.ndarray, sr: float) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    return mel


def file_opener(
    file: Path,
    feature_func: dict[str, Callable[[np.ndarray, float], np.ndarray]],
) -> dict[str, np.ndarray]:
    audio, sr = librosa.load(file, sr=44100)
    results = {}

    for name, func in feature_func.items():
        results[name] = func(audio, sr)
    return results


def transpose(
    X: Sequence[np.ndarray] | np.ndarray,
) -> Sequence[np.ndarray] | np.ndarray:
    return [x.T for x in X]


def split_feature(
    X: Sequence[np.ndarray],
    y: np.ndarray,
) -> tuple[Sequence[np.ndarray], Sequence[np.ndarray], np.ndarray, np.ndarray]:
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, stratify=y, test_size=0.1, random_state=42
    )
    return X_train_val, X_test, y_train_val, y_test

def add_gaussian_noise(
    X: Sequence[np.ndarray] | np.ndarray,
    noise_factor: float = 0.005,
    random_state: int | None = None,
) -> np.ndarray | Sequence[np.ndarray]:
    rng = np.random.default_rng(random_state)

    if isinstance(X, np.ndarray):
        noise = rng.standard_normal(X.shape) * noise_factor
        return X + noise

    noisy_X = []
    for x in X:
        noisy_X.append(x + rng.standard_normal(x.shape) * noise_factor)
    return noisy_X
