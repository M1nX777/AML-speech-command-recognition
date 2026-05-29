from pathlib import Path
from typing import Sequence

import librosa
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def load_data(path: Path) -> dict[str, list[Path]]:
    d_fold = sorted([f for f in path.iterdir() if f.is_dir()])
    data = {fldr.name: sorted(fldr.glob("*.wav")) for fldr in d_fold}
    return data


def max_timeshape(name: Sequence[np.ndarray]) -> int:
    holder = []
    for lists in name:
        holder.append(lists.shape[1])
    return max(holder)


def pad_time(x: np.ndarray, max_len: int) -> np.ndarray:
    T, F = x.shape

    pad_width = max_len - T
    return np.pad(x, ((0, pad_width), (0, 0)), mode="constant")


def one_hot_encoder(y: Sequence[str] | np.ndarray) -> tuple[np.ndarray, list[str]]:
    enc = OneHotEncoder(sparse_output=False)
    y = np.array(y).reshape(-1, 1)
    return (enc.fit_transform(y), enc.categories_[0].tolist())


def normalize_data(x: np.ndarray, name: str) -> np.ndarray:
    if name == "mfcc":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
    else:
        x = librosa.power_to_db(x, ref=np.max)
        mean = x.mean(axis=0)
        std = x.std(axis=0)

    std = np.where(std == 0, 1.0, std)
    return (x - mean) / std
