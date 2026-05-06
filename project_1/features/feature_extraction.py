import librosa
import numpy as np
from sklearn.model_selection import train_test_split


def feature_extrac_mfcc(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=30)

    return mfcc


def feature_extrac_mel(audio, sr):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    return mel


def file_opener(file, feature_func):
    audio, sr = librosa.load(file, sr=None)
    results = {}

    for name, func in feature_func.items():
        results[name] = func(audio, sr)
    return results


def transpose(X):
    return [x.T for x in X]


def split_feature(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=42
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize_data(x, name):
    if name == "mfcc":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        return (x - mean) / std
    else:
        librosa.power_to_db(x, ref=np.max)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        return (x - mean) / std
