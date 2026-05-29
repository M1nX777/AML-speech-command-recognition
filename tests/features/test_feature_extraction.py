import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from project_1.features.feature_extraction import (
    feature_extrac_mel,
    feature_extrac_mfcc,
    file_opener,
    split_feature,
    transpose,
)


class TestFeatureExtraction(unittest.TestCase):
    @patch(
        "project_1.features.feature_extraction.librosa.feature.mfcc",
        return_value=np.ones((30, 5)),
    )
    def test_feature_extrac_mfcc_calls_librosa(self, mock_mfcc) -> None:
        audio = np.zeros(100)
        result = feature_extrac_mfcc(audio, 44100)

        mock_mfcc.assert_called_once_with(y=audio, sr=44100, n_mfcc=30)
        np.testing.assert_array_equal(result, np.ones((30, 5)))

    @patch(
        "project_1.features.feature_extraction.librosa.feature.melspectrogram",
        return_value=np.ones((128, 4)),
    )
    def test_feature_extrac_mel_calls_librosa(self, mock_mel) -> None:
        audio = np.zeros(100)
        result = feature_extrac_mel(audio, 44100)

        mock_mel.assert_called_once_with(y=audio, sr=44100, n_mels=128)
        np.testing.assert_array_equal(result, np.ones((128, 4)))

    @patch(
        "project_1.features.feature_extraction.librosa.load",
        return_value=(np.array([0.1, 0.2, 0.3]), 44100),
    )
    def test_file_opener_returns_features(self, mock_load) -> None:
        dummy_file = Path("dummy.wav")
        features = {
            "mfcc": lambda audio, sr: "mfcc",
            "mel": lambda audio, sr: "mel",
        }

        result = file_opener(dummy_file, features)

        mock_load.assert_called_once_with(dummy_file, sr=44100)
        self.assertEqual(result, {"mfcc": "mfcc", "mel": "mel"})

    def test_transpose_swaps_axes(self) -> None:
        x = [np.zeros((2, 3)), np.ones((4, 2))]
        result = transpose(x)

        self.assertEqual(result[0].shape, (3, 2))
        self.assertEqual(result[1].shape, (2, 4))

    @patch("project_1.features.feature_extraction.train_test_split")
    def test_split_feature_calls_train_test_split(self, mock_split) -> None:
        mock_split.return_value = (
            "X_train_val",
            "X_test",
            "y_train_val",
            "y_test",
        )

        X = [np.array([["x1", "x2", "x3", "x4"]])]
        y = np.array([[0, 0, 1, 1]])

        output = split_feature(X, y)

        self.assertEqual(
            output,
            ("X_train_val", "X_test", "y_train_val", "y_test"),
        )
        mock_split.assert_called_once_with(
            X, y, stratify=y, test_size=0.1, random_state=42
        )
