import unittest
from pathlib import Path
from typing import Sequence
from unittest.mock import MagicMock, patch

import numpy as np
from keras.models import Model
from project_1.models.train_functions import (
    cross_callbacks,
    evaluate_model,
    get_predictions,
    model_CNN,
    model_MLP,
    normalize_and_pad,
    show_shape_model,
)


class TestTrainFunctions(unittest.TestCase):
    def test_normalize_and_pad_returns_numpy_array(self) -> None:
        X = [np.ones((2, 3)), np.ones((3, 3)) * 2]

        result = normalize_and_pad(X, "mfcc", 4)

        self.assertEqual(result[0].shape, (4, 3))

    def test_cross_callbacks_returns_reduce_lr_plateau(self) -> None:
        callback = cross_callbacks(5)

        self.assertEqual(callback.patience, 5)
        self.assertEqual(callback.monitor, "val_loss")
        self.assertEqual(callback.mode, "min")

        self.assertEqual(callback.monitor, "val_loss")
        self.assertEqual(callback.mode, "min")

    @patch("project_1.models.train_functions.plot_model", return_value="diagram")
    def test_show_shape_model_returns_plot_result(self, mock_plot) -> None:
        dummy_model = MagicMock()

        result = show_shape_model(dummy_model)

        mock_plot.assert_called_once_with(dummy_model, show_shapes=True)
        self.assertEqual(result, "diagram")

    def test_model_MLP_builds_classification_model(self) -> None:
        model = model_MLP()

        self.assertEqual(model.output_shape[-1], 8)
        self.assertEqual(model.input_shape[-1], 30)

    def test_model_CNN_builds_classification_model(self) -> None:
        model = model_CNN()

        self.assertEqual(model.output_shape[-1], 8)
        self.assertEqual(model.input_shape[-2], 128)

    def test_get_predictions_returns_indices(self) -> None:
        class DummyModel(Model):
            def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
                return np.array(
                    [
                        [0.05, 0.10, 0.05, 0.60, 0.05, 0.05, 0.05, 0.05],
                        [0.60, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10],
                    ]
                )

        y_test = np.array(
            [
                [0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        X_test = []  # or [np.zeros((1, n_feturs))] for real array
        predictions, labels = get_predictions(DummyModel(), X_test, y_test)
        np.testing.assert_array_equal(predictions, [3, 0])
        np.testing.assert_array_equal(labels, [3, 0])

    @patch(
        "project_1.models.train_functions.classification_report",
        return_value="report",
    )
    @patch("builtins.print")
    def test_evaluate_model_prints_classification_report(
        self,
        mock_print,
        mock_report,
    ) -> None:
        evaluate_model(
            np.argmax(
                np.array(
                    [
                        [0.05, 0.05, 0.05, 0.10, 0.05, 0.05, 0.60, 0.05],
                        [0.60, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                    ]
                ),
                axis=1,
            ),
            np.argmax(
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ),
                axis=1,
            ),
            ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"],
        )

        mock_report.assert_called_once_with(
            [6, 0],
            [6, 0],
            target_names=["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"],
            digits=4,
        )

        mock_print.assert_called_once_with("report")