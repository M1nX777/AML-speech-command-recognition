import tempfile
import unittest
from pathlib import Path

import numpy as np
from project_1.data.load_preprocces import (
    load_data,
    max_timeshape,
    normalize_data,
    one_hot_encoder,
    pad_time,
)


class TestLoadPreprocces(unittest.TestCase):
    def test_load_data_reads_wav_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            yes_dir = root / "yes"
            no_dir = root / "no"
            yes_dir.mkdir()
            no_dir.mkdir()
            (yes_dir / "a.wav").write_text("dummy")
            (yes_dir / "b.wav").write_text("dummy")
            (no_dir / "c.wav").write_text("dummy")

            data = load_data(root)

            self.assertEqual(set(data), {"no", "yes"})
            self.assertEqual(len(data["yes"]), 2)
            self.assertEqual(len(data["no"]), 1)

    def test_max_timeshape_returns_longest_axis(self) -> None:
        x1 = np.zeros((5, 10))
        x2 = np.zeros((5, 20))

        self.assertEqual(max_timeshape([x1, x2]), 20)

    def test_pad_time_adds_zero_padding(self) -> None:
        x = np.ones((5, 3))
        padded = pad_time(x, 8)

        self.assertEqual(padded.shape, (8, 3))
        self.assertTrue(np.all(padded[:5] == 1))
        self.assertTrue(np.all(padded[5:] == 0))

    def test_one_hot_encoder_maps_labels(self) -> None:
        encoded, target_names = one_hot_encoder(["yes", "no", "yes"])

        self.assertEqual(encoded.shape, (3, 2))
        self.assertEqual(target_names, ["no", "yes"])
        self.assertTrue((encoded[0] == [0, 1]).all())
        self.assertTrue((encoded[1] == [1, 0]).all())

    def test_normalize_data_mfcc_returns_zero_mean_unit_std(self) -> None:
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        normalized = normalize_data(x, "mfcc")

        self.assertTrue(np.allclose(normalized.mean(axis=0), np.zeros(2)))
        self.assertTrue(np.allclose(normalized.std(axis=0), np.ones(2)))

    def test_normalize_data_mel_returns_zero_mean(self) -> None:
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        normalized = normalize_data(x, "mel")

        self.assertTrue(np.allclose(normalized.mean(axis=0), np.zeros(2)))
