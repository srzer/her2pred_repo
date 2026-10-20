import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "her2pred"))
import extract_features
import run_ndpi_cv_dann
import run_zeroshot_dann
from stain_norm import MacenkoNormalizer, build_normalizer


class SupportedExperimentsTest(unittest.TestCase):
    def test_training_modes_reject_removed_experiments(self):
        for module in (run_ndpi_cv_dann, run_zeroshot_dann):
            for mode in ("source", "appearregfull"):
                with self.subTest(module=module.__name__, mode=mode):
                    with patch.object(sys, "argv", ["train", "--domain_mode", mode]):
                        self.assertEqual(module.parse_args().domain_mode, mode)
            for mode in ("site", "colour", "colourreg", "appearreg", "appearregfullsc", "tileappearreg"):
                with self.subTest(module=module.__name__, removed=mode):
                    with patch.object(sys, "argv", ["train", "--domain_mode", mode]), \
                         contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit):
                            module.parse_args()

    def test_extraction_supports_none_and_macenko(self):
        self.assertIsNone(build_normalizer("none"))
        self.assertIsInstance(build_normalizer("macenko"), MacenkoNormalizer)
        for method in ("rgb", "vahadane", "reinhard", "tilezca", "slidelut"):
            with self.subTest(method=method):
                with self.assertRaises(ValueError):
                    build_normalizer(method)
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        extract_features.parse_args([
                            "--format", "ndpi", "--slide-dir", "slides",
                            "--stain-norm", method,
                        ])

    def test_released_cache_targets_keep_their_standardization(self):
        targets = np.arange(120, dtype=np.float32).reshape(2, 60)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "appearance.npz"
            np.savez(path, X=targets, sid=np.array(["slide_a", "slide_b"]),
                     prestandardized=np.array(True),
                     selected_sid=np.array(["slide_a", "slide_b", "slide_c"]),
                     missing_sid=np.array(["slide_c"]),
                     expected_selected_n=np.array(3), expected_target_n=np.array(2))
            with patch.object(run_zeroshot_dann, "APPEAR_FULL_CACHE", str(path)), \
                 patch.dict(run_zeroshot_dann.STATE):
                vectors, dimension = run_zeroshot_dann.build_appearance_vectors_full()
                self.assertEqual(dimension, 60)
                np.testing.assert_array_equal(vectors["slidea"], targets[0])
                np.testing.assert_array_equal(vectors["slideb"], targets[1])
                self.assertEqual(run_zeroshot_dann.STATE["appearance_missing_ids"], {"slidec"})


if __name__ == "__main__":
    unittest.main()
