#!/usr/bin/env python3
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "her2pred"))
from dataset_paths import pt_leaf_dirs


class PtLeafDirsTest(unittest.TestCase):
    def test_flat_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "random-id.pt").touch()
            self.assertEqual(pt_leaf_dirs(str(root)), [str(root)])

    def test_legacy_split_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir()
            (root / "valid").mkdir()
            self.assertEqual(
                pt_leaf_dirs(str(root)),
                [str(root / "train"), str(root / "valid")],
            )

    def test_mixed_layout_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.pt").touch()
            (root / "train").mkdir()
            with self.assertRaisesRegex(RuntimeError, "ambiguous PT layout"):
                pt_leaf_dirs(str(root))


if __name__ == "__main__":
    unittest.main()
