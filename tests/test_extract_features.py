import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "her2pred"))
import extract_features as extraction


class FakeSlide:
    def read_region(self, location, level, size):
        return Image.new("RGB", size, (160, 90, 130))

    def close(self):
        pass


class MeanEncoder(torch.nn.Module):
    def forward(self, images):
        return images.mean(dim=(2, 3))


class ExtractFeaturesTest(unittest.TestCase):
    def test_public_options_and_underscore_aliases(self):
        current = extraction.parse_args([
            "--format", "ndpi", "--slide-dir", "slides", "--out", "features",
            "--coords-from", "reference", "--target-mpp", "0.5",
        ])
        previous = extraction.parse_args([
            "--format", "ndpi", "--slide_dir", "slides", "--save_dir", "features",
            "--coords_from", "reference", "--target_mpp", "0.5",
        ])
        self.assertEqual(vars(current), vars(previous))
        self.assertEqual(current.encoder, "resnet50")

    def test_extracts_features_coordinates_and_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            slides = root / "slides"
            slides.mkdir()
            (slides / "sample.ndpi").touch()
            output = root / "features"
            prepared = {"slide": FakeSlide(), "backend": "openslide",
                        "coords": [(0, 0), (224, 0)], "best_level": 0,
                        "read_size": 224, "offset_l0": 0}
            with patch.object(extraction, "prepare_slide", return_value=prepared), \
                 patch("slide_io.open_slide", return_value=FakeSlide()), \
                 patch.object(extraction, "build_encoder", return_value=(MeanEncoder(), 3)), \
                 patch.object(extraction, "tqdm", side_effect=lambda iterator, **kwargs: iterator), \
                 contextlib.redirect_stdout(io.StringIO()):
                extraction.process_slides(str(slides), str(output), "ndpi", 1,
                                          encoder="gigapath", workers=0)
            result = torch.load(output / "train/sample.pt", map_location="cpu", weights_only=False)
            self.assertEqual(tuple(result["features"].shape), (2, 3))
            self.assertEqual(result["features"].dtype, torch.float32)
            self.assertEqual(result["coords"].tolist(), [[0, 0], [224, 0]])
            self.assertEqual(result["label"].item(), 1)
            self.assertEqual(result["slide_id"], "sample")


if __name__ == "__main__":
    unittest.main()
