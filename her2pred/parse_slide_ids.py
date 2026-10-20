"""Slide-name normalization and optional case grouping.

Configure raw data locations through environment variables.
"""
from __future__ import annotations

import os
import re


RAW_DIRS = {
    key: value for key, value in {
        "positive": os.environ.get("HER2PRED_RAW_POSITIVE"),
        "additional_positive": os.environ.get("HER2PRED_RAW_ADDITIONAL_POSITIVE"),
        "negative": os.environ.get("HER2PRED_RAW_NEGATIVE"),
    }.items() if value
}


def norm_key(name):
    return re.sub(r"[\s\-_]+", "", os.path.splitext(str(name))[0]).lower()


def parse(filename, group=None):
    """Return a conservative case/block key parsed from a slide filename."""
    stem = os.path.splitext(os.path.basename(str(filename)))[0]
    specimen = next((item for item in re.findall(r"\d{7}", stem)
                     if item[0] in "12"), None)
    blocks = re.findall(r"(?<![A-Za-z0-9])([A-D])[\s\-_]?(\d{1,2})(?![0-9])", stem)
    block = f"{blocks[0][0]}{blocks[0][1]}" if blocks else ""
    case = specimen or re.sub(r"[-_](?:HE|H&E)$", "", stem, flags=re.IGNORECASE)
    return {"case": case, "block": block, "accession": None, "scheme": "generic"}

