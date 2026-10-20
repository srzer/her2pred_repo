"""TCGA Standard-B labels and cohort selection.

Standard B uses estimated ERBB2 copy number >= 2.75. Convert PanCancer Atlas
segment means with estimated_cn = 2 * 2**log2CNA. Eligibility requires a finite
copy-number estimate.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


STUDY = "coadread_tcga_pan_can_atlas_2018"
PROFILE = f"{STUDY}_log2CNA"
ERBB2 = 2064
CN_THRESHOLD = 2.75
LOG2_THRESHOLD = math.log2(CN_THRESHOLD / 2.0)
DEFAULT_MANIFEST = str(Path(__file__).with_name("data") / "tcga_standard_b.csv")
SUPPLEMENTAL_METADATA = str(Path(__file__).with_name("data") /
                            "tcga_standard_b_supplemental_neg_62.csv")
SUPPLEMENTAL_METADATA_30 = str(Path(__file__).with_name("data") /
                               "tcga_standard_b_supplemental_neg_30_20260823.csv")


def case_of(value: str) -> str | None:
    match = re.search(r"(TCGA-[0-9A-Z]{2}-[0-9A-Z]{4})", str(value).upper())
    return match.group(1) if match else None


def fetch_log2cna() -> dict[str, float]:
    url = (f"https://www.cbioportal.org/api/molecular-profiles/{PROFILE}"
           f"/molecular-data?entrezGeneId={ERBB2}"
           f"&sampleListId={STUDY}_all&projection=DETAILED")
    proc = subprocess.run(["curl", "-fsS", "--max-time", "90", url],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"cBioPortal request failed: {proc.stderr.strip()}")
    data = json.loads(proc.stdout)
    result = {}
    for item in data:
        case = case_of(item.get("sampleId", ""))
        value = item.get("value")
        if case is not None and value is not None and math.isfinite(float(value)):
            result[case] = float(value)
    return result


def _candidate_files(root: str):
    specs = [
        ("train", Path(root) / "tcga_data" / "train"),
        ("heldout", Path(root) / "tcga_data" / "heldout"),
    ]
    for cohort, directory in specs:
        if directory.is_dir():
            for path in sorted(directory.glob("*.pt")):
                yield cohort, path


def build_manifest(root: str, out_path: str = DEFAULT_MANIFEST,
                   supplemental_paths: tuple[str, ...] = (
                       SUPPLEMENTAL_METADATA, SUPPLEMENTAL_METADATA_30,
                   )) -> list[dict]:
    log2cna = fetch_log2cna()
    generated = datetime.now(timezone.utc).isoformat()
    rows = []
    for cohort, path in _candidate_files(root):
        slide_id = path.stem
        case_id = case_of(slide_id)
        log2_value = log2cna.get(case_id)
        eligible = log2_value is not None and math.isfinite(log2_value)
        estimated_cn = 2.0 * (2.0 ** log2_value) if eligible else None
        label = int(estimated_cn >= CN_THRESHOLD) if eligible else None
        rows.append({
            "slide_id": slide_id,
            "case_id": case_id or "",
            "cohort": cohort,
            "eligible": int(eligible),
            "label": "" if label is None else label,
            "estimated_erbb2_cn": "" if estimated_cn is None else f"{estimated_cn:.10g}",
            "log2_cna": "" if log2_value is None else f"{log2_value:.10g}",
            "cn_threshold": CN_THRESHOLD,
            "log2_threshold": f"{LOG2_THRESHOLD:.10g}",
            "source": f"cBioPortal:{PROFILE}",
            "generated_utc": generated,
            "feature_path": str(path),
        })

    # Supplemental batches remain physically isolated. Their versioned
    # selection metadata is merged logically into the PUMC-standard cohort.
    for supplemental_path in supplemental_paths:
        if not supplemental_path or not os.path.isfile(supplemental_path):
            continue
        with open(supplemental_path, newline="") as handle:
            for item in csv.DictReader(handle):
                slide_id = Path(item["filename"]).stem
                rows.append({
                    "slide_id": slide_id,
                    "case_id": item["case_id"],
                    "cohort": "heldout",
                    "eligible": 1,
                    "label": 0,
                    "estimated_erbb2_cn": item["estimated_erbb2_cn"],
                    "log2_cna": item["log2_cna"],
                    "cn_threshold": CN_THRESHOLD,
                    "log2_threshold": f"{LOG2_THRESHOLD:.10g}",
                    "source": f"cBioPortal:{PROFILE}",
                    "generated_utc": generated,
                    "feature_path": f"supplemental:{Path(supplemental_path).stem}",
                })

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    validate_manifest(out_path)
    return rows


def load_manifest(path: str = DEFAULT_MANIFEST, cohort: str = "full",
                  eligible_only: bool = True) -> dict[str, dict]:
    if cohort not in {"full", "train", "heldout"}:
        raise ValueError(f"unknown TCGA cohort: {cohort}")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        if cohort != "full" and row["cohort"] != cohort:
            continue
        eligible = bool(int(row["eligible"]))
        if eligible_only and not eligible:
            continue
        row["eligible"] = eligible
        row["label"] = int(row["label"]) if row["label"] != "" else None
        row["estimated_erbb2_cn"] = (
            float(row["estimated_erbb2_cn"])
            if row["estimated_erbb2_cn"] != "" else None)
        result[row["slide_id"]] = row
    return result


def validate_manifest(path: str = DEFAULT_MANIFEST,
                      expected: dict | None = None) -> dict:
    all_rows = load_manifest(path, eligible_only=False)
    eligible = [row for row in all_rows.values() if row["eligible"]]
    train = [row for row in eligible if row["cohort"] == "train"]
    heldout = [row for row in eligible if row["cohort"] == "heldout"]
    counts = {
        "candidate": len(all_rows),
        "eligible": len(eligible),
        "positive": sum(row["label"] for row in eligible),
        "train_n": len(train),
        "train_positive": sum(row["label"] for row in train),
        "heldout_n": len(heldout),
        "heldout_positive": sum(row["label"] for row in heldout),
    }
    if expected is None:
        expected = dict(candidate=280, eligible=280, positive=31,
                        train_n=70, train_positive=18,
                        heldout_n=210, heldout_positive=13)
    if counts != expected:
        raise RuntimeError(f"standard-B manifest count mismatch: {counts} != {expected}")
    for row in eligible:
        expected_label = int(row["estimated_erbb2_cn"] >= CN_THRESHOLD)
        if row["label"] != expected_label:
            raise RuntimeError(f"threshold mismatch for {row['slide_id']}")
    return counts


def apply_to_records(records: list[dict], path: str = DEFAULT_MANIFEST,
                     cohort: str = "full") -> tuple[list[dict], dict]:
    """Relabel and deduplicate TCGA records using the Standard-B manifest.

    When a slide occurs multiple times, select the last matching record in input
    order. Return the selected records and cohort counts.
    """
    labels = load_manifest(path, cohort=cohort)
    output, selected, n_tcga, n_drop, n_match = [], {}, 0, 0, 0
    for record in records:
        sid = str(record.get("slide_id", ""))
        if not sid.upper().startswith("TCGA-"):
            output.append(record)
            continue
        n_tcga += 1
        item = labels.get(sid)
        if item is None:
            n_drop += 1
            continue
        n_match += 1
        updated = dict(record)
        updated["label"] = int(item["label"])
        updated["standard_b_cn"] = float(item["estimated_erbb2_cn"])
        selected[sid] = updated
    output.extend(selected.values())
    audit = {"tcga_candidates": n_tcga, "tcga_kept": len(selected),
             "tcga_dropped_no_cn_or_wrong_cohort": n_drop,
             "tcga_duplicate_feature_records_dropped": n_match - len(selected)}
    return output, audit


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./slide_datasets_filtered")
    parser.add_argument("--out", default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    built = build_manifest(args.root, args.out)
    print(f"wrote {args.out}: {validate_manifest(args.out)}")
