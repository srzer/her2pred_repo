#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "her2pred"))
from appearance_cache_scope import validate_coverage


def record(source, slide_id):
    return {"source": source, "slide_id": slide_id}


class AppearanceCacheScopeTest(unittest.TestCase):
    def setUp(self):
        self.train = [record("ndpi", "n1"), record("svs", "s1"),
                      record("kfb", "k1"), record("tcga", "t1")]
        self.valid = [record("ndpi", "n2"), record("svs", "s2")]
        self.dom = {"n1": 0}  # n2 is held out; external records have no target.

    def test_final_zero_and_few_modes_need_only_ndpi_targets(self):
        validate_coverage(self.train, self.valid, self.dom,
                          {"n1", "n2"}, set(), external_site_only=True)

    def test_ndpi_missing_target_must_be_declared(self):
        with self.assertRaisesRegex(RuntimeError, "missing-target pattern"):
            validate_coverage(self.train, self.valid, {},
                              {"n1", "n2"}, set(), external_site_only=True)
        validate_coverage(self.train, self.valid, {},
                          {"n1", "n2"}, {"n1"}, external_site_only=True)

    def test_external_targets_remain_required_when_appearance_is_enabled(self):
        with self.assertRaisesRegex(RuntimeError, "cohort mismatch"):
            validate_coverage(self.train, self.valid, self.dom,
                              {"n1", "n2"}, set(), external_site_only=False)

    def test_legacy_cache_rejects_missing_active_targets(self):
        with self.assertRaisesRegex(RuntimeError, "coverage failure"):
            validate_coverage(self.train, self.valid, {}, None, None,
                              external_site_only=True)


if __name__ == "__main__":
    unittest.main()
