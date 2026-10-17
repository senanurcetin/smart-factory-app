"""No-network source identity tests for UCI, NASA and candidate Kaggle mirrors."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from analysis.source_registry import (
    DEFAULT_MANIFEST,
    SourceRegistryError,
    load_manifest,
    source_status,
    validate_manifest,
    verify_local_archive,
)

ROOT = Path(__file__).resolve().parents[1]


class SourceRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.canonical = load_manifest()

    def setUp(self):
        self.manifest = copy.deepcopy(self.canonical)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_source_manifest_is_explicitly_unverified(self):
        self.assertEqual(len(self.manifest["datasets"]), 2)
        for result in source_status(self.manifest):
            self.assertFalse(result["archive_checksum_pinned"])
            self.assertFalse(result["kaggle_version_pinned"])
            self.assertFalse(result["license_reviewed"])
            self.assertFalse(result["cloud_ingestion_authorized"])

    def test_official_training_fingerprints_match_committed_model_cards(self):
        locations = {
            "uci_ai4i_2020": "ai4i-case-study",
            "nasa_cmapss_fd001": "cmapss-rul-case-study",
        }
        for item in self.manifest["datasets"]:
            with self.subTest(dataset=item["id"]):
                source = ROOT / "docs" / "data" / locations[item["id"]] / "model-selection.json"
                model_card = json.loads(source.read_text(encoding="utf-8"))["model_card"]
                self.assertEqual(
                    item["benchmark_dataframe_sha256"],
                    model_card["dataset"]["sha256"],
                )
                self.assertEqual(
                    item["expected_training_rows"], model_card["dataset"]["rows"]
                )

    def test_unpinned_archive_is_rejected_without_reading_bytes(self):
        file = self.dir / "dataset.zip"
        file.write_bytes(b"synthetic-offline-test-data")
        with self.assertRaisesRegex(SourceRegistryError, "must be pinned"):
            verify_local_archive(self.manifest, "uci_ai4i_2020", file)

    def test_explicit_file_hash_verifies_without_claiming_training_parity(self):
        contents = b"safe fake archive bytes"
        file = self.dir / "dataset.zip"
        file.write_bytes(contents)
        digest = hashlib.sha256(contents).hexdigest()
        self.manifest["datasets"][0]["archive_sha256"] = digest
        result = verify_local_archive(self.manifest, "uci_ai4i_2020", file)
        self.assertEqual(result["verified_archive_sha256"], digest)
        self.assertEqual(result["archive_bytes"], len(contents))
        self.assertIsNone(result["matches_benchmark_dataframe_hash"])
        self.assertEqual(result["source_provenance"], "local_bytes_verified_only")

    def test_tampered_archive_fails_closed(self):
        file = self.dir / "dataset.zip"
        file.write_bytes(b"fake-v1")
        self.manifest["datasets"][0]["archive_sha256"] = hashlib.sha256(b"fake-v1").hexdigest()
        file.write_bytes(b"fake-v2")
        with self.assertRaisesRegex(SourceRegistryError, "mismatch"):
            verify_local_archive(self.manifest, "uci_ai4i_2020", file)

    def test_missing_and_symlink_archive_rejected(self):
        self.manifest["datasets"][0]["archive_sha256"] = "a" * 64
        with self.assertRaises(SourceRegistryError):
            verify_local_archive(self.manifest, "uci_ai4i_2020", self.dir / "missing.zip")
        target = self.dir / "real.zip"
        target.write_bytes(b"placeholder")
        link = self.dir / "symlink.zip"
        try:
            link.symlink_to(target)
        except OSError:
            self.skipTest("Host does not support symlinks")
        with self.assertRaisesRegex(SourceRegistryError, "non-symlink"):
            verify_local_archive(self.manifest, "uci_ai4i_2020", link)

    def test_duplicate_or_unknown_source_id_rejected(self):
        self.manifest["datasets"][1]["id"] = "uci_ai4i_2020"
        with self.assertRaisesRegex(SourceRegistryError, "Duplicate"):
            validate_manifest(self.manifest)
        self.manifest["datasets"][1]["id"] = "untrusted_dataset"
        with self.assertRaisesRegex(SourceRegistryError, "unknown"):
            validate_manifest(self.manifest)

    def test_rejects_misleading_source_role(self):
        self.manifest["datasets"][0]["data_role"] = "rul_regression"
        with self.assertRaisesRegex(SourceRegistryError, "role mismatch"):
            validate_manifest(self.manifest)

    def test_rejects_untrusted_plain_http_and_credential_urls(self):
        for url in ("http://example.com/test.csv", "https://user:pass@example.com/a"):
            manifest = copy.deepcopy(self.manifest)
            manifest["datasets"][0]["official_url"] = url
            with self.subTest(url=url):
                with self.assertRaises(SourceRegistryError):
                    validate_manifest(manifest)

    def test_unapproved_https_source_host_is_rejected(self):
        self.manifest["datasets"][0]["official_url"] = (
            "https://untrusted.example.com/ai4i.zip"
        )
        with self.assertRaisesRegex(SourceRegistryError, "pinned HTTPS host"):
            validate_manifest(self.manifest)

    def test_malformed_https_source_url_is_rejected(self):
        self.manifest["datasets"][0]["official_url"] = "https://[invalid"
        with self.assertRaisesRegex(SourceRegistryError, "Malformed"):
            validate_manifest(self.manifest)

    def test_rejects_unversioned_or_invalid_kaggle_mirror(self):
        self.manifest["datasets"][0]["kaggle_version"] = 3
        with self.assertRaisesRegex(SourceRegistryError, "requires"):
            validate_manifest(self.manifest)
        self.manifest["datasets"][0]["kaggle_dataset"] = "evil/../../path"
        with self.assertRaises(SourceRegistryError):
            validate_manifest(self.manifest)

    def test_valid_kaggle_mirror_requires_explicit_version_to_be_marked_pinned(self):
        self.manifest["datasets"][0]["kaggle_dataset"] = "example-owner/ai4i-mirror"
        self.assertFalse(source_status(self.manifest)[0]["kaggle_version_pinned"])
        self.manifest["datasets"][0]["kaggle_version"] = 2
        self.assertTrue(source_status(self.manifest)[0]["kaggle_version_pinned"])

    def test_rejects_invalid_digest_and_license_types(self):
        self.manifest["datasets"][0]["archive_sha256"] = "not_sha"
        with self.assertRaisesRegex(SourceRegistryError, "SHA-256"):
            validate_manifest(self.manifest)
        self.manifest["datasets"][0]["archive_sha256"] = None
        self.manifest["datasets"][0]["license_reviewed"] = "yes"
        with self.assertRaisesRegex(SourceRegistryError, "boolean"):
            validate_manifest(self.manifest)

    def test_rejects_duplicate_json_keys_in_manifest(self):
        path = self.dir / "manifest.json"
        path.write_text('{"schema_version":1,"schema_version":2}', encoding="utf-8")
        with self.assertRaisesRegex(SourceRegistryError, "Duplicate JSON key"):
            load_manifest(path)

    def test_manifest_symlink_is_rejected(self):
        target = self.dir / "real.json"
        target.write_text(DEFAULT_MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
        link = self.dir / "link.json"
        try:
            link.symlink_to(target)
        except OSError:
            self.skipTest("Host does not support symlinks")
        with self.assertRaisesRegex(SourceRegistryError, "non-symlink"):
            load_manifest(link)

    def test_unexpected_fields_cannot_claim_verified_source(self):
        self.manifest["datasets"][0]["remote_verified"] = True
        with self.assertRaisesRegex(SourceRegistryError, "entry fields"):
            validate_manifest(self.manifest)


if __name__ == "__main__":
    unittest.main()
