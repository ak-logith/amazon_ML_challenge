#!/usr/bin/env python3
"""
Unit and Integration tests for Candidate Generation & Blocking Engine (V4)
"""

import unittest
from pathlib import Path
from src.generate_candidates import (
    _norm,
    _translit_clean,
    _name_tokens,
    _name_prefixes,
    _addr_tokens,
    _postal_and_numeric_tokens,
    _char_ngrams,
    _enc,
    _dec,
    SUFFIXES,
    ADDR_STOPS,
    BUDGET_MIN,
    BUDGET_MAX,
    FALLBACK_TRIGGER,
)


class TestCandidateGenerationV4(unittest.TestCase):

    def test_normalization_and_diacritics(self):
        # US / English
        self.assertEqual(_norm("Walmart Inc."), "walmart inc")
        # French diacritics
        self.assertEqual(_norm("Société Générale & Cie"), "societe generale cie")
        # Whitespace and punctuation collapsing
        self.assertEqual(_norm("  A & B   Co.  "), "a b co")

    def test_transliteration_clean(self):
        # Indic transliteration to ASCII
        t_hindi = _translit_clean("राम मार्केटिंग")
        self.assertTrue(len(t_hindi) > 0)
        self.assertTrue(all(ord(c) < 128 for c in t_hindi))

        # French accents
        t_fr = _translit_clean("Électricité de France")
        self.assertIn("electricite", t_fr)

    def test_name_tokens_and_suffix_removal(self):
        # US legal suffix
        tokens = _name_tokens("Walmart Supercenter Inc.")
        self.assertIn("walmart", tokens)
        self.assertIn("supercenter", tokens)
        self.assertNotIn("inc", tokens)

        # French legal suffix
        tokens_fr = _name_tokens("Carrefour Banque SA")
        self.assertIn("carrefour", tokens_fr)
        self.assertIn("banque", tokens_fr)
        self.assertNotIn("sa", tokens_fr)

        # India legal suffix
        tokens_in = _name_tokens("Reliance Retail Pvt Ltd")
        self.assertIn("reliance", tokens_in)
        self.assertIn("retail", tokens_in)
        self.assertNotIn("pvt", tokens_in)
        self.assertNotIn("ltd", tokens_in)

    def test_name_tokens_script_resilience(self):
        # Transliterated tokens are added alongside original tokens
        tokens = _name_tokens("संजय ट्रेडर्स")
        self.assertTrue(len(tokens) > 0)
        # Should contain ascii transliterated tokens like 'snjy' or 'ttreddrs'
        self.assertTrue(any(t.isascii() and len(t) >= 3 for t in tokens))
        self.assertTrue(any("snjy" in t.lower() for t in tokens))

    def test_name_prefixes(self):
        prefs = _name_prefixes("Microsoft Technologies Corporation", length=5)
        self.assertIn("micro", prefs)
        self.assertIn("techn", prefs)

    def test_address_tokens_and_stopword_removal(self):
        # US address
        addr = "123 Main Street, Suite 400, New York, NY"
        tokens = _addr_tokens(addr)
        self.assertIn("main", tokens)
        self.assertIn("york", tokens)
        self.assertNotIn("street", tokens)  # in ADDR_STOPS

        # India address
        addr_in = "Plot 42, Sector 18, Near Fortis Hospital, Noida"
        tokens_in = _addr_tokens(addr_in)
        self.assertIn("fortis", tokens_in)
        self.assertIn("hospital", tokens_in)
        self.assertIn("noida", tokens_in)
        self.assertNotIn("near", tokens_in)  # in ADDR_STOPS
        self.assertNotIn("sector", tokens_in)  # in ADDR_STOPS

        # France address
        addr_fr = "15 Rue de la Paix, Cedex 02, Paris"
        tokens_fr = _addr_tokens(addr_fr)
        self.assertIn("paix", tokens_fr)
        self.assertIn("paris", tokens_fr)
        self.assertNotIn("rue", tokens_fr)
        self.assertNotIn("cedex", tokens_fr)

    def test_postal_and_numeric_tokens(self):
        addr = "Plot No. 53/1, 2nd Floor, 560001 Bangalore"
        nums = _postal_and_numeric_tokens(addr)
        self.assertIn("53", nums)
        self.assertIn("2nd", nums)
        self.assertIn("560001", nums)

    def test_char_ngrams(self):
        ngrams = _char_ngrams("Walmart", 3)
        self.assertIn("wal", ngrams)
        self.assertIn("alm", ngrams)
        self.assertIn("mar", ngrams)
        self.assertIn("art", ngrams)

    def test_entity_id_integer_encoding(self):
        # S2 encoding -> positive int
        code_s2 = _enc("S2-123456")
        self.assertEqual(code_s2, 123456)
        self.assertEqual(_dec(code_s2), "S2-123456")

        # S3 encoding -> negative int
        code_s3 = _enc("S3-789012")
        self.assertEqual(code_s3, -789012)
        self.assertEqual(_dec(code_s3), "S3-789012")


class TestV4BlockingEngineIntegration(unittest.TestCase):
    """End-to-end integration tests for V4 blocking architecture."""

    def setUp(self):
        import tempfile
        self.temp_dir = tempfile.TemporaryDirectory()
        self.d = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_removal_of_early_quota_bottleneck(self):
        """
        Verify that Channel A/B candidates beyond 180 are NOT choked early,
        and survive into the final candidate set if below BUDGET_MAX.
        """
        import os
        from io import StringIO
        from src.generate_candidates import _build_indexes, _generate_country_v4

        s2_path = os.path.join(self.d, "s2_quota.tsv")
        s3_path = os.path.join(self.d, "s3_quota.tsv")
        s1_path = os.path.join(self.d, "s1_quota.tsv")

        # Create 250 S2 entities all matching name token 'supercorp'
        with open(s2_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for i in range(1, 251):
                f.write(f"S2-{i}\tSupercorp Division {i}\tAddress {i}\tUS\n")

        with open(s3_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S3-1\tSupercorp Labs\tAddress S3\tUS\n")

        with open(s1_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-1\tSupercorp Headquarters\tAddress S1\tUS\n")

        ni, pi, ai, pos_i, c3_i, rare_tokens = _build_indexes(s2_path, s3_path, "US")
        out_buf = StringIO()

        # In V3, this was choked at 180. In V4 with budget_max=600, ALL 251 should be retained!
        stats = _generate_country_v4(
            s1_path=s1_path,
            country="US",
            ni=ni, pi=pi, ai=ai, pos_i=pos_i, c3_i=c3_i, rare_tokens=rare_tokens,
            gt_arr=None, out_fh=out_buf,
            budget_max=600,
        )

        cands = set(out_buf.getvalue().strip().split("\t")[1].split(","))
        self.assertEqual(len(cands), 251)
        self.assertIn("S3-1", cands)
        self.assertIn("S2-250", cands)

    def test_global_priority_scoring_prevents_address_displacement(self):
        """
        Verify that rare/specific name candidates score higher than
        dense common-address noise candidates.
        """
        import os
        from io import StringIO
        from src.generate_candidates import _build_indexes, _generate_country_v4

        s2_path = os.path.join(self.d, "s2_prio.tsv")
        s3_path = os.path.join(self.d, "s3_prio.tsv")
        s1_path = os.path.join(self.d, "s1_prio.tsv")

        # 100 S2 entities with dense address 'Cyber Hub' but generic names
        with open(s2_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for i in range(1, 101):
                f.write(f"S2-{i}\tRandom Firm {i}\tCyber Hub Building {i}\tIndia\n")

        # 1 S3 entity with rare name 'QuantumX' but different address
        with open(s3_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S3-999\tQuantumX Innovations\tRemote Village\tIndia\n")

        with open(s1_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-1\tQuantumX\tCyber Hub Building 1\tIndia\n")

        ni, pi, ai, pos_i, c3_i, rare_tokens = _build_indexes(s2_path, s3_path, "India")
        out_buf = StringIO()

        _generate_country_v4(
            s1_path=s1_path, country="India",
            ni=ni, pi=pi, ai=ai, pos_i=pos_i, c3_i=c3_i, rare_tokens=rare_tokens,
            gt_arr=None, out_fh=out_buf,
            budget_max=50,  # Force truncation to 50
        )

        cands = set(out_buf.getvalue().strip().split("\t")[1].split(","))
        # QuantumX S3-999 must NOT be displaced by address noise
        self.assertIn("S3-999", cands)

    def test_true_character_3gram_fallback(self):
        """
        Verify that low-candidate entities (<40) trigger the true character 3-gram
        fallback to recover typo / slight spelling variation matches.
        """
        import os
        from io import StringIO
        from src.generate_candidates import _build_indexes, _generate_country_v4

        s2_path = os.path.join(self.d, "s2_3g.tsv")
        s3_path = os.path.join(self.d, "s3_3g.tsv")
        s1_path = os.path.join(self.d, "s1_3g.tsv")

        # S2 has a name with middle typo: 'BioPharmaceutics' vs S1 'BioFarmaceutics'
        # They share 0 tokens of len >= 3, and 5-char prefixes differ ('bioph' vs 'biofa')
        with open(s2_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S2-555\tBioPharmaceutics\tUnknown Road\tUS\n")

        with open(s3_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S3-1\tCompletely Unrelated Corp\tOther St\tUS\n")

        with open(s1_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-1\tBioFarmaceutics\tDifferent Ave\tUS\n")

        ni, pi, ai, pos_i, c3_i, rare_tokens = _build_indexes(s2_path, s3_path, "US")
        out_buf = StringIO()

        stats = _generate_country_v4(
            s1_path=s1_path, country="US",
            ni=ni, pi=pi, ai=ai, pos_i=pos_i, c3_i=c3_i, rare_tokens=rare_tokens,
            gt_arr=None, out_fh=out_buf,
            fallback_trigger=40,
        )

        cands = set(out_buf.getvalue().strip().split("\t")[1].split(","))
        self.assertEqual(stats["fallback_triggered"], 1)
        self.assertIn("S2-555", cands)

    def test_source_balancing_without_artificial_inflation(self):
        """
        Verify that when clamped to BUDGET_MAX, neither source starves the other,
        and when one source has few candidates, no fake candidates are manufactured.
        """
        import os
        from io import StringIO
        from src.generate_candidates import _build_indexes, _generate_country_v4

        s2_path = os.path.join(self.d, "s2_bal.tsv")
        s3_path = os.path.join(self.d, "s3_bal.tsv")
        s1_path = os.path.join(self.d, "s1_bal.tsv")

        # S2 has 100 candidates, S3 has only 5 candidates
        with open(s2_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for i in range(1, 101):
                f.write(f"S2-{i}\tGlobal Logistics {i}\tIndustrial Area\tUS\n")

        with open(s3_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for i in range(1, 6):
                f.write(f"S3-{i}\tGlobal Logistics {i}\tIndustrial Area\tUS\n")

        with open(s1_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-1\tGlobal Logistics\tIndustrial Area\tUS\n")

        ni, pi, ai, pos_i, c3_i, rare_tokens = _build_indexes(s2_path, s3_path, "US")
        out_buf = StringIO()

        # Clamp to budget_max = 20 (target 10 per source)
        _generate_country_v4(
            s1_path=s1_path, country="US",
            ni=ni, pi=pi, ai=ai, pos_i=pos_i, c3_i=c3_i, rare_tokens=rare_tokens,
            gt_arr=None, out_fh=out_buf,
            budget_max=20,
        )

        cands = set(out_buf.getvalue().strip().split("\t")[1].split(","))
        self.assertEqual(len(cands), 20)
        s3_count = sum(1 for c in cands if c.startswith("S3-"))
        s2_count = sum(1 for c in cands if c.startswith("S2-"))

        # All 5 S3 candidates must be kept (not starved by 100 S2s, but not inflated to 10)
        self.assertEqual(s3_count, 5)
        # S2 takes the remaining 15 slots
        self.assertEqual(s2_count, 15)

    def test_open_set_country_handling(self):
        """Verify open-set country handling for unseen countries (France, Germany)."""
        import os
        from io import StringIO
        from src.generate_candidates import _build_indexes, _generate_country_v4

        s2_path = os.path.join(self.d, "s2_fr.tsv")
        s3_path = os.path.join(self.d, "s3_fr.tsv")
        s1_path = os.path.join(self.d, "s1_fr.tsv")

        with open(s2_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S2-1\tBoulangerie Pierre SARL\t10 Rue Rivoli\tFrance\n")

        with open(s3_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S3-1\tPierre Boulangerie SAS\t10 Rue Rivoli\tFrance\n")

        with open(s1_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-1\tPierre Boulangerie\t10 Rue Rivoli\tFrance\n")

        ni, pi, ai, pos_i, c3_i, rare_tokens = _build_indexes(s2_path, s3_path, "France")
        out_buf = StringIO()

        _generate_country_v4(
            s1_path=s1_path, country="France",
            ni=ni, pi=pi, ai=ai, pos_i=pos_i, c3_i=c3_i, rare_tokens=rare_tokens,
            gt_arr=None, out_fh=out_buf,
        )

        cands = set(out_buf.getvalue().strip().split("\t")[1].split(","))
        self.assertIn("S2-1", cands)
        self.assertIn("S3-1", cands)

    def test_support_zero_one_and_many_candidates(self):
        """Verify 0, 1, and many candidates are output properly."""
        import os
        from io import StringIO
        from src.generate_candidates import _build_indexes, _generate_country_v4

        s2_path = os.path.join(self.d, "s2_card.tsv")
        s3_path = os.path.join(self.d, "s3_card.tsv")
        s1_path = os.path.join(self.d, "s1_card.tsv")

        with open(s2_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S2-1\tAlpha Apex Solutions\t100 Main St\tUS\n")
            for i in range(10, 25):
                f.write(f"S2-{i}\tStarbucks Coffee {i}\tSeattle\tUS\n")

        with open(s3_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S3-1\tBeta Tech\t200 Oak St\tUS\n")

        with open(s1_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-1\tAlpha Apex\tMain St\tUS\n")          # 1 match (S2-1)
            f.write("S1-2\tStarbucks Coffee\tSeattle\tUS\n")    # 15 matches (S2-10..24)
            f.write("S1-3\tZzz Unmatched Name\tUnknown\tUS\n")  # 0 matches

        ni, pi, ai, pos_i, c3_i, rare_tokens = _build_indexes(s2_path, s3_path, "US")
        out_buf = StringIO()

        _generate_country_v4(
            s1_path=s1_path, country="US",
            ni=ni, pi=pi, ai=ai, pos_i=pos_i, c3_i=c3_i, rare_tokens=rare_tokens,
            gt_arr=None, out_fh=out_buf,
        )

        lines = dict(l.split("\t") for l in out_buf.getvalue().split("\n") if l)
        # S1-1: 1 match
        cands_s1_1 = [c for c in lines["S1-1"].split(",") if c]
        self.assertEqual(len(cands_s1_1), 1)
        self.assertEqual(cands_s1_1[0], "S2-1")

        # S1-2: 15 matches
        cands_s1_2 = [c for c in lines["S1-2"].split(",") if c]
        self.assertEqual(len(cands_s1_2), 15)

        # S1-3: 0 matches
        self.assertEqual(lines["S1-3"], "")


class TestDatasetPathResolution(unittest.TestCase):
    """Unit tests for portable dataset path resolution and file validation."""

    def setUp(self):
        import tempfile
        self.temp_dir = tempfile.TemporaryDirectory()
        self.td = Path(self.temp_dir.name).resolve()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_resolve_dataset_dir_direct(self):
        from src.generate_candidates import resolve_dataset_dir
        (self.td / "train").mkdir()
        (self.td / "test").mkdir()
        resolved = resolve_dataset_dir(str(self.td))
        self.assertEqual(resolved, self.td)

    def test_resolve_dataset_dir_nested(self):
        from src.generate_candidates import resolve_dataset_dir
        nested = self.td / "dataset"
        nested.mkdir()
        (nested / "train").mkdir()
        (nested / "test").mkdir()
        resolved = resolve_dataset_dir(str(self.td))
        self.assertEqual(resolved, nested)

    def test_get_dataset_paths_validate_success(self):
        from src.generate_candidates import get_dataset_paths
        train_dir = self.td / "train"
        train_dir.mkdir()
        for f in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv"):
            (train_dir / f).touch()
        paths = get_dataset_paths(self.td, "validate")
        self.assertTrue(paths["s1"].exists())
        self.assertTrue(paths["gt"].exists())

    def test_get_dataset_paths_generate_success(self):
        from src.generate_candidates import get_dataset_paths
        test_dir = self.td / "test"
        test_dir.mkdir()
        for f in ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv"):
            (test_dir / f).touch()
        paths = get_dataset_paths(self.td, "generate")
        self.assertTrue(paths["s1"].exists())
        self.assertTrue(paths["s2"].exists())
        self.assertTrue(paths["s3"].exists())

    def test_get_dataset_paths_missing_files_raises(self):
        from src.generate_candidates import get_dataset_paths
        train_dir = self.td / "train"
        train_dir.mkdir()
        (train_dir / "train_source1.tsv").touch()
        with self.assertRaises(FileNotFoundError) as ctx:
            get_dataset_paths(self.td, "validate")
        self.assertIn("Missing required dataset files", str(ctx.exception))
        self.assertIn("train_source2.tsv", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
