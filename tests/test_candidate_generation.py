#!/usr/bin/env python3
"""
Unit tests for Candidate Generation & Blocking Engine (V3)
"""

import unittest
from src.generate_candidates import (
    _norm,
    _name_tokens,
    _name_prefixes,
    _addr_tokens,
    _postal_and_numeric_tokens,
    _char_ngrams,
    _enc,
    _dec,
    SUFFIXES,
    ADDR_STOPS,
)


class TestCandidateGenerationV3(unittest.TestCase):

    def test_normalization_and_diacritics(self):
        # US / English
        self.assertEqual(_norm("Walmart Inc."), "walmart inc")
        # French diacritics
        self.assertEqual(_norm("Société Générale & Cie"), "societe generale cie")
        # Hindi / non-latin unicode normalization
        self.assertTrue(len(_norm("राम मार्केटिंग")) > 0)
        # Whitespace and punctuation collapsing
        self.assertEqual(_norm("  A & B   Co.  "), "a b co")

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


class TestV3BlockingEngineIntegration(unittest.TestCase):
    """End-to-end integration and behavioral tests for V3 multi-channel blocking."""

    def setUp(self):
        import io
        import tempfile
        self.temp_dir = tempfile.TemporaryDirectory()
        self.d = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_open_set_country_and_multichannel_blocking(self):
        """Verify open-set country handling (France, Germany) and S2/S3 candidate generation."""
        import os
        from io import StringIO
        from src.generate_candidates import _build_indexes, _generate_country_v3

        s2_path = os.path.join(self.d, "s2.tsv")
        s3_path = os.path.join(self.d, "s3.tsv")
        s1_path = os.path.join(self.d, "s1.tsv")

        # Create France data with both S2 and S3
        with open(s2_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S2-101\tBoulangerie Patisserie Pierre\t14 Rue de Rivoli, Paris 75001\tFrance\n")
            f.write("S2-102\tPharmacie de la Mairie\t8 Avenue des Champs, Lyon 69001\tFrance\n")

        with open(s3_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S3-201\tPierre Boulangerie Artisanal\t14 Rue Rivoli\tFrance\n")
            f.write("S3-202\tUnrelated Corp\t99 Other Street\tFrance\n")

        with open(s1_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-1\tPierre Boulangerie\t14 Rue de Rivoli, Paris\tFrance\n")
            f.write("S1-2\tPharmacie Mairie\tLyon\tFrance\n")
            f.write("S1-3\tCompletely Unknown Unique Business\tNowhere Land\tFrance\n")

        ni, pi, ai, pos_i, fpi, rare_tokens = _build_indexes(s2_path, s3_path, "France")
        out_buf = StringIO()

        stats = _generate_country_v3(
            s1_path=s1_path,
            country="France",
            ni=ni,
            pi=pi,
            ai=ai,
            pos_i=pos_i,
            fpi=fpi,
            rare_tokens=rare_tokens,
            gt_arr=None,
            out_fh=out_buf,
        )

        output_lines = [l for l in out_buf.getvalue().split("\n") if l]
        self.assertEqual(len(output_lines), 3)

        # S1-1 should find both S2-101 and S3-201 (name 'pierre' / addr 'rivoli')
        s1_1_line = [l for l in output_lines if l.startswith("S1-1\t")][0]
        cands_1 = s1_1_line.split("\t")[1].split(",")
        self.assertIn("S2-101", cands_1)
        self.assertIn("S3-201", cands_1)

        # S1-2 should find S2-102
        s1_2_line = [l for l in output_lines if l.startswith("S1-2\t")][0]
        cands_2 = s1_2_line.split("\t")[1].split(",")
        self.assertIn("S2-102", cands_2)

        # S1-3 has zero overlap
        s1_3_line = [l for l in output_lines if l.startswith("S1-3\t")][0]
        self.assertEqual(s1_3_line.split("\t")[1], "")

    def test_quota_protection_prevents_address_displacement(self):
        """
        Verify that a dense address token matching hundreds of entities
        does NOT displace a rare, high-specificity name match.
        """
        import os
        from io import StringIO
        from src.generate_candidates import (
            _build_indexes,
            _generate_country_v3,
            CHANNEL_B_QUOTA,
        )

        s2_path = os.path.join(self.d, "s2_quota.tsv")
        s3_path = os.path.join(self.d, "s3_quota.tsv")
        s1_path = os.path.join(self.d, "s1_quota.tsv")

        # 300 S2 entities sharing the dense address 'Cyber City', but completely different names
        with open(s2_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for i in range(1, 301):
                f.write(f"S2-{i}\tRandom Firm {i}\tCyber City Tower B\tIndia\n")

        # 1 S3 entity sharing the rare name 'QuantumXTech' with S1, but different address
        with open(s3_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S3-999\tQuantumXTech Labs\tSomewhere Remote\tIndia\n")

        # S1 has rare name 'QuantumXTech' AND dense address 'Cyber City'
        with open(s1_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-1\tQuantumXTech\tCyber City Tower B\tIndia\n")

        ni, pi, ai, pos_i, fpi, rare_tokens = _build_indexes(s2_path, s3_path, "India")
        out_buf = StringIO()

        stats = _generate_country_v3(
            s1_path=s1_path,
            country="India",
            ni=ni,
            pi=pi,
            ai=ai,
            pos_i=pos_i,
            fpi=fpi,
            rare_tokens=rare_tokens,
            gt_arr=None,
            out_fh=out_buf,
        )

        output_line = out_buf.getvalue().strip()
        cands = set(output_line.split("\t")[1].split(","))

        # The rare name candidate S3-999 MUST be retained thanks to Channel A quota protection
        self.assertIn("S3-999", cands)
        # The address channel must be capped by its quota, not overflowing the entire candidate set
        s2_cands = [c for c in cands if c.startswith("S2-")]
        self.assertLessEqual(len(s2_cands), CHANNEL_B_QUOTA)

    def test_support_zero_one_and_many_candidates(self):
        """Verify the pipeline supports 0, 1, and >10 candidates cleanly."""
        import os
        from io import StringIO
        from src.generate_candidates import _build_indexes, _generate_country_v3

        s2_path = os.path.join(self.d, "s2_card.tsv")
        s3_path = os.path.join(self.d, "s3_card.tsv")
        s1_path = os.path.join(self.d, "s1_card.tsv")

        with open(s2_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            # 1 match for single
            f.write("S2-1\tAlpha Apex Solutions\t100 Main St\tUS\n")
            # 15 matches for many
            for i in range(10, 25):
                f.write(f"S2-{i}\tStarbucks Coffee {i}\tSeattle\tUS\n")

        with open(s3_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S3-1\tBeta Tech\t200 Oak St\tUS\n")

        with open(s1_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-1\tAlpha Apex\tMain St\tUS\n")          # exactly 1 match (S2-1)
            f.write("S1-2\tStarbucks Coffee\tSeattle\tUS\n")    # 15 matches (S2-10..24)
            f.write("S1-3\tZzz Unmatched Name\tUnknown\tUS\n")  # 0 matches

        ni, pi, ai, pos_i, fpi, rare_tokens = _build_indexes(s2_path, s3_path, "US")
        out_buf = StringIO()

        stats = _generate_country_v3(
            s1_path=s1_path,
            country="US",
            ni=ni,
            pi=pi,
            ai=ai,
            pos_i=pos_i,
            fpi=fpi,
            rare_tokens=rare_tokens,
            gt_arr=None,
            out_fh=out_buf,
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


if __name__ == "__main__":
    unittest.main()

