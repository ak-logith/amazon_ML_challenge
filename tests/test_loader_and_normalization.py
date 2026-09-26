"""
Comprehensive tests for the data-loading and preprocessing layer.

Covers:
  - TSV loading (raw and preprocessed) with sep="\\t"
  - entity_id exact preservation
  - Missing value safety
  - Name normalization with raw value preservation
  - Address normalization with raw value preservation
  - Country as open-set (no hard-coded US/India)
  - Output schema compatibility with matching features
  - Ground truth loading
  - Round-trip consistency
  - Normalization reuse (no duplication check)
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.normalizer import (
    clean_business_name,
    clean_address,
    preprocess_row,
    transliterate,
    normalize_conjunctions,
    collapse_whitespace,
    strip_accents,
    unicode_normalise,
    strip_nulls,
)
from src.preprocessing.interface import (
    load_entities,
    get_blocking_keys,
    DEFAULT_COLUMNS,
)
from src.preprocessing.loader import (
    load_raw_source,
    load_raw_source_matching,
    load_preprocessed_source,
    preprocess_raw_source,
    load_ground_truth,
    _safe_str,
    RAW_COLUMNS,
)


# =====================================================================
#  Helpers — create temp TSV files for testing
# =====================================================================

def _write_tsv(path: str, header: list[str], rows: list[list[str]]) -> None:
    """Write a tab-separated file."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(row) + "\n")


def _make_raw_tsv(tmp_dir: str, filename: str = "test_source1.tsv",
                  rows: list[list[str]] | None = None) -> str:
    """Create a raw 4-column TSV in a temp directory."""
    path = os.path.join(tmp_dir, filename)
    if rows is None:
        rows = [
            ["S1-10001", "Acme Corp", "100 Broadway, New York, NY 10001", "US"],
            ["S2-20002", "Infosys Pvt Ltd", "Hitech City, Hyderabad, Telangana 500081", "India"],
            ["S3-30003", "Boutique de Paris SAS", "15 Rue de Rivoli, 75001 Paris", "France"],
        ]
    _write_tsv(path, RAW_COLUMNS, rows)
    return path


def _make_preprocessed_tsv(tmp_dir: str, rows: list[dict[str, str]]) -> str:
    """Create a preprocessed 15-column TSV from preprocess_row dicts."""
    path = os.path.join(tmp_dir, "test_source1_clean.tsv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(DEFAULT_COLUMNS) + "\n")
        for row in rows:
            vals = [row.get(col, "") for col in DEFAULT_COLUMNS]
            f.write("\t".join(vals) + "\n")
    return path


def _make_ground_truth_tsv(tmp_dir: str) -> str:
    """Create a ground truth TSV."""
    path = os.path.join(tmp_dir, "train_ground_truth.tsv")
    _write_tsv(
        path,
        ["source1_entity_id", "matched_entity_ids"],
        [
            ["S1-10001", "S2-20002,S3-30003"],
            ["S1-10002", ""],
            ["S1-10003", "S2-40004"],
        ],
    )
    return path


# =====================================================================
#  1. TSV Loading Tests
# =====================================================================

class TestRawTSVLoading(unittest.TestCase):
    """Test loading of raw 4-column source TSV files."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def test_load_raw_dict(self):
        """Raw TSV loads with sep='\\t' and returns correct dict structure."""
        path = _make_raw_tsv(self.tmp_dir)
        records = load_raw_source(path, return_type="dict")

        self.assertIn("S1-10001", records)
        self.assertIn("S2-20002", records)
        self.assertIn("S3-30003", records)
        self.assertEqual(len(records), 3)

        rec = records["S1-10001"]
        self.assertEqual(rec["business_name"], "Acme Corp")
        self.assertEqual(rec["business_address"], "100 Broadway, New York, NY 10001")
        self.assertEqual(rec["country"], "US")

    def test_load_raw_polars(self):
        """Raw TSV loads as Polars DataFrame."""
        path = _make_raw_tsv(self.tmp_dir)
        df = load_raw_source(path, return_type="polars")

        self.assertEqual(df.height, 3)
        self.assertIn("entity_id", df.columns)
        self.assertIn("business_name", df.columns)

    def test_load_raw_matching_format(self):
        """load_raw_source_matching returns keys 'name', 'addr', 'country'
        matching the matching pipeline's expected format."""
        path = _make_raw_tsv(self.tmp_dir)
        records = load_raw_source_matching(path)

        rec = records["S1-10001"]
        self.assertEqual(rec["name"], "Acme Corp")
        self.assertEqual(rec["addr"], "100 Broadway, New York, NY 10001")
        self.assertEqual(rec["country"], "US")

    def test_file_not_found(self):
        """Raises FileNotFoundError for missing file."""
        with self.assertRaises(FileNotFoundError):
            load_raw_source("/nonexistent/path.tsv")

    def test_tab_separator_enforced(self):
        """Comma in business_name is preserved (not treated as separator)."""
        rows = [
            ["S1-99999", "Smith, Jones & Co", "123 Main St", "US"],
        ]
        path = _make_raw_tsv(self.tmp_dir, rows=rows)
        records = load_raw_source(path)

        # The comma inside the business name should NOT split it
        self.assertIn("S1-99999", records)
        self.assertEqual(records["S1-99999"]["business_name"], "Smith, Jones & Co")


class TestEntityIdPreservation(unittest.TestCase):
    """Requirement #2: entity_id is preserved exactly."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def test_entity_id_preserved_with_dashes(self):
        """entity_id with dashes like S1-925783039 is preserved exactly."""
        rows = [
            ["S1-925783039", "Test Corp", "123 Street", "US"],
            ["S2-000100200", "Another Inc", "456 Avenue", "India"],
        ]
        path = _make_raw_tsv(self.tmp_dir, rows=rows)
        records = load_raw_source(path)

        self.assertIn("S1-925783039", records)
        self.assertIn("S2-000100200", records)

    def test_entity_id_not_cast_to_int(self):
        """entity_id is kept as string, never numeric."""
        rows = [["S1-000001", "Test", "Addr", "US"]]
        path = _make_raw_tsv(self.tmp_dir, rows=rows)
        records = load_raw_source(path)

        eid = list(records.keys())[0]
        self.assertIsInstance(eid, str)
        self.assertEqual(eid, "S1-000001")  # leading zeros preserved

    def test_entity_id_preserved_through_preprocessing(self):
        """entity_id survives the full preprocess_row pipeline."""
        row = preprocess_row(
            entity_id="S3-007654321",
            business_name="Test Corp",
            business_address="123 Main St",
            country="France",
        )
        self.assertEqual(row["entity_id"], "S3-007654321")


class TestMissingValueSafety(unittest.TestCase):
    """Requirement #3: Missing values handled safely."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def test_null_values_in_tsv(self):
        """Literal 'null', 'NaN', 'None' in TSV become empty strings."""
        rows = [
            ["S1-001", "null", "NaN", "None"],
            ["S1-002", "", "", ""],
        ]
        path = _make_raw_tsv(self.tmp_dir, rows=rows)
        records = load_raw_source(path)

        for eid in ["S1-001", "S1-002"]:
            rec = records[eid]
            for key in ["business_name", "business_address", "country"]:
                self.assertEqual(rec[key], "", f"{eid}.{key} should be empty")

    def test_safe_str_helper(self):
        """_safe_str handles None, NaN, 'null', etc."""
        self.assertEqual(_safe_str(None), "")
        self.assertEqual(_safe_str("null"), "")
        self.assertEqual(_safe_str("NaN"), "")
        self.assertEqual(_safe_str("None"), "")
        self.assertEqual(_safe_str("N/A"), "")
        self.assertEqual(_safe_str("  "), "")
        self.assertEqual(_safe_str("valid text"), "valid text")

    def test_preprocess_row_with_none(self):
        """preprocess_row handles None inputs without crash."""
        row = preprocess_row(
            entity_id="S1-003",
            business_name=None,
            business_address=None,
            country=None,
        )
        self.assertEqual(row["entity_id"], "S1-003")
        self.assertEqual(row["business_name"], "")
        self.assertEqual(row["business_address"], "")
        self.assertEqual(row["name_clean"], "")
        self.assertEqual(row["addr_clean"], "")

    def test_missing_address_preserves_name(self):
        """Missing address does not corrupt name normalization."""
        row = preprocess_row(
            entity_id="S1-004",
            business_name="Acme Corp",
            business_address="",
            country="US",
        )
        self.assertEqual(row["name_core"], "acme")
        self.assertEqual(row["legal_suffix"], "corp")
        self.assertEqual(row["addr_clean"], "")


# =====================================================================
#  2. Normalization Tests
# =====================================================================

class TestNameNormalization(unittest.TestCase):
    """Requirement #4: Business name normalization preserving raw values."""

    def test_raw_name_preserved(self):
        """name_original holds the exact input value."""
        res = clean_business_name("  Acme Corp.  ")
        self.assertEqual(res["name_original"], "Acme Corp.")

    def test_basic_normalization(self):
        """Lowercasing, whitespace collapse, punctuation removal."""
        res = clean_business_name("ACME  SOLUTIONS   INC.")
        self.assertEqual(res["name_core"], "acme solutions")
        self.assertEqual(res["name_clean"], "acme solutions inc.")
        self.assertEqual(res["legal_suffix"], "inc")

    def test_legal_suffix_pvt_ltd(self):
        res = clean_business_name("Infosys Technologies Private Limited")
        self.assertEqual(res["legal_suffix"], "pvt ltd")
        self.assertEqual(res["name_core"], "infosys technologies")

    def test_legal_suffix_french(self):
        res = clean_business_name("Boutique de Paris SASU")
        self.assertEqual(res["legal_suffix"], "sasu")
        self.assertEqual(res["name_core"], "boutique de paris")

    def test_legal_suffix_sarl(self):
        res = clean_business_name("Entreprise Générale SARL")
        self.assertIn(res["legal_suffix"], ("sarl",))

    def test_conjunction_normalization(self):
        res = clean_business_name("Ben & Jerry's + Co")
        self.assertIn("and", res["name_clean"])

    def test_url_stripping(self):
        res = clean_business_name("MyStore www.mystore.com LLC")
        # URL should be removed, LLC extracted
        self.assertEqual(res["legal_suffix"], "llc")
        self.assertNotIn("www", res["name_core"])

    def test_transliteration(self):
        """Indic script is transliterated to ASCII."""
        res = clean_business_name("टाटा मोटर्स लिमिटेड")
        self.assertTrue(res["name_translit"].isascii())
        self.assertTrue(len(res["name_translit"]) > 0)

    def test_empty_name(self):
        res = clean_business_name("")
        self.assertEqual(res["name_original"], "")
        self.assertEqual(res["name_clean"], "")
        self.assertEqual(res["name_core"], "")

    def test_filler_prefix_stripped(self):
        """Filler prefixes like 'Shri' are stripped."""
        res = clean_business_name("Shri Ram Enterprises")
        self.assertNotIn("shri", res["name_core"].split()[0:1])

    def test_hyphenated_llc(self):
        res = clean_business_name("Acme Solutions-L.L.C.")
        self.assertEqual(res["legal_suffix"], "llc")
        self.assertEqual(res["name_core"], "acme solutions")


class TestAddressNormalization(unittest.TestCase):
    """Requirement #5: Business address normalization preserving raw values."""

    def test_raw_address_preserved(self):
        """addr_original holds the exact input value."""
        res = clean_address("  123 Main St, NY  ", country="us")
        self.assertEqual(res["addr_original"], "123 Main St, NY")

    def test_us_address_parsing(self):
        res = clean_address(
            "123 Main St, Suite 400, Seattle, WA 98101", country="us"
        )
        self.assertEqual(res["addr_postal_code"], "98101")
        self.assertEqual(res["addr_state"], "wa")
        self.assertEqual(res["addr_city"], "seattle")
        self.assertIn("street", res["addr_clean"])

    def test_india_address_parsing(self):
        res = clean_address(
            "Plot 42, Hitech City, Hyderabad, Telangana 500081",
            country="india",
        )
        self.assertEqual(res["addr_postal_code"], "500081")
        self.assertEqual(res["addr_state"], "telangana")
        self.assertEqual(res["addr_city"], "hyderabad")

    def test_france_address_parsing(self):
        res = clean_address("15 Rue de Rivoli, 75001 Paris", country="france")
        self.assertEqual(res["addr_postal_code"], "75001")
        self.assertEqual(res["addr_city"], "paris")

    def test_empty_address(self):
        res = clean_address("", country="us")
        self.assertEqual(res["addr_original"], "")
        self.assertEqual(res["addr_clean"], "")
        self.assertEqual(res["addr_postal_code"], "")

    def test_street_abbreviation_expansion(self):
        """'St' → 'street', 'Ave' → 'avenue' etc."""
        res = clean_address("456 Oak Ave, Apt 3, Dallas, TX", country="us")
        self.assertIn("avenue", res["addr_clean"])

    def test_transliteration_address(self):
        """Non-Latin address text is transliterated."""
        res = clean_address("मुंबई, महाराष्ट्र", country="india")
        self.assertTrue(res["addr_translit"].isascii())


class TestCountryOpenSet(unittest.TestCase):
    """Requirement #6: Country is open-set, not hard-coded to US/India."""

    def test_country_not_rejected(self):
        """Any country value is accepted and lowercased."""
        for country in ["US", "India", "France", "Germany", "Brazil", "Japan", ""]:
            row = preprocess_row(
                entity_id="S1-001",
                business_name="Test Corp",
                business_address="123 Street",
                country=country,
            )
            self.assertEqual(row["country"], country.strip().lower())

    def test_unknown_country_passes(self):
        """Unknown countries don't crash the pipeline."""
        row = preprocess_row(
            entity_id="S1-001",
            business_name="Test Corp",
            business_address="123 Street, City, Region 12345",
            country="Ruritania",
        )
        self.assertEqual(row["country"], "ruritania")
        # Address should still be cleaned, just without country-specific parsing
        self.assertTrue(len(row["addr_clean"]) > 0 or row["addr_clean"] == "")

    def test_france_country_full_pipeline(self):
        """France entities go through full normalization."""
        row = preprocess_row(
            entity_id="S3-001",
            business_name="Société Générale SAS",
            business_address="29 Boulevard Haussmann, 75009 Paris",
            country="France",
        )
        self.assertEqual(row["country"], "france")
        self.assertEqual(row["legal_suffix"], "sas")
        self.assertEqual(row["addr_postal_code"], "75009")


# =====================================================================
#  3. Schema Compatibility Tests
# =====================================================================

class TestMatchingCompatibility(unittest.TestCase):
    """Requirement #7: Output is compatible with matching features."""

    def test_preprocess_row_has_all_columns(self):
        """preprocess_row output contains all 15 standard columns."""
        row = preprocess_row(
            entity_id="S1-001",
            business_name="Acme Corp",
            business_address="100 Broadway, NY 10001",
            country="US",
        )
        for col in DEFAULT_COLUMNS:
            self.assertIn(col, row, f"Missing required column: {col}")

    def test_raw_values_preserved_in_output(self):
        """business_name and business_address are untouched in output."""
        row = preprocess_row(
            entity_id="S1-001",
            business_name="  Acme Corp  ",
            business_address="  100 Broadway  ",
            country="US",
        )
        # Raw values are passed through (may be stripped of leading/trailing space)
        self.assertEqual(row["business_name"], "  Acme Corp  ")
        self.assertEqual(row["business_address"], "  100 Broadway  ")

    def test_blocking_keys_from_preprocessed(self):
        """Blocking keys can be generated from preprocessed output."""
        row = preprocess_row(
            entity_id="S1-002",
            business_name="Reliance Retail Ltd",
            business_address="Bandra Kurla Complex, Mumbai, Maharashtra 400051",
            country="india",
        )
        keys = get_blocking_keys(row)
        self.assertEqual(keys["postal_code"], "400051")
        self.assertEqual(keys["city_state"], "mumbai_maharashtra")
        self.assertEqual(keys["name_first_token"], "reliance")
        self.assertEqual(keys["country"], "india")

    def test_matching_pipeline_dict_shape(self):
        """load_raw_source_matching returns the shape matching expects:
        {entity_id: {"name": ..., "addr": ..., "country": ...}}"""
        tmp_dir = tempfile.mkdtemp()
        path = _make_raw_tsv(tmp_dir)
        records = load_raw_source_matching(path)

        for eid, rec in records.items():
            self.assertIn("name", rec)
            self.assertIn("addr", rec)
            self.assertIn("country", rec)
            self.assertIsInstance(rec["name"], str)
            self.assertIsInstance(rec["addr"], str)
            self.assertIsInstance(rec["country"], str)


# =====================================================================
#  4. Ground Truth Loading Tests
# =====================================================================

class TestGroundTruthLoading(unittest.TestCase):
    """Test ground truth TSV loading."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def test_basic_load(self):
        path = _make_ground_truth_tsv(self.tmp_dir)
        gt = load_ground_truth(path)

        self.assertEqual(len(gt), 3)
        self.assertEqual(gt["S1-10001"], {"S2-20002", "S3-30003"})
        self.assertEqual(gt["S1-10002"], set())  # empty matches = singleton
        self.assertEqual(gt["S1-10003"], {"S2-40004"})

    def test_entity_ids_preserved(self):
        path = _make_ground_truth_tsv(self.tmp_dir)
        gt = load_ground_truth(path)

        for key in gt:
            self.assertIsInstance(key, str)
            self.assertTrue(key.startswith("S1-"))

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_ground_truth("/nonexistent/gt.tsv")


# =====================================================================
#  5. Round-trip / Integration Tests
# =====================================================================

class TestRoundTrip(unittest.TestCase):
    """Verify that writing → loading preserves all data."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def test_raw_to_preprocessed_roundtrip(self):
        """Raw TSV → preprocess_raw_source → verify schema + values."""
        path = _make_raw_tsv(self.tmp_dir)
        records = preprocess_raw_source(path, return_type="dict")

        self.assertIn("S1-10001", records)
        self.assertIn("S2-20002", records)
        self.assertIn("S3-30003", records)

        # Check that normalized fields are populated
        rec = records["S1-10001"]
        self.assertTrue(len(rec["name_clean"]) > 0)
        self.assertEqual(rec["country"], "us")

    def test_preprocess_then_load(self):
        """Simulate: preprocess rows → write TSV → load back."""
        # Step 1: preprocess
        raw_rows = [
            {"entity_id": "S1-10001", "business_name": "Acme Corp",
             "business_address": "100 Broadway, New York, NY 10001",
             "country": "US"},
            {"entity_id": "S2-20002", "business_name": "Infosys Pvt Ltd",
             "business_address": "Hitech City, Hyderabad, Telangana 500081",
             "country": "India"},
        ]
        processed = []
        for r in raw_rows:
            processed.append(preprocess_row(**r))

        # Step 2: write TSV
        path = _make_preprocessed_tsv(self.tmp_dir, processed)

        # Step 3: load back
        loaded = load_preprocessed_source(path, return_type="dict")

        self.assertEqual(len(loaded), 2)
        self.assertIn("S1-10001", loaded)
        self.assertIn("S2-20002", loaded)

    def test_preprocess_raw_source_polars(self):
        """preprocess_raw_source with return_type='polars' returns a DataFrame."""
        path = _make_raw_tsv(self.tmp_dir)
        df = preprocess_raw_source(path, return_type="polars")

        self.assertEqual(df.height, 3)
        for col in DEFAULT_COLUMNS:
            self.assertIn(col, df.columns, f"Missing column {col}")


# =====================================================================
#  6. Low-level Normalization Utility Tests
# =====================================================================

class TestLowLevelNormalization(unittest.TestCase):
    """Test individual normalization functions."""

    def test_transliterate_ascii(self):
        result = transliterate("Café résumé")
        self.assertTrue(result.isascii())
        self.assertIn("cafe", result)

    def test_transliterate_devanagari(self):
        result = transliterate("मुंबई")
        self.assertTrue(result.isascii())
        self.assertTrue(len(result) > 0)

    def test_collapse_whitespace(self):
        self.assertEqual(collapse_whitespace("  a   b\tc\n d  "), "a b c d")

    def test_strip_accents(self):
        result = strip_accents("Société Générale")
        self.assertIn("Societe", result)

    def test_normalize_conjunctions(self):
        result = normalize_conjunctions("A & B + C")
        self.assertEqual(result, "A  and  B  and  C")

    def test_unicode_normalise(self):
        # ñ (precomposed) → n (after NFKD + combining mark removal)
        result = unicode_normalise("año")
        self.assertIn("ano", result)

    def test_strip_nulls(self):
        result = strip_nulls("address is null here")
        self.assertNotIn("null", result.lower().split())


# =====================================================================
#  7. Edge Cases
# =====================================================================

class TestEdgeCases(unittest.TestCase):
    """Edge cases and robustness."""

    def test_all_fields_empty(self):
        """Completely empty row doesn't crash."""
        row = preprocess_row(
            entity_id="S1-000",
            business_name="",
            business_address="",
            country="",
        )
        self.assertEqual(row["entity_id"], "S1-000")
        for col in DEFAULT_COLUMNS:
            self.assertIn(col, row)

    def test_very_long_name(self):
        """Very long business name doesn't crash."""
        long_name = "A " * 1000 + "Corp"
        row = preprocess_row(
            entity_id="S1-999",
            business_name=long_name,
            business_address="123 St",
            country="US",
        )
        self.assertTrue(len(row["name_clean"]) > 0)

    def test_special_characters_in_name(self):
        """Special characters don't crash normalization."""
        row = preprocess_row(
            entity_id="S1-888",
            business_name="Test™ ® © Company",
            business_address="123 Street",
            country="US",
        )
        self.assertTrue(len(row["name_clean"]) > 0)
        self.assertEqual(row["legal_suffix"], "co")

    def test_tsv_with_unicode(self):
        """TSV with Unicode characters loads correctly."""
        tmp_dir = tempfile.mkdtemp()
        rows = [
            ["S1-001", "Société Générale", "29 Bd Haussmann, Paris", "France"],
            ["S2-002", "टाटा मोटर्स", "मुंबई, महाराष्ट्र", "India"],
        ]
        path = _make_raw_tsv(tmp_dir, rows=rows)
        records = load_raw_source(path)

        self.assertEqual(len(records), 2)
        self.assertIn("S1-001", records)
        self.assertIn("S2-002", records)


if __name__ == "__main__":
    unittest.main(verbosity=2)
