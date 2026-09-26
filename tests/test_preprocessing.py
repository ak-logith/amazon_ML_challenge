"""
Unit and integration tests for data preprocessing and normalization.
"""

import sys
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
)
from src.preprocessing.interface import get_blocking_keys, DEFAULT_COLUMNS


class TestBusinessNameCleaning(unittest.TestCase):
    def test_legal_suffix_extraction(self):
        res = clean_business_name("Infosys Technologies Private Limited")
        self.assertEqual(res["legal_suffix"], "pvt ltd")
        self.assertEqual(res["name_core"], "infosys technologies")

    def test_hyphenated_llc(self):
        res = clean_business_name("Acme Solutions-L.L.C.")
        self.assertEqual(res["legal_suffix"], "llc")
        self.assertEqual(res["name_core"], "acme solutions")

    def test_french_legal_suffix(self):
        res = clean_business_name("Boutique de Paris SASU")
        self.assertEqual(res["legal_suffix"], "sasu")
        self.assertEqual(res["name_core"], "boutique de paris")

    def test_conjunction_normalization(self):
        norm = normalize_conjunctions("Ben & Jerry's + Co")
        self.assertIn("and", norm)
        # Ensure 'et' inside words is preserved
        self.assertEqual(normalize_conjunctions("street market"), "street market")

    def test_indic_transliteration(self):
        # Hindi name transliteration
        res = clean_business_name("टाटा मोटर्स लिमिटेड")
        self.assertTrue(len(res["name_translit"]) > 0)
        self.assertTrue(res["name_translit"].isascii())


class TestAddressCleaning(unittest.TestCase):
    def test_us_address_parsing(self):
        res = clean_address("123 Main St, Suite 400, Seattle, WA 98101", country="us")
        self.assertEqual(res["addr_postal_code"], "98101")
        self.assertEqual(res["addr_state"], "wa")
        self.assertEqual(res["addr_city"], "seattle")
        self.assertIn("street", res["addr_clean"])

    def test_india_address_parsing(self):
        res = clean_address("Plot 42, Hitech City, Hyderabad, Telangana 500081", country="india")
        self.assertEqual(res["addr_postal_code"], "500081")
        self.assertEqual(res["addr_state"], "telangana")
        self.assertEqual(res["addr_city"], "hyderabad")

    def test_france_address_parsing(self):
        res = clean_address("15 Rue de Rivoli, 75001 Paris", country="france")
        self.assertEqual(res["addr_postal_code"], "75001")
        self.assertEqual(res["addr_city"], "paris")


class TestInterfaceAndCompatibility(unittest.TestCase):
    def test_preprocess_row_schema(self):
        row = preprocess_row(
            entity_id="S1-001",
            business_name="Acme Corp",
            business_address="100 Broadway, NY 10001",
            country="US",
        )
        for col in DEFAULT_COLUMNS:
            self.assertIn(col, row, f"Missing required column {col}")

        self.assertEqual(row["business_name"], "Acme Corp")
        self.assertEqual(row["business_address"], "100 Broadway, NY 10001")
        self.assertEqual(row["country"], "us")

    def test_get_blocking_keys(self):
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


if __name__ == "__main__":
    unittest.main()
