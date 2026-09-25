"""
Text normalization utilities for business entity resolution.
Handles business name and address cleaning, abbreviation expansion,
legal suffix removal, and script normalization.
"""

import re
import unicodedata
from functools import lru_cache

# ── Legal suffixes to strip from business names ──────────────────────────
# Ordered longest-first so "Private Limited" matches before "Limited"
LEGAL_SUFFIXES = [
    # Indian
    "private limited", "pvt limited", "pvt ltd", "pvt. ltd.", "pvt. ltd",
    "pvt ltd.", "private ltd", "private ltd.", "priv ltd",
    # French
    "société par actions simplifiée", "société à responsabilité limitée",
    "société anonyme", "société civile immobilière",
    "sarl", "sasu", "sas", "sa", "sci", "eurl", "snc",
    # English
    "limited liability partnership", "limited liability company",
    "limited partnership", "limited", "incorporated",
    "corporation", "company",
    "l.l.c.", "llc", "llp", "l.l.p.", "l.p.", "lp",
    "ltd.", "ltd", "inc.", "inc", "corp.", "corp",
    "co.", "co", "p.c.", "pc", "p.a.", "pa",
    "pllc", "p.l.l.c.",
    # Indian misc
    "nidhi limited", "opc private limited",
]

# ── US address abbreviation expansion ────────────────────────────────────
US_ADDR_ABBREVS = {
    "st": "street", "st.": "street",
    "ave": "avenue", "ave.": "avenue",
    "rd": "road", "rd.": "road",
    "dr": "drive", "dr.": "drive",
    "blvd": "boulevard", "blvd.": "boulevard",
    "ct": "court", "ct.": "court",
    "ln": "lane", "ln.": "lane",
    "cir": "circle", "cir.": "circle",
    "pl": "place", "pl.": "place",
    "pkwy": "parkway", "pkwy.": "parkway",
    "hwy": "highway", "hwy.": "highway",
    "apt": "apartment", "apt.": "apartment",
    "ste": "suite", "ste.": "suite",
    "fl": "floor", "fl.": "floor",
    "bldg": "building", "bldg.": "building",
    "mt": "mount", "mt.": "mount",
    "ft": "fort", "ft.": "fort",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest",
}

# ── Indian state abbreviation expansion ──────────────────────────────────
INDIAN_STATE_ABBREVS = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh",
    "as": "assam", "br": "bihar", "cg": "chhattisgarh",
    "ga": "goa", "gj": "gujarat", "hr": "haryana",
    "hp": "himachal pradesh", "jk": "jammu and kashmir",
    "jh": "jharkhand", "ka": "karnataka", "kl": "kerala",
    "mp": "madhya pradesh", "mh": "maharashtra",
    "mn": "manipur", "ml": "meghalaya", "mz": "mizoram",
    "nl": "nagaland", "od": "odisha", "or": "orissa",
    "pb": "punjab", "rj": "rajasthan", "sk": "sikkim",
    "tn": "tamil nadu", "ts": "telangana", "tr": "tripura",
    "up": "uttar pradesh", "uk": "uttarakhand",
    "wb": "west bengal", "dl": "delhi",
    "py": "puducherry", "ch": "chandigarh",
}

# ── Compiled regex patterns ──────────────────────────────────────────────
_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_RE_NUMBERS = re.compile(r"\b\d+\b")
_RE_UNIT_PREFIX = re.compile(
    r"\b(?:unit|apt|apartment|suite|ste|fl|floor|bldg|building|room|rm)\s*#?\s*\w+",
    re.IGNORECASE,
)
_RE_ZIPCODE_US = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
_RE_PINCODE_IN = re.compile(r"\b(\d{6})\b")
_RE_POSTAL_FR = re.compile(r"\b(\d{5})\b")


def normalize_unicode(text: str) -> str:
    """Normalize unicode to NFC form and strip accents for comparison."""
    if not text:
        return ""
    # NFC normalize first
    text = unicodedata.normalize("NFC", text)
    return text


def strip_accents(text: str) -> str:
    """Remove diacritical marks (accents) from text."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


@lru_cache(maxsize=500_000)
def normalize_name(name: str) -> str:
    """
    Normalize a business name for comparison:
    1. Unicode NFC normalize
    2. Lowercase
    3. Replace '&' with 'and'
    4. Strip punctuation
    5. Remove legal suffixes
    6. Collapse whitespace
    """
    if not name:
        return ""

    text = normalize_unicode(name)
    text = text.lower()
    text = text.replace("&", " and ").replace("+", " and ")
    # Remove URLs/websites
    text = re.sub(r"https?://\S+|www\.\S+|\S+\.(com|org|net|co\.in)\b", "", text)
    # Strip punctuation but keep spaces and alphanumeric
    text = _RE_PUNCTUATION.sub(" ", text)
    # Remove legal suffixes (longest first match)
    for suffix in LEGAL_SUFFIXES:
        if text.rstrip().endswith(suffix):
            text = text.rstrip()[: -len(suffix)]
            break  # only strip the first (outermost) match
    # Remove common prefixes
    for prefix in ["the ", "shri ", "shree ", "sri ", "smt "]:
        if text.lstrip().startswith(prefix):
            text = text.lstrip()[len(prefix):]
            break
    # Collapse whitespace
    text = _RE_MULTI_SPACE.sub(" ", text).strip()
    return text


@lru_cache(maxsize=500_000)
def normalize_name_aggressive(name: str) -> str:
    """
    More aggressive normalization — also strips accents and
    removes all non-ASCII (handles Devanagari/Tamil script names).
    Use for cross-script matching attempts.
    """
    text = normalize_name(name)
    text = strip_accents(text)
    # Keep only ASCII alphanumeric + spaces
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = _RE_MULTI_SPACE.sub(" ", text).strip()
    return text


@lru_cache(maxsize=500_000)
def normalize_address(address: str) -> str:
    """
    Normalize a business address for comparison:
    1. Unicode NFC normalize
    2. Lowercase
    3. Strip punctuation
    4. Collapse whitespace
    """
    if not address:
        return ""

    text = normalize_unicode(address)
    text = text.lower()
    text = text.replace("&", " and ")
    # Strip punctuation
    text = _RE_PUNCTUATION.sub(" ", text)
    # Collapse whitespace
    text = _RE_MULTI_SPACE.sub(" ", text).strip()
    return text


def expand_address_abbreviations(address: str, country: str = "") -> str:
    """Expand common address abbreviations based on country."""
    if not address:
        return ""
    tokens = address.split()
    expanded = []
    country_lower = country.lower() if country else ""

    for token in tokens:
        lower_t = token.lower().rstrip(".")
        if country_lower in ("us", "france", ""):
            if lower_t in US_ADDR_ABBREVS:
                expanded.append(US_ADDR_ABBREVS[lower_t])
                continue
        if country_lower in ("india", ""):
            if lower_t in INDIAN_STATE_ABBREVS:
                expanded.append(INDIAN_STATE_ABBREVS[lower_t])
                continue
        expanded.append(token)
    return " ".join(expanded)


def extract_numbers(text: str) -> set:
    """Extract all numeric tokens from text."""
    if not text:
        return set()
    return set(_RE_NUMBERS.findall(text))


def extract_postal_code(address: str, country: str = "") -> str:
    """Extract postal/ZIP/PIN code from address."""
    if not address:
        return ""
    country_lower = country.lower() if country else ""

    if country_lower == "us":
        m = _RE_ZIPCODE_US.search(address)
        return m.group(1) if m else ""
    elif country_lower == "india":
        m = _RE_PINCODE_IN.search(address)
        return m.group(1) if m else ""
    elif country_lower == "france":
        m = _RE_POSTAL_FR.search(address)
        return m.group(1) if m else ""
    else:
        # Try all patterns
        for pattern in [_RE_ZIPCODE_US, _RE_PINCODE_IN]:
            m = pattern.search(address)
            if m:
                return m.group(1)
        return ""


def tokenize(text: str) -> list:
    """Split text into lowercase tokens."""
    if not text:
        return []
    return text.lower().split()
