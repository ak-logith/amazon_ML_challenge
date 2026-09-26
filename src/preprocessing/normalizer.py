"""
preprocessing.py — Core text-cleaning and normalisation functions.

Every function is a pure transform:  str → str  (or Series → Series).
They are composed together by `run_preprocess.py`.

Design notes
------------
*  We keep *both* the original text and a transliterated copy so that
   downstream blocking / matching can choose which representation to use.
*  Legal suffixes are stripped from the core name and stored separately.
*  Addresses are lightly parsed into (city, state, postal_code) where
   feasible; the remaining text stays in a single `address_clean` field.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from unidecode import unidecode

try:
    from . import config as C
except ImportError:
    import config as C


# ============================================================
#  1.  Low-level text helpers
# ============================================================

def unicode_normalise(text: str) -> str:
    """NFKD-normalise and strip combining marks (accents / diacritics)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")


unicode_normalize = unicode_normalise
strip_accents = unicode_normalise


def transliterate(text: str) -> str:
    """Transliterate any script to ASCII-Latin via unidecode.

    Returns a *lowercased* ASCII string.  Non-Latin scripts (Devanagari,
    Tamil, Kannada, Gujarati …) are converted to their closest Latin
    phonetic equivalents.
    """
    return unidecode(text).lower()


def collapse_whitespace(text: str) -> str:
    """Replace runs of whitespace (including tabs/newlines) with a single space."""
    return re.sub(r"\s+", " ", text).strip()


def strip_nulls(text: str) -> str:
    """Replace literal 'null', 'nan', 'None', 'N/A' tokens with empty string."""
    return C.NULL_LITERAL_RE.sub("", text)


def strip_urls(text: str) -> str:
    """Remove URL-like tokens."""
    return C.URL_RE.sub("", text)


def strip_phones(text: str) -> str:
    """Remove phone-number-like digit sequences (7+ digits)."""
    return C.PHONE_RE.sub("", text)


def strip_brackets(text: str) -> str:
    """Remove bracket noise characters: < > [ ] { } « »."""
    return C.BRACKET_NOISE_RE.sub("", text)


def strip_pipe_suffix(text: str) -> str:
    """Remove everything after a pipe character (often appended URLs / alt names)."""
    return C.PIPE_RE.sub("", text)


def normalise_conjunctions(text: str) -> str:
    """& / +  →  'and'.  Character-level only (no word-boundary issues)."""
    for src, tgt in C.CONJUNCTION_CHARS.items():
        text = text.replace(src, tgt)
    return text


normalize_conjunctions = normalise_conjunctions


# ============================================================
#  2.  Business-name normalisation
# ============================================================

def _extract_legal_suffix(tokens: list[str]) -> tuple[list[str], str]:
    """Separate legal-suffix tokens from the core name tokens.

    Returns (core_tokens, canonical_suffix_string).

    Strategy:
      1.  Try multi-word suffix phrases first (e.g. "pvt ltd").
      2.  Then try single-token suffixes at the *end* or *start* of the list.
    """
    suffix_parts: list[str] = []
    remaining = list(tokens)

    # --- multi-word phrases (scanned from the right) ---
    for phrase_tokens, canonical in C.LEGAL_SUFFIX_PHRASES:
        plen = len(phrase_tokens)
        if len(remaining) >= plen:
            tail = tuple(remaining[-plen:])
            if tail == phrase_tokens:
                suffix_parts.append(canonical)
                remaining = remaining[:-plen]
                break  # one phrase match is enough

    # --- single-token suffixes at the tail ---
    while remaining and remaining[-1] in C.LEGAL_SUFFIX_MAP:
        tok = remaining.pop()
        canonical = C.LEGAL_SUFFIX_MAP[tok]
        if canonical not in suffix_parts:
            suffix_parts.insert(0, canonical)

    # --- single-token suffixes at the head (e.g. "LLC Crystal …") ---
    while remaining and len(remaining) > 1 and remaining[0] in getattr(C, "LEGAL_PREFIXES", set()):
        tok = remaining.pop(0)
        canonical = C.LEGAL_SUFFIX_MAP.get(tok, tok)
        if canonical not in suffix_parts:
            suffix_parts.append(canonical)

    return remaining, " ".join(suffix_parts)


def clean_business_name(raw: str) -> dict[str, str]:
    """Full normalisation pipeline for a business name.

    Returns a dict with keys:
        name_original   – original value, unchanged
        name_clean      – lowercased, diacritics stripped, noise removed
        name_translit   – unidecode transliteration (ASCII only)
        name_core       – clean name with legal suffix + filler stripped
        legal_suffix    – canonical legal suffix (e.g. "pvt ltd")
    """
    if not raw or not raw.strip():
        return {
            "name_original": raw or "",
            "name_clean": "",
            "name_translit": "",
            "name_core": "",
            "legal_suffix": "",
        }

    original = raw.strip()

    # Step 1: transliterate (Indic → Latin) — keep a separate copy
    transliterated = transliterate(original)

    # Step 2: basic cleaning on the transliterated form
    clean = transliterated
    clean = strip_pipe_suffix(clean)
    clean = strip_urls(clean)
    clean = strip_phones(clean)
    clean = strip_brackets(clean)
    clean = normalise_conjunctions(clean)
    # Split hyphen-joined suffixes (e.g. "solutions-l.l.c." → "solutions l.l.c.")
    clean = re.sub(r"-(?=l\.?l\.?c|l\.?l\.?p|inc|ltd|corp|pvt|pub|llc|llp)", " ", clean, flags=re.IGNORECASE)
    # Remove stray punctuation but keep alphanumeric, spaces, hyphens, dots
    clean = re.sub(r"[^\w\s\-\.]", " ", clean)
    clean = collapse_whitespace(clean)

    # Step 3: tokenise and extract legal suffix
    tokens = clean.split()

    # Remove filler prefixes
    while tokens and tokens[0] in C.NAME_FILLER_TOKENS:
        tokens.pop(0)

    core_tokens, legal_suffix = _extract_legal_suffix(tokens)
    name_core = " ".join(core_tokens)

    # Strip dangling hyphens / dots from name_core edges
    name_core = re.sub(r"^[\-\.\s]+|[\-\.\s]+$", "", name_core)

    # Step 4: also produce a "clean" version (with suffix intact, just normalised)
    name_clean = " ".join(tokens) if tokens else clean

    return {
        "name_original": original,
        "name_clean": name_clean,
        "name_translit": transliterated,
        "name_core": name_core,
        "legal_suffix": legal_suffix,
    }


# ============================================================
#  3.  Address normalisation
# ============================================================

def _expand_street_abbrevs(text: str) -> str:
    """Expand street-type abbreviations using word-boundary matching."""
    tokens = text.split()
    expanded = []
    for tok in tokens:
        low = tok.lower().rstrip(",")
        trailing_comma = tok.endswith(",")
        if low in C.STREET_ABBREV_MAP:
            replacement = C.STREET_ABBREV_MAP[low]
            expanded.append(replacement + ("," if trailing_comma else ""))
        else:
            expanded.append(tok)
    return " ".join(expanded)


def _normalise_state_us(text: str) -> Optional[str]:
    """Try to match a US state name or abbreviation → 2-letter code."""
    low = text.strip().lower()
    if low in C.US_STATE_TO_CODE:
        return C.US_STATE_TO_CODE[low]
    if low in C.US_CODE_TO_STATE:
        return low  # already a code
    return None


def _normalise_state_india(text: str) -> Optional[str]:
    """Try to match an Indian state name / abbreviation → canonical name."""
    low = text.strip().lower()
    if low in C.INDIAN_STATE_MAP:
        return C.INDIAN_STATE_MAP[low]
    return None


def _normalise_region_france(text: str) -> Optional[str]:
    """Try to match a French region → canonical name."""
    low = text.strip().lower()
    if low in C.FRENCH_REGION_MAP:
        return C.FRENCH_REGION_MAP[low]
    return None


def _extract_postal_code(text: str, country: str) -> tuple[str, str]:
    """Try to extract a postal code from the address text.

    Returns (remaining_text, postal_code).
    """
    if country == "india":
        m = C.INDIAN_PIN_RE.search(text)
        if m:
            pin = m.group(1)
            # Basic validation: Indian PINs are 1xxxxx – 9xxxxx
            if pin[0] != "0":
                return text[:m.start()] + text[m.end():], pin
    elif country == "us":
        m = C.US_ZIP_RE.search(text)
        if m:
            prefix = m.group(1)
            pin = m.group(2)
            return text[:m.start()] + prefix + ", " + text[m.end():], pin
    elif country == "france":
        m = C.FRENCH_POSTAL_RE.search(text)
        if m:
            prefix = m.group(1)
            pin = m.group(2)
            return text[:m.start()] + prefix + ", " + text[m.end():], pin
    return text, ""


def _extract_state(parts: list[str], country: str) -> tuple[list[str], str]:
    """Try to identify and extract a state/region from comma-separated parts.

    Scans from the *last* part backwards (state is usually at the end),
    but also scans all positions for transliterated Indic state names.
    Returns (remaining_parts, canonical_state).
    """
    normaliser = {
        "us":     _normalise_state_us,
        "india":  _normalise_state_india,
        "france": _normalise_region_france,
    }.get(country)

    if not normaliser:
        return parts, ""

    # First pass: check last 2 parts (most common position for state)
    for span in (2, 1):
        if len(parts) >= span:
            candidate = ", ".join(parts[-span:]).strip()
            result = normaliser(candidate)
            if result:
                return parts[:-span], result

    # Second pass: scan ALL parts (handles reordered addresses where
    # the state appears first, e.g. transliterated "hriyaannaa, ...")
    for i, part in enumerate(parts):
        candidate = part.strip()
        result = normaliser(candidate)
        if result:
            return parts[:i] + parts[i+1:], result

    return parts, ""


def _extract_city(parts: list[str], state: str = "") -> tuple[list[str], str]:
    """Heuristic: the last remaining comma-separated part is likely the city.

    Skips parts that duplicate the already-extracted state or are pure numeric.
    """
    # Sometimes a duplicate of the state remains; skip it
    while parts and state and parts[-1].strip().lower() == state:
        parts = parts[:-1]

    # Skip trailing pure-numeric parts (phone remnants, PINs already extracted)
    while parts and re.match(r"^\d+$", parts[-1].strip()):
        parts = parts[:-1]

    if parts:
        city = parts[-1].strip()
        if city and len(city) > 1:
            return parts[:-1], city
    return parts, ""


def clean_address(raw: str, country: str) -> dict[str, str]:
    """Full normalisation pipeline for a business address.

    Returns a dict with keys:
        addr_original     – original value, unchanged
        addr_clean        – lowered, noise stripped, abbreviations expanded
        addr_translit     – unidecode transliteration
        addr_city         – extracted city (best effort)
        addr_state        – extracted state / region (canonical)
        addr_postal_code  – extracted postal / ZIP / PIN code
    """
    if not raw or not raw.strip():
        return {
            "addr_original": raw or "",
            "addr_clean": "",
            "addr_translit": "",
            "addr_city": "",
            "addr_state": "",
            "addr_postal_code": "",
        }

    original = raw.strip()
    country_low = country.strip().lower() if country else ""

    # Step 1: transliterate
    transliterated = transliterate(original)

    # Step 2: basic cleaning
    clean = transliterated
    clean = strip_nulls(clean)
    clean = strip_urls(clean)
    clean = strip_brackets(clean)

    # Strip landmark phrases
    clean = C.LANDMARK_PHRASE_RE.sub("", clean)

    # Normalise house-number prefixes
    clean = C.HOUSE_NUMBER_PREFIX_RE.sub("", clean)

    # Normalise conjunctions
    clean = normalise_conjunctions(clean)

    # Remove stray punctuation (keep alphanumeric, spaces, commas, hyphens, dots, slashes, hash)
    clean = re.sub(r"[^\w\s,\.\-/#]", " ", clean)

    # Expand street abbreviations
    clean = _expand_street_abbrevs(clean)

    clean = collapse_whitespace(clean)

    # Step 3: extract structured components
    # Extract postal code first (before splitting on commas)
    remaining, postal_code = _extract_postal_code(clean, country_low)

    # Split on commas for city / state extraction
    parts = [p.strip() for p in remaining.split(",") if p.strip()]

    # Extract state
    parts, state = _extract_state(parts, country_low)

    # Extract city
    parts, city = _extract_city(parts, state)

    # Reassemble the remaining address (street-level detail)
    addr_clean = ", ".join(p for p in parts if p)
    addr_clean = collapse_whitespace(addr_clean)

    # Remove dangling commas / dots
    addr_clean = re.sub(r"^[,.\s]+|[,.\s]+$", "", addr_clean)

    return {
        "addr_original": original,
        "addr_clean": addr_clean,
        "addr_translit": transliterated,
        "addr_city": city,
        "addr_state": state,
        "addr_postal_code": postal_code,
    }


# ============================================================
#  4.  Row-level processing  (used by run_preprocess.py)
# ============================================================

def preprocess_row(
    entity_id: str,
    business_name: str,
    business_address: str,
    country: str,
) -> dict[str, str]:
    """Process a single record → flat dict of all cleaned fields.

    This is the main entry point called per-row by the batch runner.
    """
    name_fields = clean_business_name(business_name or "")
    addr_fields = clean_address(business_address or "", country or "")

    return {
        "entity_id":        entity_id,
        "country":          (country or "").strip().lower(),
        "business_name":    business_name or "",
        "business_address": business_address or "",
        **name_fields,
        **addr_fields,
    }
