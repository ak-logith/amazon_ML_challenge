"""
config.py — Dictionaries, mappings and regex patterns for preprocessing.

All normalisation look-ups live here so they can be tuned independently
of the transformation logic in preprocessing.py.
"""

import re

# ──────────────────────────────────────────────
# 1.  Legal-suffix canonicalisation
# ──────────────────────────────────────────────
#   The *keys* are lowercased tokens (or multi-word sequences) that appear
#   in noisy business names.  The *values* are the canonical short form
#   we normalise to — kept deliberately short so downstream similarity
#   scores focus on the substantive name.

LEGAL_SUFFIX_MAP: dict[str, str] = {
    # Corporation
    "corporation":  "corp",
    "corp":         "corp",
    "corp.":        "corp",
    # Incorporated
    "incorporated": "inc",
    "inc":          "inc",
    "inc.":         "inc",
    # Limited Liability Company
    "llc":          "llc",
    "l.l.c.":       "llc",
    "l.l.c":        "llc",
    "llc.":         "llc",
    # Limited Liability Partnership
    "llp":          "llp",
    "l.l.p.":       "llp",
    "l.l.p":        "llp",
    # Limited
    "limited":      "ltd",
    "ltd":          "ltd",
    "ltd.":         "ltd",
    "ltda":         "ltd",
    # Private
    "private":      "pvt",
    "pvt":          "pvt",
    "pvt.":         "pvt",
    "pvte":         "pvt",
    # Public
    "public":       "pub",
    "pub":          "pub",
    # Company
    "company":      "co",
    "co":           "co",
    "co.":          "co",
    # Partners / Partnership
    "partners":     "partners",
    "partnership":  "partners",
    # Associates
    "associates":   "associates",
    "assoc":        "associates",
    "assoc.":       "associates",
    # Society / Association
    "association":  "assn",
    "assn":         "assn",
    "society":      "society",
    "societe":      "societe",
    # French legal forms (test set)
    "sarl":         "sarl",
    "s.a.r.l.":     "sarl",
    "s.a.r.l":      "sarl",
    "sas":          "sas",
    "s.a.s.":       "sas",
    "s.a.s":        "sas",
    "sasu":         "sasu",
    "eurl":         "eurl",
    "sa":           "sa",
    "s.a.":         "sa",
    "sci":          "sci",
    "s.c.i.":       "sci",
    # Indian forms
    "nidhi":        "nidhi",
    "opc":          "opc",
    "section-8":    "sec8",
}

# Multi-word legal suffixes (order matters — longest first).
# We search for these as contiguous phrases in the token list.
LEGAL_SUFFIX_PHRASES: list[tuple[tuple[str, ...], str]] = [
    (("pvt", "ltd"),        "pvt ltd"),
    (("private", "limited"), "pvt ltd"),
    (("pvt.", "ltd."),      "pvt ltd"),
    (("pvt.", "ltd"),       "pvt ltd"),
    (("pvt", "ltd."),       "pvt ltd"),
    (("pub", "ltd"),        "pub ltd"),
    (("public", "limited"), "pub ltd"),
    (("pub.", "ltd."),      "pub ltd"),
    (("pub.", "ltd"),       "pub ltd"),
    (("pub", "ltd."),       "pub ltd"),
]

# ──────────────────────────────────────────────
# 2.  Filler / noise prefixes to strip from names
# ──────────────────────────────────────────────
NAME_FILLER_TOKENS: set[str] = {
    "dba", "smt", "shri", "shree", "sri", "m/s", "ms", "mr", "mrs",
    "dr", "the",
}

# ──────────────────────────────────────────────
# 3.  Street-type abbreviation expansion
# ──────────────────────────────────────────────
STREET_ABBREV_MAP: dict[str, str] = {
    "st":       "street",
    "st.":      "street",
    "ave":      "avenue",
    "ave.":     "avenue",
    "blvd":     "boulevard",
    "blvd.":    "boulevard",
    "bd":       "boulevard",
    "dr":       "drive",
    "dr.":      "drive",
    "rd":       "road",
    "rd.":      "road",
    "ct":       "court",
    "ct.":      "court",
    "ln":       "lane",
    "ln.":      "lane",
    "pl":       "place",
    "pl.":      "place",
    "pkwy":     "parkway",
    "pkwy.":    "parkway",
    "hwy":      "highway",
    "hwy.":     "highway",
    "cir":      "circle",
    "cir.":     "circle",
    "trl":      "trail",
    "trl.":     "trail",
    "ter":      "terrace",
    "ter.":     "terrace",
    "sq":       "square",
    "sq.":      "square",
    "aly":      "alley",
    "aly.":     "alley",
    "expy":     "expressway",
    "fwy":      "freeway",
    "rte":      "route",
    "rte.":     "route",
    "ste":      "suite",
    "ste.":     "suite",
    "apt":      "apartment",
    "apt.":     "apartment",
    "fl":       "floor",
    "fl.":      "floor",
    "bldg":     "building",
    "bldg.":    "building",
    "dept":     "department",
    "dept.":    "department",
    # French road types (keep canonical French form)
    "rue":      "rue",
    "bd.":      "boulevard",
    "imp":      "impasse",
    "imp.":     "impasse",
    "ch":       "chemin",
    "ch.":      "chemin",
}

# ──────────────────────────────────────────────
# 4.  US State name → 2-letter code
# ──────────────────────────────────────────────
US_STATE_TO_CODE: dict[str, str] = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct",
    "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi",
    "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me",
    "maryland": "md", "massachusetts": "ma", "michigan": "mi",
    "minnesota": "mn", "mississippi": "ms", "missouri": "mo",
    "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc", "puerto rico": "pr",
}

# Reverse: code → canonical full name  (used for display, not matching)
US_CODE_TO_STATE: dict[str, str] = {v: k for k, v in US_STATE_TO_CODE.items()}

# ──────────────────────────────────────────────
# 5.  Indian State / UT normalisations
# ──────────────────────────────────────────────
#   Maps common abbreviations, Indic-script names (post-transliteration)
#   and short-forms to a canonical lowercase English name.

INDIAN_STATE_MAP: dict[str, str] = {
    # Abbreviations
    "ap":   "andhra pradesh",
    "ar":   "arunachal pradesh",
    "as":   "assam",
    "br":   "bihar",
    "cg":   "chhattisgarh",
    "ct":   "chhattisgarh",
    "dl":   "delhi",
    "ga":   "goa",
    "gj":   "gujarat",
    "hp":   "himachal pradesh",
    "hr":   "haryana",
    "jh":   "jharkhand",
    "jk":   "jammu and kashmir",
    "ka":   "karnataka",
    "kl":   "kerala",
    "mh":   "maharashtra",
    "ml":   "meghalaya",
    "mn":   "manipur",
    "mp":   "madhya pradesh",
    "mz":   "mizoram",
    "nl":   "nagaland",
    "od":   "odisha",
    "or":   "odisha",
    "pb":   "punjab",
    "rj":   "rajasthan",
    "sk":   "sikkim",
    "tg":   "telangana",
    "tn":   "tamil nadu",
    "tr":   "tripura",
    "ts":   "telangana",
    "uk":   "uttarakhand",
    "up":   "uttar pradesh",
    "wb":   "west bengal",
    # Common transliterated / alternate names
    "tamilnadu":        "tamil nadu",
    "tamil":            "tamil nadu",   # only when it's the last token in state position
    "karnatak":         "karnataka",
    "maharastra":       "maharashtra",
    "orrisa":           "odisha",
    "orissa":           "odisha",
    "pondicherry":      "puducherry",
    "uttaranchal":      "uttarakhand",
    "chattisgarh":      "chhattisgarh",
    "chhatisgarh":      "chhattisgarh",
    # Post-unidecode transliterations of Indic state names
    "hriyaannaa":       "haryana",
    "hriyanaa":         "haryana",
    "uttr prdesh":      "uttar pradesh",
    "uttar prdesh":     "uttar pradesh",
    "uttara pradesha":  "uttar pradesh",
    "mhaaraassttrraa":  "maharashtra",
    "maharashttra":     "maharashtra",
    "tmiilnnaaddu":     "tamil nadu",
    "tamilnaadu":       "tamil nadu",
    "krnaattk":         "karnataka",
    "karnaatak":        "karnataka",
    "rjaasthaan":       "rajasthan",
    "rjaasthan":        "rajasthan",
    "gujaraat":         "gujarat",
    "gujarata":         "gujarat",
    "pnjaab":           "punjab",
    "pnjab":            "punjab",
    "pschm bnggaal":    "west bengal",
    "dillii":           "delhi",
    "dilli":            "delhi",
    "keral":            "kerala",
    "kerla":            "kerala",
    "mdhy prdesh":      "madhya pradesh",
    "madhya prdesh":    "madhya pradesh",
    "bihaar":           "bihar",
    "jhaarkhnd":        "jharkhand",
    "chhaattiisgdh":    "chhattisgarh",
    "aandhr prdesh":    "andhra pradesh",
    "tlnggaanaa":       "telangana",
    "telangaanaa":      "telangana",
    "oddishaa":         "odisha",
    "uttaraakhndd":     "uttarakhand",
    # Full names (for lookup normalisation)
    "andhra pradesh":       "andhra pradesh",
    "arunachal pradesh":    "arunachal pradesh",
    "assam":                "assam",
    "bihar":                "bihar",
    "chhattisgarh":         "chhattisgarh",
    "goa":                  "goa",
    "gujarat":              "gujarat",
    "haryana":              "haryana",
    "himachal pradesh":     "himachal pradesh",
    "jammu and kashmir":    "jammu and kashmir",
    "jharkhand":            "jharkhand",
    "karnataka":            "karnataka",
    "kerala":               "kerala",
    "madhya pradesh":       "madhya pradesh",
    "maharashtra":          "maharashtra",
    "manipur":              "manipur",
    "meghalaya":            "meghalaya",
    "mizoram":              "mizoram",
    "nagaland":             "nagaland",
    "odisha":               "odisha",
    "punjab":               "punjab",
    "rajasthan":            "rajasthan",
    "sikkim":               "sikkim",
    "tamil nadu":           "tamil nadu",
    "telangana":            "telangana",
    "tripura":              "tripura",
    "uttar pradesh":        "uttar pradesh",
    "uttarakhand":          "uttarakhand",
    "west bengal":          "west bengal",
    "delhi":                "delhi",
    "puducherry":           "puducherry",
    "chandigarh":           "chandigarh",
    "dadra and nagar haveli": "dadra and nagar haveli",
    "daman and diu":        "daman and diu",
    "lakshadweep":          "lakshadweep",
    "andaman and nicobar":  "andaman and nicobar",
    "ladakh":               "ladakh",
}

# All canonical Indian state names (for reverse lookup)
INDIAN_STATE_CANONICAL: set[str] = set(INDIAN_STATE_MAP.values())

# ──────────────────────────────────────────────
# 6.  French region canonicalisation  (test set)
# ──────────────────────────────────────────────
FRENCH_REGION_MAP: dict[str, str] = {
    "ile-de-france":            "ile-de-france",
    "ile de france":            "ile-de-france",
    "idf":                      "ile-de-france",
    "nouvelle-aquitaine":       "nouvelle-aquitaine",
    "nouvelle aquitaine":       "nouvelle-aquitaine",
    "hauts-de-france":          "hauts-de-france",
    "hauts de france":          "hauts-de-france",
    "pays de la loire":         "pays de la loire",
    "bretagne":                 "bretagne",
    "normandie":                "normandie",
    "occitanie":                "occitanie",
    "auvergne-rhone-alpes":     "auvergne-rhone-alpes",
    "auvergne rhone alpes":     "auvergne-rhone-alpes",
    "provence-alpes-cote d'azur": "provence-alpes-cote d'azur",
    "paca":                     "provence-alpes-cote d'azur",
    "grand est":                "grand est",
    "bourgogne-franche-comte":  "bourgogne-franche-comte",
    "centre-val de loire":      "centre-val de loire",
    "corse":                    "corse",
}

# ──────────────────────────────────────────────
# 7.  Ampersand / conjunction normalisation
# ──────────────────────────────────────────────
#   Only & and + are safe for simple replacement.
#   'et' is NOT included because it appears inside English words
#   (e.g. 'street', 'market').
CONJUNCTION_CHARS: dict[str, str] = {
    "&":    " and ",
    "+":    " and ",
}

# Whole-word regex for 'et' (French 'and') — applied only in name context
FRENCH_ET_RE = re.compile(r"\bet\b", re.IGNORECASE)

# ──────────────────────────────────────────────
# 8.  House-number prefix patterns to strip
# ──────────────────────────────────────────────

# Patterns like "H.No.", "H.NO", "KH NO.", "Door No", "D.No."
HOUSE_NUMBER_PREFIX_RE = re.compile(
    r"\b(?:h\.?\s*no\.?|kh\.?\s*no\.?|door\s*no\.?|d\.?\s*no\.?|plot\s*no\.?|flat\s*no\.?|shop\s*no\.?|block\s*no\.?)\s*",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────
# 9.  Landmark / noise phrases to strip from addresses
# ──────────────────────────────────────────────
LANDMARK_PHRASE_RE = re.compile(
    r"\b(?:near|opp\.?|opposite|behind|beside|adjacent\s+to|next\s+to|in\s+front\s+of|c/o|care\s+of)\s+[^,]*",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────
# 10. Noise tokens / patterns to remove
# ──────────────────────────────────────────────
# URL pattern
URL_RE = re.compile(r"(?:https?://)?(?:www\.)?[\w\-]+\.(?:com|org|net|in|co\.in|co|io)\b", re.IGNORECASE)

# Phone numbers (7+ digits, possibly with separators)
PHONE_RE = re.compile(r"\b\d[\d\s\-\.]{6,}\d\b")

# Literal null / nan
NULL_LITERAL_RE = re.compile(r"\b(?:null|nan|none|n/a|na)\b", re.IGNORECASE)

# Brackets and their contents when used as noise wrappers
BRACKET_NOISE_RE = re.compile(r"[<>\[\]{}«»]")

# Pipe separator (sometimes used to append URLs or alternate names)
PIPE_RE = re.compile(r"\|.*$")

# ──────────────────────────────────────────────
# 11. Indian PIN code pattern (6 digits)
# ──────────────────────────────────────────────
INDIAN_PIN_RE = re.compile(r"\b(\d{6})\b")

# US ZIP code pattern (5 digits, optionally +4)
US_ZIP_RE = re.compile(r"(\b[a-zA-Z]+\s+|,\s*)(\d{5})(?:-\d{4})?\b")

# French postal code (5 digits)
FRENCH_POSTAL_RE = re.compile(r"(,\s*|^\s*)(\d{5})\b")

