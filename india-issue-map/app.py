"""
India Care Map — Databricks App backend (Flask)
==================================================

What this file does, step by step:
  1. Serves the map page (templates/index.html) when you open the app.
  2. When you submit an address + issue, it geocodes the address (turns the
     text into latitude/longitude) using Nominatim, the free geocoder from
     OpenStreetMap — restricted to India.
  3. Asks an AI model service (via Unity Gateway) to read the patient's
     description and identify the medical services most likely needed —
     even when the description is long, indirect or written by a helper
     rather than the patient.
  4. Recommends facilities for the problem with urgency-based rules:
      how well their specialties/equipment (facilities_medical.csv)
      cover the AI-identified needs, how urgent the case is, and how
      certain the dataset is that each place actually exists.
  5. Looks up a matching "issue statement" in a Databricks table.
  6. Optionally saves every submission back into that table.

All connection details come from ENVIRONMENT VARIABLES (see app.yaml),
so nothing sensitive is hardcoded in this file.
"""

import csv  # reads data.csv, the app's built-in highlight points
import difflib  # fuzzy matching for "did you mean" address suggestions
import gzip  # reads the gzipped score-explanation payload (section 3c)
import json  # parses the AI model's JSON replies and the capability lists
import math  # distance math between map points
import os  # lets us read environment variables
import re  # regex helpers: address-keyword classification for the geocode fallback
import time  # used to pause between geocoding requests (rate limiting)

import requests  # a simple library for making HTTP calls
from flask import Flask, jsonify, render_template, request

# Databricks libraries:
#   - DatabricksConfig picks up the OAuth credentials that Databricks Apps
#     automatically injects into the app's environment (no tokens in code).
#   - dbsql is the SQL connector we use to query a SQL warehouse.
from databricks.sdk.core import Config as DatabricksConfig
from databricks import sql as dbsql

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 1. CONFIGURATION — everything is read from environment variables.
#    os.environ.get("NAME", default) returns the variable's value, or the
#    default if the variable is not set. Set these in app.yaml or Apps UI.
# ---------------------------------------------------------------------------

# Your workspace URL and the SQL warehouse to query, e.g.
#   DATABRICKS_HOST=https://adb-123456789.0.azuredatabricks.net
#   DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/abc123def456
WAREHOUSE_HTTP_PATH = os.environ.get("DATABRICKS_HTTP_PATH", "")

# Which Unity Catalog table holds the issues (catalog.schema.table).
ISSUE_CATALOG = os.environ.get("ISSUE_CATALOG", "")
ISSUE_SCHEMA = os.environ.get("ISSUE_SCHEMA", "")
ISSUE_TABLE = os.environ.get("ISSUE_TABLE", "")

# Which columns to use. Defaults match a typical civic-issues table;
# override them with environment variables if your table differs.
LOCATION_COL = os.environ.get("ISSUE_LOCATION_COL", "location")
ISSUE_TYPE_COL = os.environ.get("ISSUE_TYPE_COL", "issue_type")
STATEMENT_COL = os.environ.get("ISSUE_STATEMENT_COL", "issue_statement")
LAT_COL = os.environ.get("ISSUE_LAT_COL", "latitude")
LNG_COL = os.environ.get("ISSUE_LNG_COL", "longitude")

# Set SAVE_SUBMISSIONS=true to also INSERT every submitted issue into the table.
SAVE_SUBMISSIONS = os.environ.get("SAVE_SUBMISSIONS", "false").lower() in ("1", "true", "yes")


def fully_qualified_table():
    """Return the table name as catalog.schema.table (Unity Catalog style)."""
    return f"{ISSUE_CATALOG}.{ISSUE_SCHEMA}.{ISSUE_TABLE}"


# ---------------------------------------------------------------------------
# 2. GEOCODING — turn a typed address into coordinates with Nominatim.
# ---------------------------------------------------------------------------


def geocode_candidates(address, limit=1):
    """
    Ask Nominatim (OpenStreetMap's free geocoder) for up to `limit`
    matches for `address`, restricted to India, best first.

    Returns a list of dicts like {"lat": 19.07, "lng": 72.87,
    "display_name": "Mumbai, ..."} — empty when nothing is found (or the
    service fails). Callers that just want one answer use geocode().
    """
    # Nominatim's usage policy allows at most 1 request per second, so pause
    # briefly before calling it.
    time.sleep(1.1)
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": address,            # the address typed by the user
                "format": "json",         # we want machine-readable JSON back
                "limit": limit,           # how many matches to fetch
                "countrycodes": "in",     # <<< restrict results to India
            },
            # Nominatim requires a User-Agent header that identifies the app.
            headers={"User-Agent": "india-issue-map-databricks-app/1.0"},
            timeout=10,
        )
        response.raise_for_status()  # raise an error if the HTTP call failed
        results = response.json()
    except requests.RequestException:
        return []

    return [
        {
            "lat": float(r["lat"]),
            "lng": float(r["lon"]),
            "display_name": r.get("display_name", address),
        }
        for r in results
    ]


def geocode(address):
    """
    The single-answer form of geocode_candidates: the best match for
    `address` in India, or None if it can't be found.
    """
    matches = geocode_candidates(address, limit=1)
    return matches[0] if matches else None


# --- Ambiguity helpers -----------------------------------------------------
# A vague query ("C. S. Lewis", "MG Road") can match many places. Picking
# Nominatim's first answer silently sends the user to ONE arbitrary spot
# among the gazillion — so short queries get checked for ambiguity, and
# ambiguous ones are answered with several "did you mean" options.


def _query_tokens(text):
    """Lowercased alphanumeric tokens of a query ("C. S. Lewis" ->
    ["c", "s", "lewis"]) — used to judge how much of what was typed the
    geocoder's answer actually contains."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _token_coverage(tokens, display_name):
    """Fraction of the query's tokens that appear as standalone words in
    the geocoder's answer. "C. S. Lewis" answered with "MAC Cosmetics,
    Lewis Road, ..." matches only "lewis" (1/3) — the answer ignored
    most of the query, a sign of a lucky keyword hit, not a real match."""
    if not tokens:
        return 1.0
    words = set(re.findall(r"[a-z0-9]+", (display_name or "").lower()))
    return sum(1 for t in tokens if t in words) / len(tokens)


def _distinct_places(candidates, min_km=10.0):
    """Collapse candidates that sit within min_km of an already-kept one
    (the same street matched several ways) and return the distinct
    clusters, best-ranked first."""
    clusters = []
    for cand in candidates:
        if all(_distance_km(cand["lat"], cand["lng"], kept["lat"], kept["lng"]) > min_km
               for kept in clusters):
            clusters.append(cand)
    return clusters


def _ambiguous_query(query, candidates):
    """True when a geocode answer is NOT confident enough to fly to
    automatically. Only short queries (three words or fewer) can be
    ambiguous; they are, when Nominatim found several different places
    for them, or when their one answer ignores most of what was typed.
    Longer, specific queries always fly to their top answer."""
    tokens = _query_tokens(query)
    if not candidates or len(tokens) > 3:
        return False
    if len(_distinct_places(candidates)) >= 2:
        return True
    return _token_coverage(tokens, candidates[0]["display_name"]) < 0.5


def _short_display(name, parts=3):
    """First few comma-parts of a long Nominatim display name, for a
    compact suggestion button."""
    bits = [b.strip() for b in (name or "").split(",") if b.strip()]
    return ", ".join(bits[:parts]) if bits else (name or "Unnamed place")





# ---------------------------------------------------------------------------
# 3. MY DATA — points in data.csv, shown as the orange dots on the map.
#    (The "best nearby facilities" search runs on the richer capability
#    file facilities_medical.csv — see section 3b below.)
# ---------------------------------------------------------------------------
# Put your own rows in data.csv (see data-template.csv for the format):
#   latitude,longitude,name,type,address,pin,city,state
#   28.6139,77.2090,Connaught Place,market,,110001,New Delhi,Delhi
# latitude/longitude are the PRE-GEOCODED positions (city-level for
# most facilities) — a search reads them straight from the file, so the
# only address geocoded live is the one the user types. address/pin/city/
# state are kept for reference and future geocoder upgrades.


def _read_data_points():
    """Read data.csv and return a list of {lat,lng,name,type} points."""
    path = os.path.join(app.root_path, "data.csv")
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = []
            for raw in csv.DictReader(f):
                # Normalize the column names so lat/lng spellings are flexible.
                row = {(k or "").strip().lower(): (v or "").strip()
                       for k, v in raw.items()}
                try:
                    lat = float(row.get("latitude") or row.get("lat") or "")
                    lng = float(row.get("longitude") or row.get("lng")
                                or row.get("lon") or "")
                except ValueError:            # missing or not-a-number coords
                    continue
                rows.append({
                    "lat": lat,
                    "lng": lng,
                    "name": row.get("name") or row.get("label") or row.get("title") or "",
                    "type": row.get("type") or row.get("kind") or row.get("category") or "",
                    "address": row.get("address") or row.get("address_line1") or "",
                    "city": row.get("city") or "",
                    "state": row.get("state") or "",
                    "pin": row.get("pin") or row.get("zip") or row.get("postcode") or "",
                })
            return rows
    except FileNotFoundError:
        # No data.csv yet — nothing to highlight or search.
        return []


@app.route("/api/my-data")
def my_data():
    """Return the points in data.csv, for the orange dots on the map."""
    return jsonify(_read_data_points())


def _distance_km(lat1, lng1, lat2, lng2):
    """
    Straight-line distance between two (lat, lng) points, in kilometers.
    Uses the haversine formula — the standard way to measure distance on
    a sphere like Earth.
    """
    radius_km = 6371.0                       # Earth's average radius
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(d_lng / 2) ** 2)
    return 2 * radius_km * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# 3b. AI PROBLEM INTERPRETATION + CAPABILITY MATCHING
#
#     A patient's problem arrives as free text — often a long, indirect
#     story written by a relative or NGO worker ("we're helping a pregnant
#     woman near Nashik, her doctor said the delivery could be risky…").
#     Two steps turn that into a hospital ranking:
#
#       1. An AI model service reads the story and extracts the medical
#          services most likely needed (obstetrics, NICU, blood bank…),
#          plus an urgency level. Cached per text — one LLM call per
#          unique problem, not per facility.
#       2. Every facility in facilities_medical.csv (9,221 facilities,
#          each with its specialties/equipment/procedures list) is scored
#          on how well it covers those needs, combined with distance and
#          facility type. Results say WHY: the matched services are
#          returned with each facility.
#
#     If the AI call fails, ranking falls back to pure distance so the
#     map keeps working.
# ---------------------------------------------------------------------------

# The Unity Gateway model service to use. app.yaml declares it as a
# uc_securable resource (which grants this app's service principal
# EXECUTE on it) and injects its three-part name here.
MODEL_SERVICE_NAME = os.environ.get("MODEL_SERVICE_NAME", "system.ai.gpt-oss-120b")

_MEDICAL_FACILITIES = None    # cache: facilities_medical.csv rows, loaded once
_INTERPRETATION_CACHE = {}    # problem text -> interpretation (LLM calls cost seconds)


# --- Recommendation rules by urgency -----------------------------------------
# The AI labels each submitted problem routine / urgent / emergency, and
# each level has its own search radius and existence-confidence rules
# (the dataset scores every facility for "does this place actually exist").

_MILES_TO_KM = 1.609344

# Life-threatening: only places that almost certainly exist, and only
# within reach of an ambulance.
EMERGENCY_RADIUS_KM = 60 * _MILES_TO_KM     # ~97 km
EMERGENCY_CONFIDENCE = 0.90                # 90%+ certain to exist

# Urgent: the same tight radius. The equipped pick may be a bit less
# certain, the existence pick does not need to be equipped.
URGENT_RADIUS_KM = 60 * _MILES_TO_KM
URGENT_EQUIPPED_CONFIDENCE = 0.75
CERTAIN_CONFIDENCE = 0.90                  # "certain to actually exist"

# Everything else: the patient can travel farther for the right place.
ROUTINE_RADIUS_KM = 200 * _MILES_TO_KM     # ~322 km
CLOSE_KM = 20 * _MILES_TO_KM               # "close" for the routine slots (~32 km)

# A facility counts as EQUIPPED for a problem when it covers at least
# half of the AI-identified needs (always at least one).
EQUIPPED_FRACTION = 0.5


# ---------------------------------------------------------------------------
# 3c. SCORE EXPLANATIONS — when a facility is recommended, recreate the
#     dataset's confidence values from the three scoring scripts so the
#     card can show HOW each number was derived, weights included:
#
#     existence_confidence (existence_confidence.py): a logistic model —
#       log-odds = -1*recency + 1*address + 1*(reference-dataset matches)
#                  + 1*years established, squashed with a sigmoid. The
#       reference datasets it cross-checks against are not shipped with
#       the app, so that term is recovered from the stored value itself.
#     quality_score (quality_score.py): a weighted mix of accreditation
#       (.35), services (.20), specialties (.18), equipment (.10),
#       staffing (.10), rating (.02) and affiliation (.05) — renormalised
#       over the components a facility actually has data for.
#     procedure confidence (procedure_confidence.py): direct evidence
#       (.80-.90, the procedure is named), supporting (.50-.60, its
#       equipment is listed), inferred (.25, a specialty covers it).
#
#     facilities_explain.csv.gz (built by running the scripts' logic over
#     the source dataset) carries each facility's raw inputs;
#     procedure_lookup.csv holds the synonym/equipment/specialty table.
#     Both are keyed by (name, city, state, lat, lng).
# ---------------------------------------------------------------------------

# Rescaling ranges the existence script computed over the source dataset
# (1st/99th percentile caps; each variable is squashed to 0-1 inside them).
_EXIST_DAYS_RANGE = (1.0, 4357.4)    # days since last website/social activity
_EXIST_YEARS_RANGE = (3.0, 118.0)    # years since the facility was established

# quality_score.py's component weights, and which components each
# facility type is scored on.
_QUALITY_WEIGHTS = {
    "accreditation": 0.35, "services": 0.20, "specialties": 0.18,
    "equipment": 0.10, "staffing": 0.10, "rating": 0.02, "affiliation": 0.05,
}
_QUALITY_APPLIES = {
    "hospital": ("accreditation", "services", "specialties", "equipment",
                 "staffing", "rating", "affiliation"),
    "clinic": ("accreditation", "specialties", "equipment", "staffing", "rating"),
    "dentist": ("accreditation", "specialties", "equipment", "staffing", "rating"),
    "other": ("accreditation", "specialties", "equipment", "staffing", "rating"),
}
_NABH_LABEL = {0: "no NABH status on record", 1: "NABH-empanelled",
               2: "NABH-certified", 3: "NABH-accredited"}

# procedure_confidence.py's evidence-level scores.
_PC_DIRECT_BASE = 0.80           # one direct mention
_PC_DIRECT_PER_EXTRA = 0.05      # each additional distinct mention ...
_PC_DIRECT_CAP = 0.90           # ... up to this
_PC_DIRECT_EQUIPMENT_BONUS = 0.05
_PC_DIRECT_SPECIALTY_BONUS = 0.03
_PC_EQUIPMENT_ONLY = 0.50        # needed equipment listed, not the procedure
_PC_EQUIPMENT_SPECIALTY_BONUS = 0.10
_PC_SPECIALTY_ONLY = 0.25        # only the relevant specialty is listed

_SCORE_EXPLANATIONS = None       # cache: facilities_explain.csv.gz rows
_PROCEDURE_LOOKUP = None          # cache: procedure_lookup.csv


def _load_score_explanations():
    """Read the per-facility scoring inputs once. The file is gzipped to
    keep the app bundle small; a missing file simply disables the
    per-card score breakdowns."""
    global _SCORE_EXPLANATIONS
    if _SCORE_EXPLANATIONS is None:
        table = {}
        try:
            path = os.path.join(app.root_path, "facilities_explain.csv.gz")
            with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
                for raw in csv.DictReader(f):
                    table[(raw.get("name") or "", raw.get("city") or "",
                           raw.get("state") or "",
                           round(float(raw.get("lat") or 0), 6),
                           round(float(raw.get("lng") or 0), 6))] = raw
        except (OSError, ValueError):     # missing or unreadable payload
            pass
        _SCORE_EXPLANATIONS = table
    return _SCORE_EXPLANATIONS


def _load_procedure_lookup():
    """procedure_confidence.py's synonym / equipment / specialty table
    (semicolon-separated cells, editable)."""
    global _PROCEDURE_LOOKUP
    if _PROCEDURE_LOOKUP is None:
        table = {}

        def _split(cell):
            return [x.strip().lower() for x in (cell or "").split(";")
                    if x.strip()]

        try:
            path = os.path.join(app.root_path, "procedure_lookup.csv")
            with open(path, newline="", encoding="utf-8") as f:
                for raw in csv.DictReader(f):
                    table[(raw.get("procedure") or "").strip().lower()] = {
                        "synonyms": _split(raw.get("synonyms")),
                        "specialty_keywords": _split(raw.get("specialty_keywords")),
                        "equipment": _split(raw.get("equipment")),
                    }
        except OSError:                    # missing lookup: bare-word matching
            pass
        _PROCEDURE_LOOKUP = table
    return _PROCEDURE_LOOKUP


# --- procedure_confidence, ported to run on the app's facility rows -------
# Same matching rules as the script: word-bounded, case-insensitive
# phrases with interchangeable spaces/hyphens and optional plurals;
# negation windows before/after a hit; JSON-list or sentence-split cell
# parsing; specialty keywords matched on letters/digits only.

_PC_NEG_BEFORE = re.compile(
    r"\b(no|not|non|without|lacks?|lacking|unavailable|discontinued|planned|upcoming|proposed|future|"
    r"soon|refers?|referred|referral to|does not|doesn't|do not|cannot|can't|yet to)\b[^.;]{0,40}$", re.I)
_PC_NEG_AFTER = re.compile(
    r"^[^.;]{0,40}\b(not (?:available|offered|provided|functional|done|performed)|unavailable|"
    r"to be (?:shut|closed|started|set up|established)|discontinued|is planned|are planned|coming soon|"
    r"under construction|non[- ]functional|shut down|referred (?:to|out))", re.I)

_PC_PATTERN_CACHE = {}


def _pc_pattern(phrase):
    """Word-bounded, case-insensitive pattern; spaces/hyphens
    interchangeable, optional plural (exactly as in the script)."""
    pat = _PC_PATTERN_CACHE.get(phrase)
    if pat is None:
        parts = [re.escape(p) for p in re.split(r"[\s\-]+", phrase) if p]
        body = r"[\s\-]*".join(parts)
        pat = _PC_PATTERN_CACHE[phrase] = re.compile(
            r"(?<![a-z0-9])" + body + r"(?:s|es)?(?![a-z0-9])", re.I)
    return pat


def _pc_items(value):
    """Turn a raw dataset cell (JSON list or free text) into short items."""
    s = value if isinstance(value, str) else ""
    s = s.strip()
    if not s or s.lower() == "nan":
        return []
    if s.startswith("["):
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(x) for x in v if x is not None and str(x).strip()]
        except ValueError:
            pass
    return [x for x in re.split(r"(?<=[.!?])\s+", s) if x.strip()]


def _pc_find(phrases, items):
    """(positive_hits, negated_hits) for the phrases across the items;
    a hit is negated when a negation phrase sits within 40 characters
    before or after it, with no sentence break in between."""
    pos, neg = [], []
    for it in items:
        matches = [m for ph in phrases for m in _pc_pattern(ph).finditer(it)]
        if not matches:
            continue
        affirmed = any(not (_PC_NEG_BEFORE.search(it[:m.start()])
                            or _PC_NEG_AFTER.search(it[m.end():]))
                       for m in matches)
        (pos if affirmed else neg).append(it.strip())
    return list(dict.fromkeys(pos)), list(dict.fromkeys(neg))


def procedure_confidence(procedure, facility, details=False):
    """Confidence (0-1) that `facility` offers `procedure` — the app-side
    port of procedure_confidence.py, scoring evidence in three levels
    (direct / supporting / inferred) against the facility's raw
    procedure, capability, description, equipment and specialties text
    from the explanation payload."""
    key = (procedure or "").strip().lower()
    lookup = _load_procedure_lookup()
    entry = lookup.get(key)
    if entry is None:               # the word may be a known synonym
        for value in lookup.values():
            if key in value["synonyms"]:
                entry = value
                break
    if entry is None:               # unknown term: match on the word itself
        entry = {"synonyms": [key], "specialty_keywords": [], "equipment": []}

    direct_items = [it for col in ("procedure", "capability", "description")
                    for it in _pc_items(facility.get(col))]
    direct, negated = _pc_find(entry["synonyms"], direct_items)
    equipment, _ = (_pc_find(entry["equipment"], _pc_items(facility.get("equipment")))
                    if entry["equipment"] else ([], []))
    specs = [s for s in _pc_items(facility.get("specialties"))
             if any(k in re.sub(r"[^a-z0-9]", "", s.lower())
                    for k in entry["specialty_keywords"])]
    specs = list(dict.fromkeys(specs))

    if direct:
        score = min(_PC_DIRECT_CAP,
                    _PC_DIRECT_BASE + _PC_DIRECT_PER_EXTRA * (len(direct) - 1))
        score += _PC_DIRECT_EQUIPMENT_BONUS if equipment else 0
        score += _PC_DIRECT_SPECIALTY_BONUS if specs else 0
        level = "direct"
    elif equipment:
        score = _PC_EQUIPMENT_ONLY + (_PC_EQUIPMENT_SPECIALTY_BONUS if specs else 0)
        level = "supporting"
    elif specs and not negated:
        score = _PC_SPECIALTY_ONLY
        level = "inferred"
    else:
        score = 0.0
        level = "negated" if negated else "none"
    score = round(min(score, 1.0), 4)

    if not details:
        return score
    return {"procedure": procedure, "score": score, "level": level,
            "direct": direct, "equipment": equipment, "specialties": specs,
            "negated": negated, "in_lookup": key in lookup}


# --- The three explainers, one per confidence value ----------------------

def _explain_existence(facility, stored):
    """Recreate existence_confidence from its raw inputs (the logistic
    model in existence_confidence.py) and narrate every term with its
    weight. Returns None when the facility has no payload row."""
    if "matches_tier" not in facility:
        return None

    def _scaled(raw, rng, fill):
        try:
            x = float(raw)
        except (TypeError, ValueError):
            x = fill                  # missing: worst case, as the script
        x = min(max(x, rng[0]), rng[1])
        return (x - rng[0]) / (rng[1] - rng[0])

    # Missing days = the WORST case in the original script, i.e. the
    # most-stale end of the range AFTER scaling — so fill with the range
    # top (4357.4 raw days -> x_days = 1.0), not with 1.0 raw days
    # (which would scale to 0.0, the best case, and inflate the score).
    x_days = _scaled(facility.get("days_raw"), _EXIST_DAYS_RANGE,
                     _EXIST_DAYS_RANGE[1])
    x_years = _scaled(facility.get("years_raw"), _EXIST_YEARS_RANGE, 0.0)
    has_addr = (facility.get("addr") or "0") == "1"
    try:
        tier = int(facility.get("matches_tier") or 0)
    except ValueError:
        tier = 0
    x_matches = min(3, max(0, tier)) / 3.0
    log_odds = -x_days + (1.0 if has_addr else 0.0) + x_matches + x_years
    recreated = 1 / (1 + math.exp(-log_odds))

    days = facility.get("days_raw") or ""
    recency = (f"Recent online activity: last seen {float(days):.0f} days ago "
               "(a place that has been active more recently is more likely "
               "to still be open)"
               if days else "Recent online activity: no date on record "
               "(counts against it - the dataset could not tell whether the "
               "place is still open)")
    years = facility.get("years_raw") or ""
    age = (f"Age: running for {float(years):.0f} years (older places are "
           "more likely to still be around)"
           if years else "Age: founding year unknown (neither helps nor "
           "hurts)")
    if tier > 0:
        matches = (f"Cross-checks: found in {tier} of the 3 outside "
                   "datasets the project compared against (the more "
                   "independent sources that agree, the better)")
    else:
        matches = ("Cross-checks: not found in the outside datasets it "
                   "was compared against (could not be double-confirmed)")
    addr_txt = ("Address: a real street address is on file (strong evidence "
                "the place exists)" if has_addr else
                "Address: no street address on file")
    if abs(recreated - (stored or 0.0)) <= 0.002:
        verdict = "the same score the dataset stores"
    else:
        verdict = (f"a close match to the {(stored or 0.0):.0%} the dataset "
                   "stores (some details have changed since the score was "
                   "first computed)")
    return (f"The dataset started at a 50/50 guess that this place exists, "
            f"then adjusted it with four checks: {addr_txt}; {recency}; "
            f"{matches}; {age}. Putting the four together gives "
            f"{recreated:.0%} - {verdict}.")


def _explain_quality(facility, stored):
    """Recreate quality_score from its components (the weighted mix in
    quality_score.py) and narrate each component's value, weight and
    meaning. Returns None when the facility has no payload row."""
    if "q_accreditation" not in facility:
        return None
    ftype = facility.get("type")
    if ftype not in _QUALITY_APPLIES:
        ftype = "other"
    applies = _QUALITY_APPLIES[ftype]

    def _val(name):
        try:
            return float(facility.get("q_" + name) or "")
        except ValueError:
            return None             # no data for this component

    # Plain-language labels for the seven components, in the order the
    # scoring script weights them.
    labels = {
        "accreditation": "Accreditation (India's NABH quality stamp)",
        "services": "Life-saving services",
        "specialties": "Breadth of specialties",
        "equipment": "Advanced equipment",
        "staffing": "Staffing",
        "rating": "Patient rating",
        "affiliation": "Affiliations",
    }
    def _num(v):
        """Render 1391.0 as '1391', 12.5 as '12.5', junk as '?'."""
        try:
            f = float(v)
        except (TypeError, ValueError):
            return v or "?"
        return f"{f:.0f}" if f == int(f) else f"{f:g}"

    def _plural(v, word):
        """'1 specialty' vs '27 specialties'."""
        try:
            return word if float(v) == 1 else word + "s"
        except (TypeError, ValueError):
            return word + "s"

    notes = {
        "accreditation": lambda: _NABH_LABEL.get(
            int(facility.get("nabh_grade") or 0), "an accreditation claim"),
        "services": lambda: "how confidently its records show it can "
                            "handle ICU, emergency care, CT, MRI and "
                            "dialysis",
        "specialties": lambda: (
            f"{_num(facility.get('n_spec'))} "
            f"{'specialty' if _num(facility.get('n_spec')) == '1' else 'specialties'}, "
            f"compared with other {ftype}s"),
        "equipment": lambda: (
            f"{_num(facility.get('n_eq'))} "
            f"{_plural(facility.get('n_eq'), 'type')} of advanced "
            "equipment named in its records"),
        "staffing": lambda: (
            f"{_num(facility.get('doctors'))} "
            f"{_plural(facility.get('doctors'), 'doctor')} and "
            f"{_num(facility.get('capacity'))} "
            f"{_plural(facility.get('capacity'), 'bed')}, compared with "
            f"other {ftype}s"),
        "rating": lambda: "a star rating quoted in its records",
        "affiliation": lambda: "teaching-hospital or "
                                "government-empanelment links",
    }
    parts, num, den = [], 0.0, 0.0
    for name, weight in _QUALITY_WEIGHTS.items():
        if name not in applies:
            continue               # not scored for this facility type
        value = _val(name)
        if value is None:
            parts.append(f"{labels[name]}: none on record (its "
                         f"{weight:.0%} share is shared out among the rest)")
            continue
        num += value * weight
        den += weight
        note = notes[name]()
        parts.append(f"{labels[name]}, worth {weight:.0%} of the score: "
                     f"{value:.2f} out of 1" +
                     (f" - {note}" if note else ""))
    if not den:
        return None
    recreated = num / den

    text = ("The dataset grades this " + ftype + " on what good ones have, "
            "though the grades do not all count equally: " +
            "; ".join(parts) + ".")
    if facility.get("approx") == "1":
        text += (f" Putting those together gives {recreated:.0%}; the "
                 f"dataset stores {(stored or 0.0):.0%} - some details "
                 "have changed since the score was first computed, so "
                 "this breakdown is approximate.")
    elif abs(recreated - (stored or 0.0)) <= 0.002:
        text += (f" Putting those together gives {recreated:.0%} - the "
                 "same as the score stored in the dataset.")
    else:
        text += (f" Putting those together gives {recreated:.0%}; the "
                 f"dataset stores {(stored or 0.0):.0%}.")
    return text


def _explain_procedures(terms, facility, limit=8):
    """Run procedure_confidence() live for the AI-identified services and
    explain the evidence level behind each score."""
    if not terms or "procedure" not in facility:
        return None
    lines = []
    for term in terms[:limit]:
        d = procedure_confidence(term, facility, details=True)
        if d["level"] == "direct":
            ev = (d["direct"] or [""])[0]
            why = f"very likely - it is named in this place's own records (\"{ev[:70]}\")"
            if d["equipment"]:
                why += ", and the equipment it needs is listed"
            if d["specialties"]:
                why += ", and it has a matching department"
        elif d["level"] == "supporting":
            ev = (d["equipment"] or [""])[0]
            why = f"likely - the equipment it needs is listed (\"{ev[:70]}\")"
            if d["specialties"]:
                why += ", and it has a matching department"
        elif d["level"] == "inferred":
            why = (f"possible - only a department that performs it is listed "
                   f"(\"{(d['specialties'] or [''])[0][:50]}\"), which is weaker evidence")
        elif d["level"] == "negated":
            why = "unlikely - it is mentioned, but the records say it is not offered"
        else:
            why = "unknown - no sign of it in this place's records"
        lines.append(f"{term}: {d['score']:.0%} - {why}")
    return lines or None


def _to_float(value):
    """'0.75' -> 0.75; anything else (missing/blank) -> 0.0."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _load_medical_facilities():
    """Read facilities_medical.csv once and cache it: every facility with
    its lat/lng and a lowercase capability blob (specialties + equipment
    + procedures) to match patient needs against."""
    global _MEDICAL_FACILITIES
    if _MEDICAL_FACILITIES is None:
        path = os.path.join(app.root_path, "facilities_medical.csv")
        try:
            with open(path, newline="", encoding="utf-8") as f:
                rows = []
                for raw in csv.DictReader(f):
                    try:
                        lat = float(raw.get("lat") or "")
                        lng = float(raw.get("lng") or "")
                    except ValueError:            # missing/not-a-number coords
                        continue
                    facility_row = {
                        "lat": lat,
                        "lng": lng,
                        "name": raw.get("name") or "",
                        "type": (raw.get("type") or "").lower(),
                        "city": raw.get("city") or "",
                        "address": raw.get("address") or "",
                        "capabilities": (raw.get("capabilities") or "").lower(),
                        # Existence confidence / quality score and
                        # contact details from the enriched dataset
                        # (0.0 / "" when a value is missing).
                        "existence_confidence": _to_float(raw.get("existence_confidence")),
                        "quality_score": _to_float(raw.get("quality_score")),
                        "phone": raw.get("phone") or "",
                        "email": raw.get("email") or "",
                        "website": raw.get("website") or "",
                        "doctors": raw.get("doctors") or "",
                        "capacity": raw.get("capacity") or "",
                        # Pre-split word set: word-order-free term matching.
                        # Split on non-alphanumerics — the blob separates
                        # terms with ";", and a word like "obstetrics;"
                        # must still match the term word "obstetrics".
                        "words": frozenset(re.findall(
                            r"[a-z0-9]+",
                            (raw.get("capabilities") or "").lower())),
                    }
                    # Attach the scoring-script inputs (section 3c) this
                    # row's explanations recreate the scores from. A
                    # couple of facilities are missing from the payload —
                    # their cards simply skip the score breakdown.
                    _ex = _load_score_explanations().get(
                        (facility_row["name"], facility_row["city"],
                         raw.get("state") or "", round(lat, 6), round(lng, 6)))
                    if _ex:
                        facility_row.update({
                            k: _ex.get(k) or ""
                            for k in ("days_raw", "years_raw", "addr",
                                      "matches_tier", "nabh_grade", "approx",
                                      "n_spec", "n_eq",
                                      "q_accreditation", "q_services",
                                      "q_specialties", "q_equipment",
                                      "q_staffing", "q_rating",
                                      "q_affiliation",
                                      "procedure", "capability",
                                      "description", "equipment",
                                      "specialties")})
                    rows.append(facility_row)
            _MEDICAL_FACILITIES = rows
        except FileNotFoundError:
            # No capability file yet — ranking falls back to data.csv.
            _MEDICAL_FACILITIES = []
    return _MEDICAL_FACILITIES


def _chat_json(messages, max_tokens=500):
    """
    Send a chat request to the Unity Gateway model service and return the
    model's reply parsed as a JSON object, or None on any failure.

    Authentication uses the app's own OAuth credential, which the
    Databricks Apps runtime injects — WorkspaceClient() picks it up
    automatically; no tokens are ever stored or typed.
    """
    try:
        from databricks.sdk import WorkspaceClient   # lazy: only needed here
        workspace = WorkspaceClient()
        response = requests.post(
            f"{workspace.config.host.rstrip('/')}/ai-gateway/mlflow/v1/chat/completions",
            headers={**workspace.config.authenticate(),
                     "Content-Type": "application/json"},
            json={
                "model": MODEL_SERVICE_NAME,
                "messages": messages,
                "temperature": 0.0,        # triage: consistent, repeatable answers
                "max_tokens": max_tokens,
            },
            timeout=45,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
    except Exception:
        return None
    # Models sometimes wrap JSON in ```fences``` or add chatter — take the
    # outermost {...} and parse exactly that.
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except ValueError:
        return None


_INTERPRET_SYSTEM_PROMPT = """You are the triage classifier inside India Care Map, an app that helps people in India find the right hospital.

The user will describe a patient's situation in their own words — often long, indirect or emotional, and the person writing may not be the patient (a relative, an NGO worker, a neighbour). Work out what medical services the patient most likely needs.

Rules:
- Focus on the CLINICAL need. Ignore logistics (money, transport, who is asking).
- If a complication is mentioned or hinted at, include the service that handles it (e.g. a risky delivery needs obstetrics AND neonatal intensive care).
- Give 3 to 8 concrete services. Each lowercase, 1-4 words, named the way an Indian hospital department or equipment list would name it (e.g. "obstetrics and gynecology", "nicu", "dialysis", "trauma care", "cardiology", "ct scan", "blood bank", "burns unit").
- urgency is exactly one of: routine, urgent, emergency.
- patient_summary is one short sentence saying what the patient needs.

Reply with ONLY this JSON object, no other text:
{"services": ["..."], "urgency": "routine|urgent|emergency", "patient_summary": "..."}"""


def interpret_problem(problem_text):
    """
    Turn a free-text patient story into structured medical needs via the
    AI model service. Cached per text. Returns a dict:
        {"services": [...], "urgency": "...", "patient_summary": "..."}
    On failure the services list is empty and ranking falls back to pure
    distance — the map must keep working even when the model is down.
    """
    key = problem_text.strip().lower()
    if key in _INTERPRETATION_CACHE:
        return _INTERPRETATION_CACHE[key]

    result = _chat_json([
        {"role": "system", "content": _INTERPRET_SYSTEM_PROMPT},
        {"role": "user", "content": problem_text[:2000]},
    ]) or {}

    services = [str(s).strip().lower() for s in result.get("services") or []
                if str(s).strip()][:8]
    urgency = (result.get("urgency") or "").strip().lower()
    if urgency not in ("routine", "urgent", "emergency"):
        urgency = ""
    interpretation = {
        "services": services,
        "urgency": urgency,
        "patient_summary": (result.get("patient_summary") or "").strip(),
    }
    _INTERPRETATION_CACHE[key] = interpretation   # cache failures too —
    return interpretation                          # retrying costs seconds


# Small words that carry no clinical meaning — ignored when matching a
# needed service against a facility's word set.
_STOPWORDS = {"and", "or", "of", "the", "in", "for", "with", "a", "an", "to"}


def _term_matches(term, words):
    """
    True when every meaningful word of a needed service appears in the
    facility's capability words — WORD-ORDER FREE, so "obstetrics and
    gynecology" matches a facility listing "gynecology and obstetrics"
    (the dataset stores both orders). Simple singular/plural variants are
    accepted ("nicu beds" / "nicu bed").
    """
    needed = [w for w in term.split() if w not in _STOPWORDS]
    if not needed:
        return False
    for w in needed:
        stem = w.rstrip("s")
        if not (w in words or stem in words or (w + "s") in words):
            return False
    return True


def matched_services(terms, facility):
    """Which of the AI-identified needed services does this facility's
    capability list actually cover?"""
    words = facility.get("words")
    if words is None:
        words = set(re.findall(
            r"[a-z0-9]+", (facility.get("capabilities") or "").lower()))
    return [t for t in terms if _term_matches(t, words)]


def _equipped_count(terms):
    """
    How many of the AI-identified needs a facility must cover to count
    as EQUIPPED for the problem: at least half of them (and at least
    one). "Equipped" = the serious version; covering just one need is
    only "could possibly deal with it".
    """
    if not terms:
        return 0
    return max(1, math.ceil(len(terms) * EQUIPPED_FRACTION))


def _pct(confidence):
    """0.75 -> '75%' for the human-readable recommendation reasons."""
    return f"{round((confidence or 0) * 100)}%"


def recommend_facilities(user_lat, user_lng, interpretation=None):
    """
    Pick which facilities to recommend, following urgency-based rules
    (constants at the top of this file):

    Emergency — the closest HOSPITAL within 60 miles that could deal
      with the problem (covers at least one need) and is 90%+ certain
      to actually exist.
    Urgent — two facilities within 60 miles: the closest EQUIPPED one
      (75%+ existence confidence) and, separately, the closest one that
      is 90%+ certain to exist even if less equipped.
    Routine — within 200 miles: one close+equipped but not certain, one
      close+certain but possibly not equipped, and one certain+equipped
      but farther away. A single facility that is close AND equipped
      AND certain is recommended on its own.

    Returns (radius_km, cards). Each card carries the facility's
    contact details, existence confidence and quality score, the needs
    it matched, and a plain-English REASON: which keywords matched, how
    urgent the case is, and which radius/confidence rule picked it —
    including any relaxation, stated honestly (e.g. when nothing inside
    the radius meets the strict rule).
    """
    terms = (interpretation or {}).get("services") or []
    urgency = (interpretation or {}).get("urgency") or "routine"
    equipped_need = _equipped_count(terms)

    radius_km = {"emergency": EMERGENCY_RADIUS_KM,
                 "urgent": URGENT_RADIUS_KM}.get(urgency, ROUTINE_RADIUS_KM)

    # Every facility inside the radius, closest first, with the needs it
    # covers attached — one sorted pool that all the "closest facility
    # that ..." rules below draw from.
    pool = []
    for facility in _load_medical_facilities():
        distance = _distance_km(user_lat, user_lng,
                                facility["lat"], facility["lng"])
        if distance <= radius_km:
            pool.append((distance, facility,
                         matched_services(terms, facility)))
    pool.sort(key=lambda entry: entry[0])

    def pick(predicate):
        """The closest pool entry satisfying predicate(distance, facility,
        matched) — or None when nothing in the radius qualifies."""
        for entry in pool:
            if predicate(entry[0], entry[1], entry[2]):
                return entry
        return None

    def conf(facility):
        return facility.get("existence_confidence") or 0.0

    def need_str(facility, matched):
        return f"{len(matched)}/{len(terms)}"

    def card(entry, reason):
        distance, facility, matched = entry
        # Recreate the dataset's confidence values from the scoring
        # scripts and explain the weights behind them (section 3c), so
        # every card shows the derivation behind each score.
        explain = {
            "existence": _explain_existence(
                facility, facility.get("existence_confidence")),
            "quality": _explain_quality(
                facility, facility.get("quality_score")),
            "procedures": _explain_procedures(terms, facility),
        }
        explain = {k: v for k, v in explain.items() if v} or None
        return {
            "lat": facility["lat"], "lng": facility["lng"],
            "name": facility["name"],
            "type": facility.get("type") or "",
            "city": facility.get("city") or "",
            "address": facility.get("address") or "",
            "distance_km": round(distance, 1),
            "matched": matched,
            "existence_confidence": facility.get("existence_confidence"),
            "quality_score": facility.get("quality_score"),
            "phone": facility.get("phone") or "",
            "email": facility.get("email") or "",
            "website": facility.get("website") or "",
            "reason": reason,
            "explain": explain,
        }

    # No AI read (model down, or text too vague): show the closest few
    # and say so — never leave the user with nothing.
    if not terms:
        return radius_km, [
            card(entry, "AI interpretation was unavailable, so this is one of "
                        f"the closest facilities ({_pct(conf(entry[1]))} "
                        "existence confidence).")
            for entry in pool[:3]]

    # --- Emergency: closest equipped hospital that definitely exists ----
    if urgency == "emergency":
        entry = (pick(lambda d, f, m: f.get("type") == "hospital"
                      and m and conf(f) >= EMERGENCY_CONFIDENCE)
                 or pick(lambda d, f, m: f.get("type") == "hospital"
                         and conf(f) >= EMERGENCY_CONFIDENCE)
                 or pick(lambda d, f, m: f.get("type") == "hospital" and m)
                 or pick(lambda d, f, m: f.get("type") == "hospital"))
        if entry is None and pool:
            entry = pool[0]
        if entry is None:
            return radius_km, []
        facility, matched = entry[1], entry[2]
        strict = (facility.get("type") == "hospital" and matched
                  and conf(facility) >= EMERGENCY_CONFIDENCE)
        if strict:
            reason = ("Life-threatening: the closest hospital within 60 miles "
                      "that can handle this (covers "
                      f"{need_str(facility, matched)} identified needs) and "
                      f"is {_pct(conf(facility))} certain to actually exist.")
        else:
            reason = ("Life-threatening: no hospital within 60 miles is both "
                      "equipped for this and 90%+ certain to exist, so the "
                      "closest fallback is shown — covers "
                      f"{need_str(facility, matched)} needs, "
                      f"{_pct(conf(facility))} existence confidence.")
        return radius_km, [card(entry, reason)]

    # --- Urgent: one equipped pick, one existence pick -------------------
    if urgency == "urgent":
        equipped = (pick(lambda d, f, m: len(m) >= equipped_need
                         and conf(f) >= URGENT_EQUIPPED_CONFIDENCE)
                   or pick(lambda d, f, m: m
                           and conf(f) >= URGENT_EQUIPPED_CONFIDENCE)
                   or pick(lambda d, f, m: len(m) >= equipped_need))
        certain = (pick(lambda d, f, m: conf(f) >= CERTAIN_CONFIDENCE)
                  or pick(lambda d, f, m: conf(f) >= 0.80))
        if certain is None and pool:
            # Nothing inside the radius is 90%+ certain — show the most
            # certain one there is, and say so.
            certain = max(pool, key=lambda e: conf(e[1]))
        cards = []
        if equipped:
            cards.append(card(equipped,
                "Urgent: the closest facility within 60 miles equipped for "
                f"this — covers {need_str(equipped[1], equipped[2])} "
                f"identified needs, {_pct(conf(equipped[1]))} existence "
                "confidence (75%+ required)."))
        if certain:
            if equipped and certain[1] is equipped[1]:
                cards = [card(equipped,
                    "Urgent: this is BOTH the closest equipped facility "
                    "within 60 miles (covers "
                    f"{need_str(equipped[1], equipped[2])} needs) and the "
                    f"closest one {_pct(conf(equipped[1]))} certain to "
                    "actually exist.")]
            elif conf(certain[1]) >= CERTAIN_CONFIDENCE:
                cards.append(card(certain,
                    "Urgent: the closest facility within 60 miles that is "
                    f"{_pct(conf(certain[1]))} certain to actually exist "
                    "(90%+ rule), even though it covers only "
                    f"{need_str(certain[1], certain[2])} needs."))
            else:
                cards.append(card(certain,
                    "Urgent: no facility within 60 miles is 90%+ certain to "
                    "exist — this is the most certain one available "
                    f"({_pct(conf(certain[1]))}), covering "
                    f"{need_str(certain[1], certain[2])} needs."))
        return radius_km, cards

    # --- Routine: all-rounder, or one of each kind ------------------------
    allrounder = pick(lambda d, f, m: d <= CLOSE_KM
                      and len(m) >= equipped_need
                      and conf(f) >= CERTAIN_CONFIDENCE)
    if allrounder:
        return radius_km, [card(allrounder,
            "Recommended on its own: close "
            f"({allrounder[0]:.0f} km), covers "
            f"{need_str(allrounder[1], allrounder[2])} identified needs, and "
            f"is {_pct(conf(allrounder[1]))} certain to actually exist — it "
            "satisfies all three requirements.")]

    # Each slot tries its strict rule first, then relaxations — the
    # reason text always reflects the rule that actually fired.
    slots = [
        ("Close + equipped (not certain)", [
            (lambda d, f, m: d <= CLOSE_KM and len(m) >= equipped_need
             and conf(f) < CERTAIN_CONFIDENCE,
             lambda e: f"Close ({e[0]:.0f} km) and covers "
                       f"{need_str(e[1], e[2])} identified needs, but only "
                       f"{_pct(conf(e[1]))} certain to actually exist — "
                       "verify before travelling."),
            (lambda d, f, m: d <= CLOSE_KM and len(m) >= equipped_need,
             lambda e: f"Close ({e[0]:.0f} km) and covers "
                       f"{need_str(e[1], e[2])} identified needs "
                       f"({_pct(conf(e[1]))} existence confidence)."),
            (lambda d, f, m: len(m) >= equipped_need,
             lambda e: "The closest equipped facility within 200 miles — "
                       f"{e[0]:.0f} km away, covers "
                       f"{need_str(e[1], e[2])} identified needs."),
        ]),
        ("Close + certain (maybe not equipped)", [
            (lambda d, f, m: d <= CLOSE_KM and conf(f) >= CERTAIN_CONFIDENCE,
             lambda e: f"Close ({e[0]:.0f} km) and {_pct(conf(e[1]))} certain "
                       "to actually exist, though it covers only "
                       f"{need_str(e[1], e[2])} identified needs."),
            (lambda d, f, m: conf(f) >= CERTAIN_CONFIDENCE,
             lambda e: "The closest 90%+-certain facility — "
                       f"{e[0]:.0f} km away (covers "
                       f"{need_str(e[1], e[2])} identified needs)."),
        ]),
        ("Certain + equipped (not close)", [
            (lambda d, f, m: d > CLOSE_KM and len(m) >= equipped_need
             and conf(f) >= CERTAIN_CONFIDENCE,
             lambda e: f"{_pct(conf(e[1]))} certain to exist and covers "
                       f"{need_str(e[1], e[2])} identified needs, at "
                       f"{e[0]:.0f} km — farther away but a safe choice."),
            (lambda d, f, m: d > CLOSE_KM and len(m) >= equipped_need,
             lambda e: f"Equipped for this (covers "
                       f"{need_str(e[1], e[2])} identified needs) at "
                       f"{e[0]:.0f} km."),
        ]),
    ]
    seen, cards = [], []
    for _label, attempts in slots:
        for predicate, build_reason in attempts:
            entry = pick(predicate)
            if entry and entry[1] not in seen:
                seen.append(entry[1])
                cards.append(card(entry, build_reason(entry)))
                break
    return radius_km, cards


# Generic words that appear in many facility names — matching a typo
# against them alone would give meaningless suggestions like "Hospital".
_GENERIC_NAME_WORDS = {
    "hospital", "clinic", "medical", "centre", "center", "care",
    "research", "institute", "memorial", "college", "nursing", "home",
    "general", "specialty", "speciality", "super", "pvt", "ltd",
    "private", "limited", "the", "and", "of", "trust", "association",
    "mission", "society", "group", "multispeciality", "multispecialty",
}


def suggest_addresses(address, limit=4):
    """
    When an address can't be geocoded, look for near-miss spellings among
    ALL location names in data.csv — every city, state and facility name
    (e.g. "Mumbia" -> "Mumbai", "Chenai" -> "Chennai").

    Returns (best_score, suggestions): the similarity of the best match
    (0.0 when nothing matched) and up to `limit` suggestions, best first.
    Each suggestion is a dict {"text", "lat", "lng"} — the coordinates
    come from data.csv itself, so clicking a suggestion can jump straight
    to that place without geocoding its name (Nominatim usually can't
    resolve private facility names). The score lets the caller decide
    whether to also try harder fixes.
    """
    # Collect every distinct location name, properly cased, together with
    # the average of the coordinates seen for it. For longer facility
    # names we also match word by word, so a typo can find
    # "Some Hospital, Chhatapur" via its "Chhatapur" part.
    entries = {}   # lowercased name -> {display, words, lat_sum, lng_sum, n}
    for point in _read_data_points():
        for value in (point.get("city"), point.get("state"), point.get("name")):
            value = (value or "").strip()
            if not value:
                continue
            lower = value.lower()
            entry = entries.get(lower)
            if entry is None:
                words = [w for w in lower.replace(",", " ").split()
                         if len(w) > 3 and w not in _GENERIC_NAME_WORDS]
                entry = {"display": value, "words": words,
                         "lat_sum": 0.0, "lng_sum": 0.0, "n": 0}
                entries[lower] = entry
            entry["lat_sum"] += point["lat"]
            entry["lng_sum"] += point["lng"]
            entry["n"] += 1
    if not entries:
        return 0.0, []

    # Compare the whole typed text and each of its words against each
    # entry and each of the entry's words, keeping the best similarity
    # (0 = nothing alike, 1 = identical).
    typed = address.lower()
    typed_words = [w for w in typed.replace(",", " ").split() if len(w) > 3]
    scored = []
    for lower, entry in entries.items():
        best = difflib.SequenceMatcher(None, typed, lower).ratio()
        for typed_word in typed_words:
            best = max(best, difflib.SequenceMatcher(None, typed_word, lower).ratio())
            for entry_word in entry["words"]:
                best = max(best,
                           difflib.SequenceMatcher(None, typed_word, entry_word).ratio())
        if best >= 0.6:
            scored.append((best, lower))

    # Best score first; on ties prefer the SHORTER name — "Mumbai" is a
    # better suggestion than "Vile Parle East, Mumbai".
    scored.sort(key=lambda item: (-item[0], len(entries[item[1]]["display"]),
                                  entries[item[1]]["display"]))

    suggestions, seen = [], set()
    for _, lower in scored:
        if lower in seen:
            continue
        seen.add(lower)
        entry = entries[lower]
        suggestions.append({
            "text": entry["display"],
            "lat": round(entry["lat_sum"] / entry["n"], 5),
            "lng": round(entry["lng_sum"] / entry["n"], 5),
        })
        if len(suggestions) == limit:
            break
    return (scored[0][0] if scored else 0.0), suggestions


_LOCALITY_SUFFIX = re.compile(
    r"\b(road|rd|street|st|marg|nagar|colony|sector|area|east|west|north|south|"
    r"village|gaon|town|market|bazaar|chowk|circle|park)\b", re.IGNORECASE)
_ADDRESS_NOISE = re.compile(
    r"^(flat|room|door|no\.?|number|block|wing|floor|plot|building|apartment|"
    r"apt\.?|society|tower|complex|phase|stage|house|near|opp\.?|opposite|"
    r"behind|beside)\b", re.IGNORECASE)


def _address_rungs(address, max_rungs=3):
    """
    Degraded versions of a failed address, most promising first — chosen
    by WHAT each comma-separated part looks like, not by its position
    (people type addresses in any order: "Mumbai, Andheri East, MG Road").

    Rung 1 keeps every part that isn't obvious noise (flat numbers,
    building names); rung 2 keeps only the geographic core (parts that
    match a city/state in data.csv, contain a pincode, or look like a
    locality); rung 3 keeps the two most geographic parts (usually city
    + pincode, which the geocoder resolves almost every time).
    """
    parts = [p.strip() for p in address.split(",") if p.strip()]
    if len(parts) < 2:
        return []

    known = {v.lower() for p in _read_data_points()
             for v in (p.get("city"), p.get("state")) if v}
    pin = re.compile(r"\b\d{6}\b")

    def weight(part):
        lower = part.lower()
        if lower in known or pin.search(part):
            return 3   # a city/state name or a pincode — geocodable gold
        if _LOCALITY_SUFFIX.search(lower):
            return 2   # looks like a street/locality
        if _ADDRESS_NOISE.match(lower):
            return 0   # flat number, building, floor — geocoder junk
        return 1       # might mean something; keep at first, drop later

    scored = [(weight(p), p) for p in parts]
    rung1 = [p for g, p in scored if g >= 1] or parts
    rung2 = [p for g, p in scored if g >= 2]
    rung3 = [p for _, p in sorted(scored, key=lambda t: -t[0])[:2]]

    rungs, seen = [], set()
    for cand in (rung1, rung2, rung3):
        query = ", ".join(cand)
        if len(cand) >= 2 and query.lower() not in seen:
            seen.add(query.lower())
            rungs.append(query)
        if len(rungs) == max_rungs:
            break
    return rungs


def _typo_variants(address, max_attempts=5):
    """
    Try deleting one letter at a time from the typed address, re-geocoding
    each variant — catches the classic "typed an extra letter" mistake
    ("Chhartapur" -> "Chhatapur") even when the real place is not in our
    dataset at all. Returns the geocoded display name of the first variant
    that resolves somewhere in India, or None.

    Each attempt is one extra Nominatim call (~1.2 s), so this only runs
    when the typed address has already failed to geocode AND nothing in
    data.csv is a confident match.
    """
    tried = set()
    attempts = 0
    for i in range(len(address)):
        variant = address[:i] + address[i + 1:]
        if not variant or variant == address or variant in tried:
            continue
        tried.add(variant)
        location = geocode(variant)      # sleeps 1.1 s (Nominatim limit)
        attempts += 1
        if location is not None:
            # Keep the suggestion readable: place, area, state — and send
            # its coordinates too, so clicking it never needs geocoding.
            return {
                "text": ", ".join(location["display_name"].split(", ")[:3]),
                "lat": location["lat"],
                "lng": location["lng"],
            }
        if attempts >= max_attempts:
            break
    return None


# ---------------------------------------------------------------------------
# 4. DATABRICKS LOOKUP — find a matching issue in the table.
# ---------------------------------------------------------------------------


def get_connection():
    """
    Open a connection to the app's SQL warehouse.

    DatabricksConfig() automatically finds the OAuth client credentials
    that Databricks Apps injects into the environment, so we never handle
    tokens ourselves. We still need the warehouse's HTTP path (set via env).
    """
    cfg = DatabricksConfig()
    return dbsql.connect(
        server_hostname=cfg.host,
        http_path=WAREHOUSE_HTTP_PATH,
        credentials_provider=lambda: cfg.authenticate,
    )


def lookup_issue(address, issue):
    """
    Search the table for a row whose issue TYPE matches the issue the user
    typed, OR whose LOCATION matches the address (case-insensitive "contains").

    Returns the matching issue statement, or None when nothing matches.

    Note on safety: the values typed by the user are passed as `?` parameters,
    never pasted into the SQL text — this prevents SQL injection. Only the
    table/column NAMES (which come from admin-set env vars, not user input)
    are placed into the SQL string.
    """
    sql_text = (
        f"SELECT `{STATEMENT_COL}` "
        f"FROM {fully_qualified_table()} "
        f"WHERE lower(`{ISSUE_TYPE_COL}`) LIKE lower(?) "
        f"   OR lower(`{LOCATION_COL}`) LIKE lower(?) "
        f"LIMIT 1"
    )
    # Wrapping the text in '%' makes LIKE behave as a "contains" match.
    params = [f"%{issue}%", f"%{address}%"]

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql_text, params)
            row = cursor.fetchone()
    return row[0] if row else None


def save_submission(address, issue, lat, lng):
    """
    Optional feature (SAVE_SUBMISSIONS=true): INSERT the submitted address,
    issue and coordinates as a new row in the table.

    The insert uses the same configurable columns as the lookup. If your
    table stores the issue text differently, adjust the column env vars.
    Errors are caught by the caller so the map keeps working even if the
    write fails.
    """
    sql_text = (
        f"INSERT INTO {fully_qualified_table()} "
        f"(`{LOCATION_COL}`, `{ISSUE_TYPE_COL}`, `{STATEMENT_COL}`, `{LAT_COL}`, `{LNG_COL}`) "
        f"VALUES (?, ?, ?, ?, ?)"
    )
    # Save the typed address, the typed issue (as both type and statement
    # text), and the geocoded coordinates.
    params = [address, issue, issue, lat, lng]
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql_text, params)


# ---------------------------------------------------------------------------
# 5. FLASK ROUTES — the URLs the app responds to.
# ---------------------------------------------------------------------------


@app.route("/")
def home():
    """Show the map page."""
    return render_template("index.html")


@app.route("/api/submit", methods=["POST"])
def submit():
    """
    Handle the form submission from the map page.

    The page sends JSON like {"address": "...", "issue": "..."}.
    We reply with the geocoded location plus the matching issue statement,
    or an error message the page displays to the user.
    """
    data = request.get_json(silent=True) or {}
    address = (data.get("address") or "").strip()
    issue = (data.get("issue") or "").strip()

    # Both boxes must be filled in.
    if not address or not issue:
        return jsonify({"error": "Please type both an address and an issue."}), 400

    # Optional: the page can send known coordinates for the address —
    # used when the user clicks a "Did you mean" suggestion, which carry
    # the place's coordinates from data.csv. Then no geocoding is needed
    # (and none would work: the geocoder can't resolve facility names).
    try:
        given_lat = float(data.get("lat"))
        given_lng = float(data.get("lng"))
        # Ignore anything outside India's rough box.
        if not (5 <= given_lat <= 40 and 66 <= given_lng <= 100):
            given_lat = given_lng = None
    except (TypeError, ValueError):
        given_lat = given_lng = None

    # Step 1: convert the address to coordinates (India only), unless the
    # page already sent trustworthy coordinates.
    location = None
    pool = []              # every candidate seen — used for ambiguous queries
    tried_queries = set()  # queries already sent to Nominatim

    def fetch(query):
        tried_queries.add(query.lower())
        return geocode_candidates(query, limit=6)

    if given_lat is not None:
        location = {"lat": given_lat, "lng": given_lng, "display_name": address}
    else:
        # Fetch SEVERAL candidates, not just one: a vague query can match
        # many places, and silently picking the first is how "autocorrect"
        # sent the user to one arbitrary spot among the gazillion.
        candidates = fetch(address)
        pool.extend(candidates)
        if candidates and not _ambiguous_query(address, candidates):
            location = candidates[0]

        if location is None:
            # Full street addresses often fail: OpenStreetMap's India
            # data rarely includes flat/building-level detail, so
            # "Flat 1203, Green Heights Apartment, ..." finds nothing.
            # Retry with degrading KEYWORD versions of the address (see
            # _address_rungs): keep the geocodable parts — locality,
            # city, state, pincode — and drop flat/building noise. Parts
            # are picked by what they look like, not their position,
            # because addresses aren't always typed specific-to-general.
            # A rung that resolves confidently wins; an ambiguous rung's
            # candidates join the pool for the suggestion list below.
            for rung in _address_rungs(address):
                if rung.lower() in tried_queries:
                    continue
                rung_candidates = fetch(rung)
                if rung_candidates and not _ambiguous_query(rung, rung_candidates):
                    location = rung_candidates[0]
                    break
                pool.extend(rung_candidates)

        if location is None and len(_distinct_places(pool)) < 2:
            # The query is vague but we hold few options: geocode its
            # longest keyword on its own ("C. S. Lewis" -> "lewis") —
            # that is what actually surfaces the many Lewis places,
            # instead of one lucky match on the full string.
            tokens = _query_tokens(address)
            keyword = max(tokens, key=len) if tokens else ""
            if len(keyword) >= 4 and keyword.lower() not in tried_queries:
                pool.extend(fetch(keyword))

        if location is None:
            options = _distinct_places(pool)
            if options:
                # Ambiguous: offer the distinct matches as clickable
                # suggestions (each carries its own coordinates) instead
                # of flying to one arbitrary pick.
                return jsonify({
                    "error": ("That address matches several places — "
                              "did you mean one of these?"),
                    "did_you_mean": [
                        {"text": _short_display(c["display_name"]),
                         "lat": c["lat"], "lng": c["lng"]}
                        for c in options[:5]
                    ],
                }), 404
            # Still nothing. Suggest near-miss spellings from every
            # location name in data.csv (cities, states, facility names)
            # — instant, no geocoding needed. The slower single-letter
            # deletion fixes are fetched separately by the page from
            # /api/suggest-variant, so they never hold up this answer.
            _, suggestions = suggest_addresses(address)
            return jsonify({
                "error": "Address not found in India. Try adding the city, state or pincode.",
                "did_you_mean": suggestions,
            }), 404

    # Step 1b: ask the AI to read the patient's problem, then pick the
    # facilities to recommend — which ones and how many depends on how
    # urgent the case is and how certain the dataset is that each place
    # actually exists (see recommend_facilities).
    interpretation = interpret_problem(issue)
    search_radius_km, recommendations = recommend_facilities(
        location["lat"], location["lng"], interpretation)

    # Step 2: look up a matching issue statement in the Databricks table.
    # If no table is configured yet, or the table can't be reached, we still
    # show the map — just without a statement, and with a note in the panel.
    issue_statement = None
    sql_error = None
    if not (ISSUE_CATALOG and ISSUE_SCHEMA and ISSUE_TABLE):
        # No table configured yet (the ISSUE_* env vars are empty) — skip
        # the query SILENTLY: the panel already shows "No matching issue
        # found.", and a setup note under the recommendations only
        # confuses readers (user request: never display it).
        pass
    else:
        try:
            issue_statement = lookup_issue(address, issue)
        except Exception as exc:
            sql_error = f"Could not query the issues table: {exc}"

    # Step 3 (optional): save this submission as a new row.
    # Saving needs the table too — without one there is nothing to
    # INSERT into, and a second "not configured" note would just repeat
    # the lookup note above.
    save_error = None
    if SAVE_SUBMISSIONS and (ISSUE_CATALOG and ISSUE_SCHEMA and ISSUE_TABLE):
        try:
            save_submission(address, issue, location["lat"], location["lng"])
        except Exception as exc:
            save_error = f"Could not save the submission: {exc}"

    # Step 4: send everything the page needs back as one JSON answer.
    return jsonify({
        "lat": location["lat"],
        "lng": location["lng"],
        "display_name": location["display_name"],
        "issue_statement": issue_statement if issue_statement else "No matching issue found.",
        "ai": interpretation,
        "recommendations": recommendations,
        "search_radius_km": round(search_radius_km, 1),
        "sql_error": sql_error,
        "save_error": save_error,
    })


@app.route("/api/suggest-variant", methods=["POST"])
def suggest_variant():
    """
    Slower spelling-fix search for addresses that failed to geocode:
    try deleting one letter at a time and re-geocoding each variant
    (see _typo_variants). The page calls this in the BACKGROUND after it
    already showed the instant data.csv suggestions, so the few seconds
    of geocoding never block the user — the fix simply appears as an
    extra "Did you mean" button when it's found.
    """
    data = request.get_json(silent=True) or {}
    address = (data.get("address") or "").strip()
    if not address:
        return jsonify({"suggestion": None})

    # Same gate as before: only try when the data.csv names have no
    # confident match for what was typed.
    best_score, _ = suggest_addresses(address)
    if best_score >= 0.85:
        return jsonify({"suggestion": None})

    return jsonify({"suggestion": _typo_variants(address)})


@app.after_request
def _no_cache_page(response):
    """Ask the browser to never serve a stale page after a redeploy.

    Without this, a Chrome tab that still holds the old index.html keeps
    running the previous version's JavaScript until a hard refresh —
    which makes newly deployed fixes look like they don't work.
    """
    if request.path == "/":
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


if __name__ == "__main__":
    # Databricks Apps sets a PORT variable for us; use it if present.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
