# datathon-fall-2026

India Care Map — a Databricks app that maps Indian healthcare facilities and recommends the nearest appropriate one for a described medical problem.

## App

The full application lives in [`india-issue-map/`](india-issue-map/):

* `app.py` — Flask backend (geocoding, AI issue interpretation, urgency-based recommendations, scoring ports)
* `templates/index.html` — Leaflet/OSM frontend
* `app.yaml` — app configuration (env vars)
* `requirements.txt`
* `facilities_medical.csv`, `facilities_explain.csv.gz`, `procedure_lookup.csv`, `data.csv` — bundled datasets

## Scoring

Three reverse-engineered scoring scripts are ported into the app and reproduce the dataset's stored scores exactly:

* **existence_confidence** — logistic model: address, online-activity recency, cross-dataset matches, facility age
* **quality_score** — weighted components (NABH accreditation 35%, services 20%, specialties 18%, equipment 10%, staffing 10%, affiliations 5%, rating 2%), renormalized over available data
* **procedure_confidence** — evidence levels in the facility's own records: direct mention 0.80–0.90, equipment only 0.50, department only 0.25, negated
