# Business Entity Resolution

A reproducible, fair-play business entity resolution system for the Amazon ML Challenge. It uses only supplied TSV files, classical scikit-learn models, and no business lookup, geocoding, or external enrichment.

## Quick start

```bash
python -m pip install -r requirements.txt
python run_pipeline.py --mode synthetic
pytest -q
uvicorn src.api.main:app --reload
```

`--mode all --data-dir dataset` detects `dataset/train/train_source1.tsv`. If it is absent, it generates deterministic **synthetic development data** and labels all reports as `data_type: synthetic`; otherwise it uses only challenge data and labels reports `challenge`. To run official files: place all required TSVs in `dataset/train` and `dataset/test`, then run `python run_pipeline.py --mode all --data-dir dataset`.

## Architecture and method

* `src/data`: explicit `pd.read_csv(..., sep="\t")`, schema/prefix/reference validation.
* `src/preprocessing`: Unicode-aware name/address normalization, legal suffix and abbreviation handling, postal and house-number extraction.
* `src/blocking`: exact name/address indexes, name/address inverted token indexes, sparse character n-gram and TF-IDF retrieval, plus a country fallback. Candidate sets are unioned, deduplicated, and limited; there is no full source Cartesian comparison.
* `src/features`: string, token, n-gram, component, postal, house-number, country/missingness, and combined features.
* `src/models`: a lightweight HistGradientBoosting pair classifier (well below the challenge parameter limit).
* `src/evaluation`: held-out Source 1 validation, macro singleton-aware F0.5, threshold search, and output invariant validation.

## Outputs and reports

The inference model scores exactly the candidate set written to `output/candidate_pairs.tsv`. `output/matching_results.tsv` contains predictions only from that candidate set. The pipeline validates one row per test Source 1 ID, legal candidate IDs, and the prediction subset constraint before it reports `submission_ready`.

`reports/` contains actual run-derived blocking metrics, validation metrics, threshold analysis, feature list, and summary. If supplied, run the organizer validator too:

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## API and UI

Start FastAPI with `uvicorn src.api.main:app --reload`. It exposes `/health`, dataset/blocking/evaluation/results endpoints, entity lookup, and pipeline endpoints. `frontend/Dashboard.tsx` is a Tailwind-ready React dashboard component that obtains real API values and displays a prominent demo banner for synthetic runs; its routes map to dashboard, dataset, blocking, matching, evaluation, and results views.

## Limitations

The supplied baseline uses deterministic rules and a classical model; tune blocking limits and model choices only through training validation when official data arrives. Country is treated as arbitrary normalized text, not a closed category. No leaderboard performance is claimed from synthetic results.
