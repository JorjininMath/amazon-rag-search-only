# P0 Checklist (Baseline Only)

## Goal

Create a reproducible baseline for search-only retrieval without changing ranking logic.

## Files

- `data/eval/golden_queries.jsonl`
- `data/eval/baseline_results.csv`

## Run Loop

Option A (recommended): run automatic baseline script:

- `python run_p0_baseline.py`

Then fill two manual columns in `baseline_results.csv`:

- `meets_expectation` (`yes/no`)
- `notes` (one-line reason)

Option B (manual loop):

For each query in `golden_queries.jsonl`:

1. Run:
   - `python query_evidence.py --query "<QUERY>" --top-k 5 --index-dir data/index`
2. Count low-rating evidence:
   - `low_rating_count = number of results where rating <= 2`
3. Judge expectation:
   - `meets_expectation = yes/no` (manual)
4. Append one row to `baseline_results.csv`:
   - `id,query,top_k,low_rating_count,meets_expectation,notes`

## Definition of Done

- All 10 queries have result rows.
- Each row has a clear `notes` reason.
- You can summarize current weakness in one sentence.
