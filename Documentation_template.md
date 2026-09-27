# Methodology — Business Entity Resolution

## Methodology
The system normalizes supplied business names, addresses, and arbitrary country labels, generates a bounded multi-strategy candidate set, scores candidate pairs with a classical HistGradientBoosting classifier, and emits a conservative thresholded set of matches. It uses no external identity or geocoding source.

## Candidate generation / blocking strategy
The final model input is the union of exact normalized name blocks, exact address blocks, name-token and address-token inverted-index blocks, character n-gram sparse retrieval, word TF-IDF sparse retrieval, and country-aware fallback blocks. Limits are configurable. `candidate_pairs.tsv` is written from this exact final union, after deduplication, and all predictions are asserted to be a subset.

## Model architecture
HistGradientBoosting is a lightweight scikit-learn binary pair classifier. Positives are ground-truth candidate pairs; negatives are the realistic non-matching pairs created by the same candidate generator. Validation holds out Source 1 records and optimizes the operating threshold without test labels.

## Feature engineering
Features include normalized-name exact/edit/string/token/n-gram similarity; address exact/token/character/component similarity; postal and house number agreement; country equality and field missingness; a combined score; and source indicator.

## Evaluation
Metrics include macro, singleton-aware F0.5 (beta 0.5), aggregate precision/recall, false positives/negatives, singleton accuracy, candidate recall, candidate reduction, and candidate counts. Values in generated reports are actual run values. Synthetic runs are explicitly marked and are not challenge results.

## Other relevant information
The dataset switch is automatic based on `dataset/train/train_source1.tsv`. Outputs are TSV and are internally checked before `submission_ready` is set. Use the organizer validator, when included, as a final format check.
