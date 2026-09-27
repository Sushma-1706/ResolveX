from __future__ import annotations
import pandas as pd
from .schema import SOURCE_COLUMNS, GROUND_TRUTH_COLUMNS

class DatasetValidationError(ValueError): pass

def validate_source(df: pd.DataFrame, expected_prefix: str, label: str) -> None:
    missing = set(SOURCE_COLUMNS) - set(df.columns)
    if missing: raise DatasetValidationError(f"{label}: missing required columns {sorted(missing)}; verify TSV separator")
    if df.empty: raise DatasetValidationError(f"{label}: file is empty")
    if df.entity_id.isna().any() or (df.entity_id.astype(str).str.strip() == "").any(): raise DatasetValidationError(f"{label}: missing entity_id")
    if df.entity_id.duplicated().any(): raise DatasetValidationError(f"{label}: duplicate entity IDs: {df.loc[df.entity_id.duplicated(), 'entity_id'].head().tolist()}")
    bad = ~df.entity_id.astype(str).str.startswith(expected_prefix)
    if bad.any(): raise DatasetValidationError(f"{label}: invalid IDs for {expected_prefix}: {df.loc[bad, 'entity_id'].head().tolist()}")

def validate_ground_truth(gt: pd.DataFrame, s1: pd.DataFrame, candidates: pd.DataFrame) -> None:
    missing = set(GROUND_TRUTH_COLUMNS) - set(gt.columns)
    if missing: raise DatasetValidationError(f"ground truth: missing columns {sorted(missing)}")
    if gt.source1_entity_id.duplicated().any(): raise DatasetValidationError("ground truth: duplicate source1_entity_id")
    s1ids, candidate_ids = set(s1.entity_id), set(candidates.entity_id)
    unknown_s1 = set(gt.source1_entity_id) - s1ids
    if unknown_s1: raise DatasetValidationError(f"ground truth references unknown Source 1 IDs: {sorted(unknown_s1)[:5]}")
    for row in gt.fillna("").itertuples(index=False):
        ids = [x for x in str(row.matched_entity_ids).split(',') if x]
        if len(ids) != len(set(ids)): raise DatasetValidationError(f"ground truth duplicate match: {row.source1_entity_id}")
        unknown = set(ids) - candidate_ids
        if unknown: raise DatasetValidationError(f"ground truth unknown candidates for {row.source1_entity_id}: {sorted(unknown)}")
