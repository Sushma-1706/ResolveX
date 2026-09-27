from __future__ import annotations
from pathlib import Path
import pandas as pd
from .validator import validate_source, validate_ground_truth

def read_tsv(path: Path) -> pd.DataFrame:
    try: return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    except Exception as exc: raise ValueError(f"Malformed TSV {path}: {exc}") from exc

def load_split(directory: str | Path, split: str, ground_truth: bool = False) -> dict:
    d = Path(directory)
    records = {f"source{i}": read_tsv(d / f"{split}_source{i}.tsv") for i in (1, 2, 3)}
    for i in (1,2,3): validate_source(records[f"source{i}"], f"S{i}-", f"{split}_source{i}")
    if ground_truth:
        records["ground_truth"] = read_tsv(d / f"{split}_ground_truth.tsv")
        validate_ground_truth(records["ground_truth"], records["source1"], pd.concat([records["source2"], records["source3"]]))
    return records
