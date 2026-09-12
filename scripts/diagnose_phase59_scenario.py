"""Read-only first-difference inventory of saved scenario checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    for step in ("mandatory_tour_frequency", "mandatory_tour_scheduling", "tour_mode_choice_simulate"):
        suffix = Path("pipeline.parquetpipeline/tours") / (step+".parquet")
        left, right = pd.read_parquet(args.reference/suffix).sort_index(), pd.read_parquet(args.candidate/suffix).sort_index()
        if not left.index.equals(right.index):
            print(step, "different row identities")
            continue
        common = [c for c in left if c in right and not c.endswith("logsum")]
        different = left[common].ne(right[common]) & ~(left[common].isna() & right[common].isna())
        ids = different.any(axis=1)
        print(step, "rows", int(ids.sum()), "columns", different.sum()[different.sum() > 0].to_dict())
        if ids.any():
            cols = [c for c in ("person_id", "tour_type", "tour_num", "tour_count", "start", "end", "tdd", "tour_mode") if c in left]
            print("REFERENCE", left.loc[ids, cols].to_string())
            print("CANDIDATE", right.loc[ids, cols].to_string())


if __name__ == "__main__": main()
