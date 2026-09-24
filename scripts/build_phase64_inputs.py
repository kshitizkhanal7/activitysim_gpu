"""Prepare verified raw numeric inputs, never model answers; report setup cost."""
import argparse
import json
from pathlib import Path
import time

from phase63_commands import ROOT, PROJECT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--tag",required=True)
    args=parser.parse_args()
    if not args.tag.isalnum():
        raise ValueError("Alphanumeric tag required")
    target=ROOT/f"benchmark-results/phase64-{args.tag}-input-preparation.json"
    if target.exists():
        raise FileExistsError(target)
    from activitysim.core.input import _read_csv_with_fallback_encoding
    from choiceforge.phase64_inputs import NumericInputs
    cache=NumericInputs(ROOT/"benchmark-data/phase64-numeric-inputs")
    started=time.perf_counter()
    records=[]
    for name in ("households.csv","persons.csv","land_use.csv"):
        records.append(cache.prepare(_read_csv_with_fallback_encoding,PROJECT/"data_full"/name,{}))
    result=dict(complete=True,preparation_seconds=time.perf_counter()-started,records=records,
                raw_inputs_only=True,modeled_answers=False,summary=cache.summary())
    target.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
