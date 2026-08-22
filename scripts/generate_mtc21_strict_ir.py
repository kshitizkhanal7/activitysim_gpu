"""Generate and validate the canonical strict IR for public Prototype MTC."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from choiceforge.sharrow_ir import specification_ir, write_ir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = specification_ir(pd.read_csv(args.spec, comment="#"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_ir(document, args.output)
    print(f"terms={len(document['terms'])} alternatives={len(document['alternatives'])} sha256={document['sha256']}")


if __name__ == "__main__":
    main()
