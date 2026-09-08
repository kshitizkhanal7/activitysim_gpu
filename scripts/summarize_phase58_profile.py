"""Summarize diagnostic profiles; cumulative entries are nested, not additive."""
import argparse
import json
from pathlib import Path
import pstats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = {}
    for path in sorted(args.directory.glob("*.pstats")):
        stats = pstats.Stats(str(path))
        rows = [{"file": key[0], "line": key[1], "function": key[2],
                 "primitive_calls": value[0], "calls": value[1],
                 "self_seconds": value[2], "cumulative_seconds": value[3]}
                for key, value in stats.stats.items()]
        output[path.stem] = {"total_profiled_seconds": stats.total_tt,
            "by_cumulative": sorted(rows, key=lambda r:r["cumulative_seconds"], reverse=True)[:50],
            "by_self": sorted(rows, key=lambda r:r["self_seconds"], reverse=True)[:50]}
    (args.output or args.directory / "summary.json").write_text(json.dumps(output, indent=2)+"\n")
    for model, result in output.items():
        print(model, result["total_profiled_seconds"])
        for row in result["by_cumulative"][:22]:
            print(f"  {row['cumulative_seconds']:.3f} cumulative / {row['self_seconds']:.3f} self: {Path(row['file']).name}:{row['line']} {row['function']}")


if __name__ == "__main__":
    main()
