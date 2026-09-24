"""Summarize measured cProfile costs; explicitly not speed evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import pstats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = sorted(args.directory.glob("*.pstats"))
    if not paths:
        raise ValueError("No completed component profiles")
    result = {"diagnostic_not_performance_evidence":True,"components":{},"evidence_sha256":{}}
    for path in paths:
        stats = pstats.Stats(str(path))
        records = [dict(file=f,line=line,function=func,primitive_calls=value[0],calls=value[1],
                        self_seconds=value[2],cumulative_seconds=value[3])
                   for (f,line,func),value in stats.stats.items()]
        result["components"][path.stem] = dict(total_seconds=stats.total_tt,
            top_self=sorted(records,key=lambda r:r["self_seconds"],reverse=True)[:35],
            top_cumulative=sorted(records,key=lambda r:r["cumulative_seconds"],reverse=True)[:35])
        result["evidence_sha256"][str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    result["evidence_sha256"][str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({name:r["total_seconds"] for name,r in result["components"].items()}))


if __name__=="__main__":
    main()
