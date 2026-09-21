"""Build and verify the raw public skim image without distributing model answers."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import re

from phase63_commands import ROOT, PROJECT, PYTHON, candidate_command, environment, output_directory
from phase62_batch_worker import file_digest, verify_files
from run_phase58_comparison import source_fingerprint, config_fingerprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--cache",type=Path,default=ROOT/"cache_sharrow/phase56_taz.mmap")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9-]+",args.tag):
        raise ValueError("Invalid tag")
    cache = args.cache.resolve()
    prefix = f"phase63-{args.tag}-image"
    result_path = ROOT/"benchmark-results"/(prefix+".json")
    if result_path.exists():
        raise FileExistsError(result_path)
    source,configs = source_fingerprint(),config_fingerprint()
    source.update({str(p.relative_to(ROOT)):file_digest(p) for p in (Path(__file__),ROOT/"scripts/phase63_commands.py")})
    result = dict(complete=False,built=False)
    if not cache.exists():
        output = output_directory("o-"+prefix)
        output.parent.mkdir(parents=True,exist_ok=True)
        report = ROOT/"benchmark-results"/(prefix+"-report.json")
        kernels = ROOT/"benchmark-results"/(prefix+"-kernels")
        for path in (output,report,kernels,Path(str(cache)+".meta.pkl")):
            if path.exists():
                raise FileExistsError(path)
        (kernels/"mode").mkdir(parents=True)
        command = candidate_command(output,args.reference.resolve(),report,kernels,
                                    ROOT/"benchmark-results"/(prefix+"-checkpoint.json"))
        command += ["--phase56-skim-cache",str(cache),"--phase56-build-skim-cache"]
        env = environment(True)
        env.update(CHOICEFORGE_PHASE17_REPORT_DIR=str(kernels),CHOICEFORGE_PHASE17_MODE_REPORT_DIR=str(kernels/"mode"),
                   CHOICEFORGE_PHASE17_RUN_ID=prefix)
        started = time.perf_counter()
        with (ROOT/"benchmark-results"/(prefix+".stdout.log")).open("w") as out,(ROOT/"benchmark-results"/(prefix+".stderr.log")).open("w") as err:
            subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=err,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        result.update(built=True,build_model_process_seconds=time.perf_counter()-started,command=command)
        proof = json.loads(report.read_text())
        if not all(proof["proof_gates"].values()):
            raise ValueError("Image build model failed live gates")
        from verify_phase15_outputs import verify as decisions
        from verify_phase59_matrices import verify as matrices
        from verify_phase59_reports import verify as reports
        result.update(exact=decisions(args.reference,output),matrices=matrices(args.reference,output),
                      summary_reports=reports(args.reference,output))
    from choiceforge.phase56_skim_cache import validate_phase56_cache
    result["image"] = validate_phase56_cache(cache,PROJECT,PROJECT/"data_full",verify_cache_content=True)
    verify_files({str(ROOT/p):v for p,v in source.items()})
    verify_files(configs)
    result.update(complete=True,source_sha256=source,configuration_sha256=configs,
                  scope="Raw immutable network image only. Build work is preparation, excluded from warm fresh-model timing.")
    result_path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"complete":True,"built":result["built"],"full_image_digest_verified":True}))


if __name__=="__main__":
    main()
