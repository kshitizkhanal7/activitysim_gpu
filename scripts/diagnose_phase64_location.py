"""Live, read-only arithmetic capture for a requested location chooser.

Stops after workplace location; partial outputs are explicitly not a full model
or performance measurement. No reference answers are injected into execution.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

import numpy as np
from phase63_commands import ROOT, PROJECT, candidate_command, environment, output_directory
from run_phase58_comparison import source_fingerprint


class DiagnosticComplete(BaseException):
    pass


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--tag",required=True)
    parser.add_argument("--mode",choices=("cpu","gpu"),required=True)
    parser.add_argument("--chooser",type=int,required=True)
    parser.add_argument("--households",type=int,default=250000)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--features",default="expressions,inputs")
    args=parser.parse_args()
    output=output_directory("o-phase64-"+args.tag)
    directory=ROOT/"benchmark-results"/f"phase64-{args.tag}-location"
    if output.exists() or directory.exists():
        raise FileExistsError("Preserve old diagnostic evidence")
    directory.mkdir()
    output.mkdir()
    os.environ.update(environment(args.mode=="gpu"))
    from activitysim.core import logit
    from activitysim.core.workflow.runner import Runner
    original_step=Runner.by_name
    original_probs=logit.utils_to_probs
    original_choices=logit.make_choices
    records=[]
    source=source_fingerprint()
    def capture(kind,label,**arrays):
        path=directory/f"{len(records):02d}-{kind}.npz"
        np.savez(path,**arrays)
        records.append(dict(kind=kind,trace_label=str(label),path=str(path)))
    def target(frame):
        return np.flatnonzero(np.asarray(frame.index)==args.chooser)
    def probabilities(state,utilities,*a,**k):
        result=original_probs(state,utilities,*a,**k)
        rows=target(utilities)
        if len(rows):
            probs=result[0] if isinstance(result,tuple) else result
            capture("cpu-probabilities",k.get("trace_label"),utilities=utilities.to_numpy()[rows],
                    probabilities=probs.to_numpy()[rows])
        return result
    def choices(state,probs,*a,**k):
        result=original_choices(state,probs,*a,**k)
        rows=target(probs)
        if len(rows):
            selected,draws=result
            capture("cpu-choices",k.get("trace_label"),probabilities=probs.to_numpy()[rows],
                    choices=np.asarray(selected)[rows],draws=np.asarray(draws)[rows])
        return result
    logit.utils_to_probs=probabilities
    logit.make_choices=choices
    if args.mode=="gpu":
        from choiceforge import modelwide_graph
        original_final=modelwide_graph.finish_resident_final_choice
        def final(**kwargs):
            rows=target(kwargs["choosers"])
            if not len(rows):
                return original_final(**kwargs)
            service=kwargs["service"]
            original_rng=service.random_for_df
            captured={}
            def rng(state,frame,*a,**k):
                result=original_rng(state,frame,*a,**k)
                indices=target(frame)
                if len(indices):
                    captured["draws"]=(service.cp.asnumpy(result[1][indices]) if result[0] is None
                                       else np.asarray(result[0])[indices].copy())
                return result
            service.random_for_df=rng
            try:
                result=original_final(**kwargs)
            finally:
                service.random_for_df=original_rng
            cp=service.cp
            row=int(rows[0])
            offsets=kwargs["offsets"]
            lo,hi=int(offsets[row]),int(offsets[row+1])
            alternatives=kwargs["alternatives"]
            capture("gpu-final",kwargs["trace_label"],
                utilities=cp.asnumpy(kwargs["padded"][rows]),
                weights=cp.asnumpy(kwargs["workspace"]["weights"][rows]),
                guard=cp.asnumpy(kwargs["workspace"]["guard"][rows]),
                alternative_ids=alternatives[kwargs["choice_column"]].to_numpy()[lo:hi],
                mode_logsums=cp.asnumpy(kwargs["resident_mode_logsum_device"][lo:hi]),
                chosen=np.asarray(result.loc[[args.chooser]]),**captured)
            alternatives.iloc[lo:hi].to_parquet(directory/"gpu-alternatives.parquet")
            return result
        modelwide_graph.finish_resident_final_choice=final
    def step(runner,name,*a,**k):
        result=original_step(runner,name,*a,**k)
        if str(name)=="workplace_location":
            raise DiagnosticComplete()
        return result
    Runner.by_name=step
    try:
        if args.mode=="gpu":
            command=candidate_command(output,args.reference,directory/"unused-report.json",directory/"kernels",
                directory/"unused-checkpoint.json",args.households,None,"plans,files,rss")
            command += ["--phase64-features",args.features]
            sys.argv=command[1:]
            runpy.run_path(command[1],run_name="__main__")
        else:
            os.chdir(PROJECT)
            sys.argv=["activitysim","run","-c","configs_sh","-c","configs","-d","data_full",
                      "-o",str(output),"--households_sample_size",str(args.households),"--fast"]
            runpy.run_module("activitysim",run_name="__main__")
    except DiagnosticComplete:
        if not records or source_fingerprint()!=source:
            raise ValueError("No capture or changed production source")
        (directory/"receipt.json").write_text(json.dumps(dict(complete=True,diagnostic_only=True,
            mode=args.mode,chooser=args.chooser,records=records,source_sha256=source),indent=2)+"\n")
        print(f"Location diagnostic complete: {directory}")


if __name__=="__main__":
    main()
