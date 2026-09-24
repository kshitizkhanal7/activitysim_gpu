"""Recheck retained tiny live captures; never a production answer lookup."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    base=ROOT/"benchmark-results"
    target=base/"phase64-location-boundary-diagnostic.json"
    if target.exists():
        raise FileExistsError(target)
    directories={key:base/f"phase64-{tag}-location" for key,tag in
                 (("cpu","loccpu1"),("before","locgpu2"),("after","locfix2"))}
    evidence=[]
    arrays={}
    for name,directory in directories.items():
        receipt=directory/"receipt.json"
        metadata=json.loads(receipt.read_text())
        assert metadata["complete"] and metadata["diagnostic_only"]
        evidence.append(receipt)
        for record in metadata["records"]:
            path=Path(record["path"])
            evidence.append(path)
            with np.load(path,allow_pickle=False) as data:
                arrays[name,record["kind"],record["trace_label"]]={k:data[k].copy() for k in data.files}
    def get(name,kind,part):
        return next(v for (n,k,label),v in arrays.items() if n==name and k==kind and part in label)
    cpu=get("cpu","cpu-probabilities","simulate.work_veryhigh")
    cpu_choice=get("cpu","cpu-choices","simulate.work_veryhigh")
    before=get("before","gpu-final","simulate.work_veryhigh")
    after=get("after","cpu-probabilities","phase48_exact_guard")
    final=get("after","gpu-final","simulate.work_veryhigh")
    np.testing.assert_array_equal(cpu["utilities"].view(np.uint32),after["utilities"].view(np.uint32))
    np.testing.assert_array_equal(cpu["probabilities"].view(np.uint32),after["probabilities"].view(np.uint32))
    np.testing.assert_array_equal(cpu_choice["draws"].reshape(-1),before["draws"].reshape(-1))
    np.testing.assert_array_equal(cpu_choice["draws"].reshape(-1),final["draws"].reshape(-1))
    expected=int(before["alternative_ids"][int(cpu_choice["choices"][0])])
    assert int(final["chosen"][0,0])==expected!=int(before["chosen"][0,0])
    result=dict(complete=True,diagnostic_not_full_model_proof=True,
        original_utility_bit_mismatches=int(np.count_nonzero(cpu["utilities"].view(np.uint32)!=before["utilities"].view(np.uint32))),
        old_guard_already_flagged=bool(before["guard"][0]),
        corrected_utilities_bit_exact=True,corrected_probabilities_bit_exact=True,
        draw=float(cpu_choice["draws"][0]),draw_unchanged=True,
        cpu_internal_zone=expected,original_gpu_internal_zone=int(before["chosen"][0,0]),
        corrected_internal_zone=int(final["chosen"][0,0]),
        scope="One live captured public-population chooser, not a universal rounding theorem. Internal zero-based zone codes are one less than published zone IDs. Full-model regression required. These captures are diagnostic data, never loaded by production.",
        evidence_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence+[Path(__file__)]})
    target.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="evidence_sha256"}))


if __name__=="__main__":
    main()
