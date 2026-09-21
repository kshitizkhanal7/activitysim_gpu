"""Seal or materialize the exact public benchmark, never modeled answers.

The release archive is authenticated by its published SHA256 and every selected
member by our benchmark digest. Configuration snapshots replace moving branches.
Existing mismatched files are rejected rather than overwritten.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT/"benchmark-data/phase9-mtc-full/prototype_mtc_extended"
MANIFEST = ROOT/"reproducibility/phase63-inputs.json"
SOURCE_SNAPSHOT = ROOT/"reproducibility/phase63-upstream-source.json"
UPSTREAM = "16ab11180a26912987eb902daf945e268f3efc11"
ARCHIVE = {"url":"https://github.com/ActivitySim/activitysim-prototype-mtc/releases/download/v1.3.4/data_full.tar.zst",
           "sha256":"b402506a61055e2d38621416dd9a5c7e3cf7517c0a9ae5869f6d760c03284ef3"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def ensure_file(path,data):
    """Idempotent exact writes; never replace an existing different file."""
    path = Path(path)
    if path.exists():
        if path.read_bytes()!=data:
            raise ValueError(f"Existing file differs, refusing overwrite: {path}")
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def seal():
    files = {}
    for directory in ("configs","configs_sh"):
        for path in sorted((PROJECT/directory).rglob("*")):
            if path.is_file() and path.suffix.lower() in {".yaml",".yml",".csv"}:
                files[path.relative_to(PROJECT).as_posix()] = {"text":path.read_bytes().decode("utf8"),"sha256":digest(path)}
    for name in ("LICENSE",):
        path = PROJECT/name
        files[name] = {"text":path.read_bytes().decode("utf8"),"sha256":digest(path)}
    data = {p.name:{"sha256":digest(p),"bytes":p.stat().st_size}
            for p in sorted((PROJECT/"data_full").iterdir()) if p.is_file()}
    packages = sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
                       if d.metadata['Name'].lower() not in {"activitysim","choiceforge"}})
    payload = dict(format=1,upstream_commit=UPSTREAM,activitysim_distribution_version=importlib.metadata.version("activitysim"),
                   python=platform.python_version(),platform=platform.platform(),
                   public_archive=ARCHIVE,data=data,configuration=files,
                   integration_patch="integration/activitysim-phase63.patch",
                   integration_patch_sha256=digest(ROOT/"integration/activitysim-phase63.patch"),
                   limits="Prepared Windows x64 workstation; NVIDIA sm86 path. Not independent replication.")
    ensure_file(MANIFEST,(json.dumps(payload,indent=2)+"\n").encode())
    ensure_file(ROOT/"requirements-phase63-lock.txt",("\n".join(packages)+"\n").encode())
    seal_code()
    return {"configurations":len(files),"data_files":len(data),"packages":len(packages)}


def seal_code():
    checkout = ROOT/"tmp/activitysim-phase8-source"
    names = ("activitysim/abm/models/trip_destination.py",
             "activitysim/abm/models/util/vectorize_tour_scheduling.py","activitysim/_generated_version.py")
    records = {name:{"text":(checkout/name).read_bytes().decode("utf8"),"sha256":digest(checkout/name)} for name in names}
    ensure_file(SOURCE_SNAPSHOT,(json.dumps(records,indent=2)+"\n").encode())
    return {"sealed_upstream_sources":len(records)}


def normalize_checkout():
    """Reproduce reviewed newline bytes and generated pinned version metadata.

    Never replace an unexpected edited model. The two model files must already
    match the integration patch semantically, allowing CRLF/LF differences only.
    """
    checkout = ROOT/"tmp/activitysim-phase8-source"
    manifest = json.loads(MANIFEST.read_text())
    records = json.loads(SOURCE_SNAPSHOT.read_text())
    for name,record in records.items():
        path = checkout/name
        wanted = record["text"].encode("utf8")
        if hashlib.sha256(wanted).hexdigest()!=record["sha256"]:
            raise ValueError("Corrupt upstream source snapshot")
        actual = path.read_bytes()
        if name.endswith("/_generated_version.py"):
            if manifest["activitysim_distribution_version"] not in actual.decode("utf8"):
                raise ValueError("Build did not use the pinned version override")
        elif actual.replace(b"\r\n",b"\n")!=wanted.replace(b"\r\n",b"\n"):
            raise ValueError(f"Unexpected patched code; refusing overwrite: {name}")
        if actual!=wanted:
            path.write_bytes(wanted)
    return {"exact_patched_and_generated_sources":True}


def materialize(project,archive=None):
    manifest = json.loads(MANIFEST.read_text())
    for name,record in manifest["configuration"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError("Unsafe configuration path")
        content = record["text"].encode("utf8")
        if hashlib.sha256(content).hexdigest()!=record["sha256"]:
            raise ValueError("Corrupt configuration snapshot")
        ensure_file(project/name,content)
    missing = []
    for name,record in manifest["data"].items():
        if PurePosixPath(name).name!=name or "\\" in name or ":" in name:
            raise ValueError("Unsafe data name")
        target = project/"data_full"/name
        if target.exists():
            if digest(target)!=record["sha256"]:
                raise ValueError(f"Existing input differs: {target}")
        else:
            missing.append(name)
    if missing:
        if archive is None:
            archive = project/"downloads/data_full.tar.zst"
        if not archive.exists():
            archive.parent.mkdir(parents=True,exist_ok=True)
            partial = archive.with_suffix(archive.suffix+".part")
            with urllib.request.urlopen(manifest["public_archive"]["url"]) as source,partial.open("xb") as out:
                shutil.copyfileobj(source,out)
            if digest(partial)!=manifest["public_archive"]["sha256"]:
                raise ValueError("Downloaded archive digest mismatch; partial file retained")
            partial.rename(archive)
        if digest(archive)!=manifest["public_archive"]["sha256"]:
            raise ValueError("Archive digest mismatch")
        import zstandard
        with archive.open("rb") as source,zstandard.ZstdDecompressor().stream_reader(source) as expanded:
            with tarfile.open(fileobj=expanded,mode="r|") as tar:
                for member in tar:
                    name = PurePosixPath(member.name).name
                    if name not in missing:
                        continue
                    if not member.isfile() or member.size!=manifest["data"][name]["bytes"]:
                        raise ValueError("Unexpected archive member")
                    target = project/"data_full"/name
                    target.parent.mkdir(parents=True,exist_ok=True)
                    partial = target.with_suffix(target.suffix+".part")
                    with tar.extractfile(member) as data,partial.open("xb") as out:
                        shutil.copyfileobj(data,out)
                    if digest(partial)!=manifest["data"][name]["sha256"]:
                        raise ValueError("Archive member digest mismatch")
                    partial.rename(target)
                    missing.remove(name)
        if missing:
            raise ValueError(f"Missing archive members: {missing}")
    return {"project":str(project),"all_public_inputs_exact":True}


def check():
    manifest = json.loads(MANIFEST.read_text())
    for name,record in manifest["configuration"].items():
        if digest(PROJECT/name)!=record["sha256"]:
            raise ValueError(f"Configuration differs: {name}")
    for name,record in manifest["data"].items():
        if digest(PROJECT/"data_full"/name)!=record["sha256"]:
            raise ValueError(f"Public data differs: {name}")
    if digest(ROOT/manifest["integration_patch"])!=manifest["integration_patch_sha256"]:
        raise ValueError("Integration patch differs")
    checkout = ROOT/"tmp/activitysim-phase8-source"
    commit = subprocess.check_output(["git","-C",str(checkout),"rev-parse","HEAD"],text=True).strip()
    if commit!=manifest["upstream_commit"]:
        raise ValueError("Upstream revision differs")
    if importlib.metadata.version("activitysim")!=manifest["activitysim_distribution_version"]:
        raise ValueError("ActivitySim distribution metadata differs")
    for name,record in json.loads(SOURCE_SNAPSHOT.read_text()).items():
        if digest(checkout/name)!=record["sha256"]:
            raise ValueError(f"Patched/generated source bytes differ: {name}")
    subprocess.run(["git","-C",str(checkout),"apply","--reverse","--check",
                    str(ROOT/manifest["integration_patch"])],check=True)
    wanted = (ROOT/"requirements-phase63-lock.txt").read_text().splitlines()
    for pin in wanted:
        name,version = pin.split("==")
        if importlib.metadata.version(name)!=version:
            raise ValueError(f"Environment differs: {pin}")
    if platform.python_version()!=manifest["python"]:
        raise ValueError("Python version differs")
    return {"prepared_workspace_verified":True,"packages":len(wanted),
            "second_machine_replication":False,"public_input_count":len(manifest["data"])}


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=("seal","seal-code","normalize-checkout","materialize","check"))
    parser.add_argument("--project",type=Path,default=PROJECT)
    parser.add_argument("--archive",type=Path)
    args = parser.parse_args()
    result = {"seal":seal,"seal-code":seal_code,"normalize-checkout":normalize_checkout,"check":check}.get(
        args.action,lambda:materialize(args.project,args.archive))()
    print(json.dumps(result))
