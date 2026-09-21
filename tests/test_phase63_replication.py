import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest
import zstandard

sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
import prepare_phase63_replication as prep


def sha(content):
    return hashlib.sha256(content).hexdigest()


def test_public_archive_materialization_and_no_overwrite(tmp_path,monkeypatch):
    content = b"x\n1\n"
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes,mode="w") as tar:
        member = tarfile.TarInfo("data_full/households.csv")
        member.size = len(content)
        tar.addfile(member,io.BytesIO(content))
    archive = tmp_path/"data.tar.zst"
    archive.write_bytes(zstandard.ZstdCompressor().compress(tar_bytes.getvalue()))
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps(dict(configuration={"configs/settings.yaml":{"text":"test: true\n","sha256":sha(b"test: true\n")}},
        data={"households.csv":{"sha256":sha(content),"bytes":len(content)}},
        public_archive={"sha256":prep.digest(archive)})))
    monkeypatch.setattr(prep,"MANIFEST",manifest)
    project = tmp_path/"project"
    assert prep.materialize(project,archive)["all_public_inputs_exact"]
    assert prep.materialize(project,archive)["all_public_inputs_exact"]
    target = project/"data_full/households.csv"
    target.write_bytes(b"changed")
    with pytest.raises(ValueError,match="Existing input differs"):
        prep.materialize(project,archive)
    assert target.read_bytes()==b"changed"


def test_snapshot_rejects_path_escape_and_corrupt_contents(tmp_path,monkeypatch):
    manifest = tmp_path/"manifest.json"
    monkeypatch.setattr(prep,"MANIFEST",manifest)
    for name,expected in (("../outside","Unsafe"),("..\\outside","Unsafe"),("C:outside","Unsafe"),("configs/test","Corrupt")):
        manifest.write_text(json.dumps({"configuration":{name:{"text":"abc","sha256":"wrong"}}}))
        with pytest.raises(ValueError,match=expected):
            prep.materialize(tmp_path/"project")
    assert not (tmp_path/"outside").exists()
