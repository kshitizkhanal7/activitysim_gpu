import json
import os
import numpy as np
import pandas as pd
import pytest
from choiceforge.phase64_inputs import NumericInputs, exact_numeric


def reader(path,dtypes=None):
    return pd.read_csv(path,dtype=dtypes)


def test_private_bit_exact_round_trip(tmp_path):
    source=tmp_path/"data.csv"
    source.write_text("a,b\n1,-0.0\n2,3.5\n")
    cache=NumericInputs(tmp_path/"artifacts")
    record=cache.prepare(reader,source)
    assert record["bit_exact_round_trip"]
    actual=cache.read(reader,source)
    exact_numeric(reader(source),actual)
    actual.iloc[0,0]=999
    assert cache.read(reader,source).iloc[0,0]==1
    assert cache.summary()["hits"]==2


def test_same_size_preserved_timestamp_source_mutation_falls_back(tmp_path):
    source=tmp_path/"data.csv"
    source.write_text("a\n1\n")
    cache=NumericInputs(tmp_path/"artifacts")
    cache.prepare(reader,source)
    stamp=source.stat()
    source.write_text("a\n9\n")
    os.utime(source,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
    assert cache.read(reader,source).a.iloc[0]==9
    assert cache.summary()["misses"]==1


def test_artifact_corruption_rejected(tmp_path):
    source=tmp_path/"data.csv"
    source.write_text("a\n1\n")
    cache=NumericInputs(tmp_path/"artifacts")
    cache.prepare(reader,source)
    _,artifact=cache.paths(cache.identity(source,None))
    with artifact.open("r+b") as stream:
        stream.write(b"BAD!")
    with pytest.raises(ValueError,match="identity/content"):
        cache.read(reader,source)


def test_dtype_identity_and_schema_corruption(tmp_path):
    source=tmp_path/"data.csv"
    source.write_text("a\n1\n")
    cache=NumericInputs(tmp_path/"artifacts")
    cache.prepare(reader,source)
    assert cache.read(reader,source,{"a":"float32"}).a.dtype==np.dtype("float32")
    metadata,_=cache.paths(cache.identity(source,None))
    record=json.loads(metadata.read_text())
    record["schema"]["rows"]=5
    metadata.write_text(json.dumps(record))
    with pytest.raises(ValueError,match="schema"):
        cache.read(reader,source)


def test_unsupported_objects_not_prepared(tmp_path):
    source=tmp_path/"data.csv"
    source.write_text("a\nhello\n")
    cache=NumericInputs(tmp_path/"artifacts")
    with pytest.raises(ValueError,match="numeric"):
        cache.prepare(reader,source)
    assert cache.read(reader,source,{"a":str}).a.iloc[0]=="hello"


def test_existing_artifacts_never_overwritten(tmp_path):
    source=tmp_path/"data.csv"
    source.write_text("a\n1\n")
    cache=NumericInputs(tmp_path/"artifacts")
    cache.prepare(reader,source)
    with pytest.raises(FileExistsError):
        cache.prepare(reader,source)
