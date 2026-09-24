"""Verified raw numeric input artifacts; no model outputs and no pickle.

The source CSV and stored Feather file are hashed on EVERY load. Keys include
parser source, numeric dtype requests and library versions. Models receive
private frames; unsupported schemas use the original reader. Preparation is
explicit and never silently performed inside a warm timing claim.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def supported(frame):
    return (isinstance(frame.index,pd.RangeIndex) and frame.index.start==0 and frame.index.step==1
            and frame.index.name is None and frame.columns.is_unique
            and all(type(c) is str for c in frame.columns)
            and all(isinstance(dtype,np.dtype) and dtype.kind in "biuf" for dtype in frame.dtypes))


def schema(frame):
    return dict(rows=len(frame),columns=list(frame.columns),dtypes=[d.str for d in frame.dtypes])


def exact_numeric(left,right):
    if schema(left)!=schema(right) or not supported(left) or not supported(right):
        raise ValueError("Numeric input schema changed")
    for name in left:
        a=np.ascontiguousarray(left[name].to_numpy())
        b=np.ascontiguousarray(right[name].to_numpy())
        if a.tobytes()!=b.tobytes():
            raise ValueError(f"Numeric input bits changed in {name}")


class NumericInputs:
    def __init__(self,directory):
        self.directory=Path(directory)
        self.events=[]

    def identity(self,filepath,dtypes):
        from activitysim.core import input as upstream
        if dtypes is not None and (not isinstance(dtypes,dict) or any(type(k) is not str for k in dtypes)):
            return None
        try:
            types=sorted((k,np.dtype(v).str) for k,v in (dtypes or {}).items())
            if any(np.dtype(v).kind not in "biuf" for _,v in types):
                return None
        except (TypeError,ValueError):
            return None
        return dict(format=1,source_sha256=digest(filepath),dtypes=types,
            parser_sha256=digest(upstream.__file__),pandas=pd.__version__,numpy=np.__version__,
            pyarrow=pyarrow.__version__)

    def paths(self,identity):
        key=hashlib.sha256(encoded(identity)).hexdigest()
        return self.directory/(key+".json"),self.directory/(key+".feather")

    def read(self,original,filepath,dtypes=None):
        started=time.perf_counter()
        identity=self.identity(filepath,dtypes)
        if identity is None:
            return original(filepath,dtypes)
        metadata,artifact=self.paths(identity)
        if not metadata.exists():
            self.events.append(dict(path=str(filepath),hit=False,reason="not prepared"))
            return original(filepath,dtypes)
        record=json.loads(metadata.read_text())
        # JSON converts tuple dtype entries to lists; compare canonical bytes.
        if encoded(record["identity"])!=encoded(identity) or digest(artifact)!=record["artifact_sha256"]:
            raise ValueError("Numeric input artifact identity/content changed")
        verified=time.perf_counter()
        frame=pd.read_feather(artifact)
        if schema(frame)!=record["schema"] or not supported(frame):
            raise ValueError("Numeric input artifact schema changed")
        result=frame.copy(deep=True)
        result.index=frame.index.copy(deep=True)
        result.columns=frame.columns.copy(deep=True)
        self.events.append(dict(path=str(filepath),hit=True,source_sha256=identity["source_sha256"],
            artifact_sha256=record["artifact_sha256"],verification_seconds=verified-started,
            load_copy_seconds=time.perf_counter()-verified,rows=len(result),private_copy=True))
        return result

    def prepare(self,original,filepath,dtypes=None):
        identity=self.identity(filepath,dtypes)
        if identity is None:
            raise ValueError("Unsupported numeric dtype request")
        metadata,artifact=self.paths(identity)
        if metadata.exists() or artifact.exists():
            raise FileExistsError("Preserve existing artifacts; use a new preparation directory")
        frame=original(filepath,dtypes)
        if not supported(frame):
            raise ValueError("Only raw numeric tables with default RangeIndex are supported")
        self.directory.mkdir(parents=True,exist_ok=True)
        frame.to_feather(artifact,compression="uncompressed")
        # Independent round trip, including signed zero/NaN bit representation.
        exact_numeric(frame,pd.read_feather(artifact))
        if self.identity(filepath,dtypes)!=identity:
            raise ValueError("Source CSV changed while preparing numeric inputs")
        record=dict(identity=identity,source=str(Path(filepath).resolve()),schema=schema(frame),
                    artifact_sha256=digest(artifact),artifact_bytes=artifact.stat().st_size,
                    bit_exact_round_trip=True,modeled_answers=False)
        metadata.write_text(json.dumps(record,indent=2)+"\n")
        return record

    def summary(self):
        return dict(events=self.events,hits=sum(e["hit"] for e in self.events),
                    misses=sum(not e["hit"] for e in self.events),saved_answers_read=False,
                    scope="raw CSV values, full source/artifact SHA each read, private numeric frames")
