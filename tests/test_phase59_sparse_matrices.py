from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import openmatrix
from activitysim.abm.models.trip_matrices import write_matrices as upstream_writer
from scripts.verify_phase59_matrices import verify


@pytest.mark.parametrize("chunk", [4, 16, 32, 64, 128, 256])
def test_sparse_omx_is_logically_identical_and_readable(tmp_path, chunk):
    from choiceforge.phase59_sparse_matrices import write_matrices
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    zones = pd.Index(np.arange(259)*17+2, name="zone_id")
    orig, dest = np.array([0, 1, 8, 258]), np.array([258, 1, 4, 0])
    counts = pd.DataFrame({"drive":[2., 1., -0., 9.], "zero":np.zeros(4),
                           "sample_rate":[.1, .2, .3, .4]})
    # Repeated fields preserve upstream's successive in-place scaling behavior.
    tables = [SimpleNamespace(name="drive", data_field="drive"),
              SimpleNamespace(name="drive_again", data_field="drive"),
              SimpleNamespace(name="zero", data_field="zero")]
    settings = SimpleNamespace(MATRICES=[SimpleNamespace(file_name="trips.omx", tables=tables)],
                               HH_EXPANSION_WEIGHT_COL="sample_rate")
    upstream_writer(SimpleNamespace(get_output_file_path=lambda name:left/name), counts.copy(),
                    zones, orig, dest, settings)
    event = write_matrices(SimpleNamespace(get_output_file_path=lambda name:right/name), counts.copy(),
                          zones, orig, dest, settings, chunk_size=chunk, workers=2)
    assert verify(left, right)["exact"]
    with openmatrix.open_file(right / "trips.omx", "r") as file:
        assert file.shape() == (259,259)
        assert set(file.list_matrices()) == {"drive", "drive_again", "zero"}
        assert file.mapping("zone_id") == dict(zip(zones, range(len(zones))))
        assert file["drive"][0,258] == 20.
    assert event["omx_files"] == 1
    assert event["matrices"] == 3
    assert event["compression_workers"] == 2
