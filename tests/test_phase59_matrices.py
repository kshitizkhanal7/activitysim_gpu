import h5py
import numpy as np
import pytest
from scripts.verify_phase59_matrices import verify


def write(path, *, compression=None):
    with h5py.File(path, "w") as file:
        file.create_dataset("data/drive", data=np.eye(3), compression=compression)
        file.create_dataset("lookup/zone", data=np.array([7, 19, 31]))


def test_identical_content_ignores_container_compression(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    write(left / "trips.omx")
    write(right / "trips.omx", compression="gzip")
    assert verify(left, right)["exact"]


@pytest.mark.parametrize("corruption", ["cell", "mapping", "missing", "extra"])
def test_matrix_verifier_rejects_corruption(tmp_path, corruption):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    write(left / "trips.omx")
    write(right / "trips.omx")
    with h5py.File(right / "trips.omx", "r+") as file:
        if corruption == "cell":
            file["data/drive"][0, 0] = np.nextafter(1., 2.)
        elif corruption == "mapping":
            file["lookup/zone"][0] = 8
        elif corruption == "missing":
            del file["data/drive"]
        else:
            file.create_dataset("data/extra", data=np.eye(3))
    with pytest.raises(ValueError):
        verify(left, right)
