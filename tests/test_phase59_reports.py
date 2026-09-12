import pytest
from scripts.verify_phase59_reports import verify


@pytest.mark.parametrize("corruption", [None, "cell", "missing", "extra"])
def test_summary_report_inventory_and_bytes(tmp_path, corruption):
    for side in ("cpu", "gpu"):
        directory = tmp_path/side/"summarize"
        directory.mkdir(parents=True)
        (directory/"trips.csv").write_text("trips\n123\n")
    target = tmp_path/"gpu"/"summarize"
    if corruption == "cell": (target/"trips.csv").write_text("trips\n124\n")
    if corruption == "missing": (target/"trips.csv").unlink()
    if corruption == "extra": (target/"extra.csv").write_text("unexpected\n")
    if corruption:
        with pytest.raises(ValueError): verify(tmp_path/"cpu", tmp_path/"gpu")
    else:
        assert verify(tmp_path/"cpu", tmp_path/"gpu")["report_count"] == 1


@pytest.mark.parametrize("candidate,accepted", [
    ("depart,tours\n5,111\n", True),
    ("depart,tours\n5.0000000000000001,111\n", False),
    ("depart,tours\n5,111.0\n", False),
    ("depart,tours\n5,112\n", False),
    ("depart,tours\nNaN,111\n", False),
    ("depart,tours\nInfinity,111\n", False),
    ("time,tours\n5,111\n", False),
    ("depart,tours\n5,111,extra\n", False),
    ("depart,tours\n5,111\n6,112\n", False),
])
def test_only_declared_departure_spelling_is_normalized(tmp_path, candidate, accepted):
    for side, content in (("cpu", "depart,tours\n5.0,111\n"), ("gpu", candidate)):
        directory = tmp_path/side/"summarize"
        directory.mkdir(parents=True)
        (directory/"school_tours_tod_count.csv").write_text(content)
    if not accepted:
        with pytest.raises(ValueError): verify(tmp_path/"cpu", tmp_path/"gpu")
    else:
        result = verify(tmp_path/"cpu", tmp_path/"gpu")
        assert result["exact"] and not result["byte_identical"]
        assert result["representation_only_differences"]["school_tours_tod_count.csv"] == [
            {"row":2, "column":"depart", "reference":"5.0", "candidate":"5"}]


def test_departure_spelling_is_not_normalized_in_unknown_report(tmp_path):
    for side, value in (("cpu", "5.0"), ("gpu", "5")):
        directory = tmp_path/side/"summarize"
        directory.mkdir(parents=True)
        (directory/"unexpected.csv").write_text(f"depart\n{value}\n")
    with pytest.raises(ValueError): verify(tmp_path/"cpu", tmp_path/"gpu")
