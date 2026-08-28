from pathlib import Path

import pandas as pd
import pytest

from scripts.verify_phase15_outputs import verify


def test_phase15_benchmark_exposes_machine_enforced_promotion_gate():
    source = (Path(__file__).parents[1] / "benchmarks" / "benchmark_phase15_candidate.py").read_text()
    assert "--require-promotion" in source
    assert "all_candidate_model_times_below_all_baselines" in source
    assert "all_candidate_destination_times_below_all_baselines" in source


def _outputs(root: Path, *, choice="WALK", logsum="1.000000") -> Path:
    root.mkdir()
    pd.DataFrame({"household_id": [1]}).to_csv(
        root / "final_households.csv", index=False
    )
    pd.DataFrame({
        "trip_id": [10],
        "destination": [4],
        "trip_mode": [choice],
        "destination_logsum": [logsum],
    }).to_csv(root / "final_trips.csv", index=False)
    return root


def _outputs_with_tours(
    root: Path, *, tour_mode="WALK", tour_logsum="2.000000"
) -> Path:
    _outputs(root)
    pd.DataFrame({
        "tour_id": [20],
        "tour_mode": [tour_mode],
        "mode_choice_logsum": [tour_logsum],
    }).to_csv(root / "final_tours.csv", index=False)
    return root


def test_phase15_accepts_exact_decisions_and_bounded_diagnostic_drift(tmp_path):
    reference = _outputs(tmp_path / "reference")
    candidate = _outputs(tmp_path / "candidate", logsum="1.000008")
    result = verify(reference, candidate)
    assert result["success"]
    assert result["decision_cells_different"] == 0
    assert result["diagnostic_max_abs"] == pytest.approx(8e-6)


def test_phase15_rejects_a_changed_modeled_decision(tmp_path):
    reference = _outputs(tmp_path / "reference")
    candidate = _outputs(tmp_path / "candidate", choice="DRIVEALONEFREE")
    with pytest.raises(RuntimeError, match="modeled cells"):
        verify(reference, candidate)


def test_phase15_rejects_unbounded_diagnostic_drift(tmp_path):
    reference = _outputs(tmp_path / "reference")
    candidate = _outputs(tmp_path / "candidate", logsum="1.001")
    with pytest.raises(RuntimeError, match="max abs"):
        verify(reference, candidate)


def test_verifier_accepts_bounded_tour_diagnostics(tmp_path):
    reference = _outputs_with_tours(tmp_path / "reference")
    candidate = _outputs_with_tours(
        tmp_path / "candidate", tour_logsum="2.000008"
    )
    result = verify(reference, candidate)
    assert result["success"]
    assert result["diagnostic_outputs"]["final_tours.csv"][
        "mode_choice_logsum"
    ]["max_abs"] == pytest.approx(8e-6)


def test_verifier_rejects_changed_tour_decision(tmp_path):
    reference = _outputs_with_tours(tmp_path / "reference")
    candidate = _outputs_with_tours(
        tmp_path / "candidate", tour_mode="DRIVEALONEFREE"
    )
    with pytest.raises(RuntimeError, match="final_tours.csv changed"):
        verify(reference, candidate)
