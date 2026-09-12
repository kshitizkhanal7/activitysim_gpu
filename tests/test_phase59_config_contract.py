import hashlib
from choiceforge.phase59_config_contract import compatible_config


def test_only_extra_non_arithmetic_settings_can_change(tmp_path):
    expected = {"1:settings.yaml":"original-settings", "2:tour_mode_choice.csv":"spec"}
    current = {"2:settings.yaml":"original-settings", "3:tour_mode_choice.csv":"spec"}
    assert compatible_config(expected, current, [])
    path = tmp_path / "settings.yaml"
    path.write_text("inherit_settings: true\nrng_base_seed: 991\n")
    current["0:settings.yaml"] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert compatible_config(expected, current, [tmp_path])
    current["3:tour_mode_choice.csv"] = "changed-spec"
    assert not compatible_config(expected, current, [tmp_path])
    current["3:tour_mode_choice.csv"] = "spec"
    for text in ("inherit_settings: true\nCONSTANTS: {walkSpeed: 99}\n",
                 "inherit_settings: false\nrng_base_seed: 991\n"):
        path.write_text(text)
        current["0:settings.yaml"] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert not compatible_config(expected, current, [tmp_path])
