import pytest

from src.settings import ROOT_DIR, load_config, validate_config


def test_config_loads() -> None:
    config = load_config()

    assert config["project"]["task"] == "binary_audio_classification"
    assert config["audio"]["sample_rate"] == 16000
    assert ROOT_DIR.exists()


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="Missing configuration sections"):
        validate_config({"project": {}})


def test_label_mapping_is_fixed() -> None:
    config = load_config()
    config["data"]["labels"] = {"bonafide": 1, "spoof": 0}

    with pytest.raises(ValueError, match="Labels must map"):
        validate_config(config)
