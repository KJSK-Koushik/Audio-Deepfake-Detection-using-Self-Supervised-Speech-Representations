from src.settings import ROOT_DIR, load_config


def test_config_loads() -> None:
    config = load_config()

    assert config["project"]["task"] == "binary_audio_classification"
    assert config["audio"]["sample_rate"] == 16000
    assert ROOT_DIR.exists()
