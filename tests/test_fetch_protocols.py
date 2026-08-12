from pathlib import Path

from scripts.fetch_asvspoof2019_protocols import file_sha256


def test_file_sha256(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_bytes(b"audio-deepfake\n")

    assert file_sha256(sample) == "dec25f573db29e8d2802a6579b4085247d297cf9e70f8a9a1819d81bb46ccace"
