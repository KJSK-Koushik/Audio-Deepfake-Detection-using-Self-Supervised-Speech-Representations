"""Report whether the local machine is ready for project development."""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from importlib import metadata

REQUIRED_PACKAGES = {
    "librosa": "librosa",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "PyYAML": "yaml",
    "scikit-learn": "sklearn",
    "soundfile": "soundfile",
    "streamlit": "streamlit",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "transformers": "transformers",
}


def package_status(distribution: str, module: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module)
        return True, metadata.version(distribution)
    except (ImportError, OSError, metadata.PackageNotFoundError) as error:
        return False, str(error).splitlines()[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with an error when a required package is unavailable.",
    )
    args = parser.parse_args()

    print(f"Operating system : {platform.platform()}")
    print(f"Python           : {platform.python_version()}")
    print(f"Executable       : {sys.executable}")
    print("\nRequired packages:")

    missing = []
    for distribution, module in REQUIRED_PACKAGES.items():
        available, detail = package_status(distribution, module)
        marker = "OK" if available else "MISSING"
        print(f"  {marker:7} {distribution:16} {detail}")
        if not available:
            missing.append(distribution)

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"\nCUDA available   : {cuda_available}")
        if cuda_available:
            print(f"CUDA device      : {torch.cuda.get_device_name(0)}")
        else:
            print("Training note    : CPU is enough for early phases; use a GPU for Phase 4.")
    except (ImportError, OSError):
        print("\nCUDA available   : unknown (PyTorch is unavailable)")

    if missing:
        print("\nMissing packages : " + ", ".join(missing))
        return 1 if args.strict else 0

    print("\nEnvironment is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
