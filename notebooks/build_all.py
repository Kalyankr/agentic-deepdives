"""Build every notebook from its generator and validate the output JSON.

Usage (from the repo root):
    uv run --project labs python notebooks/build_all.py

Each generator in ``src/nb_XX_*.py`` writes a ``.ipynb`` into this folder. We run them
as subprocesses (identical to running each individually) and then load every notebook to
confirm the JSON is valid.
"""

import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")


def build_all():
    builders = sorted(glob.glob(os.path.join(SRC, "nb_*.py")))
    print(f"building {len(builders)} notebooks...\n")
    for path in builders:
        name = os.path.basename(path)
        print(f"  -> {name}")
        subprocess.check_call([sys.executable, name], cwd=SRC)
    return builders


def validate_all():
    notebooks = sorted(glob.glob(os.path.join(HERE, "*.ipynb")))
    print(f"\nvalidating {len(notebooks)} notebooks...")
    for path in notebooks:
        with open(path, encoding="utf-8") as f:
            nb = json.load(f)
        assert nb["nbformat"] == 4, f"{path}: unexpected nbformat"
        assert nb["cells"], f"{path}: no cells"
        print(f"  OK  {os.path.basename(path):42} ({len(nb['cells'])} cells)")
    print(f"\nall {len(notebooks)} notebooks built and valid.")


if __name__ == "__main__":
    build_all()
    validate_all()
