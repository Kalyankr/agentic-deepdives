"""Tiny helper to build valid .ipynb (nbformat 4.5) notebooks from Python.

Using a generator keeps every notebook consistent and guarantees valid JSON
(Python's json module handles all escaping). Each builder under ``src/`` calls
``md(...)`` / ``code(...)`` to make cells and ``write(cells, name)`` to emit the
notebook into the ``notebooks/`` folder.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent


def _source(text: str) -> list[str]:
    """nbformat wants source as a list of lines, each keeping its trailing \\n."""
    return text.strip("\n").splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _source(text),
    }


def write(cells: list[dict], name: str) -> Path:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NOTEBOOKS_DIR / name
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path.name}  ({len(cells)} cells)")
    return path
