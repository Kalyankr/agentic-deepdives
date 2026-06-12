"""Execute every notebook's code cells to catch runtime errors.

Cells that need an optional dependency (torch / matplotlib) are skipped gracefully so the
pure-numpy teaching code is still verified. Run from the repo root:

    uv run --project labs python notebooks/_exec_check.py
"""

import glob
import json
import os
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))

OPTIONAL = ("torch", "matplotlib")


def run_notebook(path):
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    ns = {"__name__": "__nb__"}
    skipped = 0
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if any(f"import {m}" in src for m in OPTIONAL):
            skipped += 1
            continue
        try:
            exec(compile(src, f"{os.path.basename(path)}#cell{i}", "exec"), ns)
        except Exception:
            return f"FAIL at cell {i}\n{traceback.format_exc()}"
    return "OK" if not skipped else f"OK ({skipped} optional-dep cell(s) skipped)"


if __name__ == "__main__":
    failures = 0
    for path in sorted(glob.glob(os.path.join(HERE, "*.ipynb"))):
        status = run_notebook(path)
        print(f"{os.path.basename(path):42} {status.splitlines()[0]}")
        if status.startswith("FAIL"):
            failures += 1
            print(status)
    print(
        f"\n{'all runnable cells passed' if not failures else f'{failures} notebook(s) failed'}"
    )
