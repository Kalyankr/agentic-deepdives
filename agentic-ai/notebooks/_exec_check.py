"""Execute every code cell of every notebook in-process to catch runtime errors.

Usage (from the agentic-ai folder):
    uv run python notebooks/_exec_check.py

Each notebook runs in its own fresh namespace, cells in order. Any exception fails
the check and prints the offending cell. This guarantees the notebooks are runnable
offline (with the MockLLM) exactly as a learner would experience them.
"""

import glob
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)  # make `import shared` work


def run_notebook(path: str) -> bool:
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    ns: dict = {}
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for i, cell in enumerate(cells):
        src = "".join(cell["source"])
        # The bootstrap cell walks up for shared/; emulate by ensuring ROOT is importable.
        try:
            exec(compile(src, f"{os.path.basename(path)}#cell{i}", "exec"), ns)
        except Exception:
            print(f"\nFAILED {os.path.basename(path)} cell {i}:\n{src}\n")
            traceback.print_exc()
            return False
    print(f"  OK  {os.path.basename(path):42} ({len(cells)} code cells ran)")
    return True


def main() -> int:
    notebooks = sorted(glob.glob(os.path.join(HERE, "*.ipynb")))
    print(f"executing {len(notebooks)} notebooks...\n")
    ok = all(run_notebook(p) for p in notebooks)
    print("\nall runnable." if ok else "\nFAILURES above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
