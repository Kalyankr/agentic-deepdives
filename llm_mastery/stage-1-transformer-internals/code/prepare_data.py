"""Stage 1 — data prep: download TinyShakespeare to ./data/input.txt.

Run once:
    uv run python prepare_data.py
"""
import os
import urllib.request

# Canonical TinyShakespeare (Karpathy char-rnn) — a single plain-text file.
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    os.makedirs(data_dir, exist_ok=True)
    out = os.path.join(data_dir, "input.txt")
    if os.path.exists(out):
        print(f"already present: {out} ({os.path.getsize(out)} bytes)")
        return
    print("downloading TinyShakespeare ...")
    urllib.request.urlretrieve(URL, out)
    print(f"saved {out} ({os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    main()
