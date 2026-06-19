"""Stage 1 — Lab A: byte-level BPE tokenizer (build it yourself).

Goal: understand tokenization as the model's atom. You implement the merge
algorithm; the scaffolding gives you the structure and a self-test.

Quick test once implemented:
    uv run python bpe.py

Key idea (BPE):
  1. Start from raw bytes (0..255) — every string is representable, no <unk>.
  2. Repeatedly find the most frequent adjacent pair of tokens and merge it
     into a new token id (256, 257, ...), until you reach vocab_size.
  3. encode() replays those merges greedily; decode() maps ids -> bytes -> text.
"""
from __future__ import annotations


def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """Count how often each adjacent pair appears in `ids`.

    Example: [1, 2, 1, 2, 3] -> {(1,2): 2, (2,1): 1, (2,3): 1}

    TODO(stage1-labA): build and return the counts dict.
    """
    raise NotImplementedError("Implement get_stats (Lab A)")


def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """Return a new list where every occurrence of `pair` is replaced by `idx`.

    Example: merge([1,2,1,2,3], (1,2), 256) -> [256, 256, 3]

    TODO(stage1-labA): walk through ids, collapsing matches of `pair`.
    """
    raise NotImplementedError("Implement merge (Lab A)")


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}  # (a, b) -> new_id
        self.vocab: dict[int, bytes] = {}             # id -> bytes

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Learn `vocab_size - 256` merges from `text`.

        TODO(stage1-labA):
          1. ids = list(text.encode("utf-8"))
          2. for i in range(vocab_size - 256):
               - stats = get_stats(ids)
               - pick the most frequent pair
               - new_id = 256 + i; record self.merges[pair] = new_id
               - ids = merge(ids, pair, new_id)
          3. build self.vocab (ids 0..255 map to single bytes, then add merges)
        """
        assert vocab_size >= 256
        raise NotImplementedError("Implement BPETokenizer.train (Lab A)")

    def encode(self, text: str) -> list[int]:
        """Bytes -> ids, greedily applying learned merges in order.

        TODO(stage1-labA): repeatedly find the mergeable pair with the LOWEST
        merge id present in the sequence and apply it, until none remain.
        """
        raise NotImplementedError("Implement BPETokenizer.encode (Lab A)")

    def decode(self, ids: list[int]) -> str:
        """Ids -> bytes -> utf-8 string.

        TODO(stage1-labA): concatenate self.vocab[id] for each id, then
        .decode('utf-8', errors='replace').
        """
        raise NotImplementedError("Implement BPETokenizer.decode (Lab A)")


if __name__ == "__main__":
    tok = BPETokenizer()
    sample = "the quick brown fox jumps over the lazy dog. " * 30
    tok.train(sample, vocab_size=300)
    roundtrip = tok.decode(tok.encode("the lazy fox"))
    assert roundtrip == "the lazy fox", f"round-trip failed: {roundtrip!r}"
    print("BPE round-trip OK ✓")
