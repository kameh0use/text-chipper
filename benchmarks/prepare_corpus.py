"""Download WikiText-103 (raw) and write it as one UTF-8 text file.

Usage: python benchmarks/prepare_corpus.py corpus.txt
Prints the size and SHA-256 of the file so the exact input can be reproduced.
"""
import hashlib
import sys

from datasets import load_dataset

out = sys.argv[1] if len(sys.argv) > 1 else "corpus.txt"

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
h = hashlib.sha256()
size = 0
with open(out, "w", encoding="utf-8", newline="") as f:
    for row in ds:
        text = row["text"]
        f.write(text)
        b = text.encode("utf-8")
        h.update(b)
        size += len(b)

print(f"wrote {out}: {size / (1024 * 1024):.1f} MiB, sha256={h.hexdigest()}")
