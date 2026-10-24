#!/usr/bin/env python3
"""Deprecated entry point — the corpus builder now handles every dataset.

Kept because README, DECISIONS and the ingest script all document this path.
It delegates to scripts/build_corpus.py, which enforces the same invariants
plus the cross-dataset ones v3/v4 coexistence needs.
"""
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    print("note: build_v3_corpus.py now delegates to build_corpus.py "
          "--dataset v3\n")
    sys.argv = [str(REPO / "scripts" / "build_corpus.py"), "--dataset", "v3"]
    runpy.run_path(str(REPO / "scripts" / "build_corpus.py"),
                   run_name="__main__")
