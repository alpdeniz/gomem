#!/usr/bin/env python3
"""End-to-end gomem demo from Python: remember -> recall -> stats -> close.

Mirrors examples/basic_recall (the Go demo) through the ctypes bridge.
It uses the zero-network "offline" embedder, so it runs anywhere with
no llama-server.

Prerequisites::

    sh python/build.sh                 # produces python/libgomem.dylib (macOS) / .so (Linux)

Run::

    python3 examples/python_demo/demo.py
"""
from __future__ import annotations

import os
import sys
import tempfile

# Make the bridge client importable regardless of the CWD.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))

from gomem import Memory  # noqa: E402


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="gomem-py-demo-")
    path = os.path.join(tmp, "demo.db")

    mem = Memory(path, provider="offline", log_level="warn")
    print(f"opened {path}")

    print("\n== remembering items ==")
    # Items start hot (tier 2) unless you pass tier=; they then decay through
    # the tiers on reflect(). Pass tier=0 to store one cold.
    items = [
        dict(
            kind="fact",
            title="Composite indexes speed up filtered queries",
            body="Adding a composite index aligned with WHERE and ORDER BY clauses avoids table scans and sorts.",
            tags=["database", "index"],
            source="run", run_id="demo-run", confidence=0.9,
        ),
        dict(
            kind="lesson",
            title="Cache hot reads to cut redundant work",
            body="Memoizing frequently read, slowly changing results reduces repeated computation and lowers latency.",
            tags=["cache", "performance"],
            source="run", run_id="demo-run", confidence=0.8,
        ),
        dict(
            kind="skill",
            title="Profile before optimizing",
            body="Measure where time actually goes with a profiler; optimize the measured hotspot rather than guessing.",
            tags=["profiling", "performance"],
            source="human", confidence=0.7,
        ),
        dict(
            kind="fact",
            title="Connection pooling limits setup cost",
            body="Reusing a bounded pool of database connections amortizes expensive connect/auth handshakes across many queries.",
            tags=["database", "connection"],
            source="run", run_id="demo-run", confidence=0.85,
        ),
    ]
    for it in items:
        iid = mem.remember(**it)
        print(f"  stored {it['title']:<45} id={iid}")

    print('\n== recalling for goal: "speed up slow database queries" ==')
    hits = mem.recall("speed up slow database queries", top_k=5)
    if not hits:
        print("  (no hits)")
    for rank, h in enumerate(hits, 1):
        print(f"  #{rank} score={h['score']:.3f} [{h['kind']}] {h['title']}")

    print("\n== stats ==")
    st = mem.stats()
    print(f"  total items: {st['total_items']}")
    for k, c in sorted(st.get("by_kind", {}).items()):
        print(f"    kind {k:<10}: {c}")
    for t, c in sorted((st.get("by_tier") or {}).items()):
        print(f"    tier {t:<10}: {c}")
    print(f"  db size: {st['db_size_bytes']} bytes")

    print("\n== reflect (housekeeping) ==")
    rep = mem.reflect("run_end")
    print(f"  decayed={rep['decayed']} evicted={rep['evicted']} "
          f"purged={rep['purged']} ({rep['duration_ms']} ms)")

    mem.close()
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
