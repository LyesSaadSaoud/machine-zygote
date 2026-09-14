"""Minimal stdlib test runner (pytest is optional and not required offline).

    python tests/run_tests.py
    python -m pytest tests/        # equivalent, if pytest is installed
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def main() -> int:
    files = sorted(f for f in os.listdir(HERE)
                   if f.startswith("test_") and f.endswith(".py"))
    passed, failed = 0, []
    t0 = time.time()
    for fname in files:
        spec = importlib.util.spec_from_file_location(fname[:-3],
                                                      os.path.join(HERE, fname))
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            failed.append((fname, "<import>", traceback.format_exc()))
            print(f"E {fname} (import failed)")
            continue
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
                print(f". {fname}::{name}")
            except Exception:
                failed.append((fname, name, traceback.format_exc()))
                print(f"F {fname}::{name}")
    dt = time.time() - t0
    print(f"\n{passed} passed, {len(failed)} failed in {dt:.1f}s")
    for fname, name, tb in failed:
        print(f"\n=== FAILED {fname}::{name} ===\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
