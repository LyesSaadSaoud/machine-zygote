"""
Run every experiment in order.

Usage::

    python scripts/run_all.py                # full run
    python scripts/run_all.py --smoke-test   # fast reduced run, separate outputs
    python scripts/run_all.py --only exp02_causal_swap

The full pipeline is::

    python scripts/run_all.py
    python scripts/analyze_all.py
    python scripts/make_figures.py
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (Config, LOG_DIR, ensure_dirs, log_line, provenance,
                        save_json, tag)

EXPERIMENTS = (
    "exp01_parental_crosses",
    "exp02_causal_swap",
    "exp03_baselines",
    "exp04_multigeneration",
    "exp05_robustness",
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Machine Zygote — run all experiments")
    ap.add_argument("--smoke-test", action="store_true",
                    help="reduced sample sizes and horizons; writes *_smoke outputs")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only the named experiments")
    args = ap.parse_args()

    cfg = Config()
    if args.smoke_test:
        cfg = cfg.smoke()
    ensure_dirs()

    todo = args.only if args.only else list(EXPERIMENTS)
    unknown = [e for e in todo if e not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"unknown experiment(s): {unknown}")

    log_line(cfg, "run_all", f"master seed {cfg.master_seed}, "
                             f"config digest {cfg.digest()}, "
                             f"smoke={cfg.run.smoke_test}")
    timings = {}
    t_start = time.time()
    for name in todo:
        mod = importlib.import_module(f"experiments.{name}")
        t0 = time.time()
        mod.run(cfg)
        timings[name] = round(time.time() - t0, 2)

    total = round(time.time() - t_start, 2)
    rec = provenance(cfg, "run_all", {"timings_seconds": timings,
                                      "total_seconds": total,
                                      "experiments": todo})
    save_json(os.path.join(LOG_DIR, f"run_all_provenance{tag(cfg)}.json"), rec)
    log_line(cfg, "run_all", f"all experiments finished in {total}s: {timings}")
    print("\nNext:\n  python scripts/analyze_all.py"
          + (" --smoke-test" if cfg.run.smoke_test else "")
          + "\n  python scripts/make_figures.py"
          + (" --smoke-test" if cfg.run.smoke_test else ""))


if __name__ == "__main__":
    main()
