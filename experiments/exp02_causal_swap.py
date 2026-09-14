"""
Experiment 2 — causal germline swap (Condition 7).

For every one of ``n_backgrounds_swap`` fixed developmental backgrounds the
same child is produced five ways: unchanged, with the paternal germline
replaced, with the maternal germline replaced, with a fresh mutation draw, and
with a fresh developmental-noise seed.  Because the background is held fixed,
the difference between ``base`` and ``swap_*`` is the effect of the
intervention ``do(G := G_alt)`` and nothing else.

The two null arms are what licence the word *causal*: they measure how far the
phenotype moves when nothing causal is changed.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config, TRAIT_NAMES, log_line, rng_for, save_raw
from src.germline import FOUNDER_NAMES, founder_germlines, make_spec
from src.interventions import SWAP_ARMS, causal_swap

NAME = "exp02_causal_swap"


def run(cfg: Config) -> str:
    t0 = time.time()
    spec = make_spec(cfg)
    F = founder_germlines(spec, len(FOUNDER_NAMES))
    rng = rng_for(cfg.master_seed, NAME)

    exp = causal_swap(F, spec, cfg, rng, dam_base=0, sire_base=1,
                      dam_alt=2, sire_alt=3)
    log_line(cfg, NAME,
             f"{exp.n_backgrounds} matched backgrounds x {len(SWAP_ARMS)} arms "
             f"(base = {FOUNDER_NAMES[exp.dam_base]}x{FOUNDER_NAMES[exp.sire_base]}, "
             f"alt dam = {FOUNDER_NAMES[exp.dam_alt]}, "
             f"alt sire = {FOUNDER_NAMES[exp.sire_alt]})")

    arrays = {
        "arm_names": np.array(SWAP_ARMS),
        "trait_names": np.array(TRAIT_NAMES),
        "founder_names": np.array(FOUNDER_NAMES),
        "dam_base": exp.dam_base, "sire_base": exp.sire_base,
        "dam_alt": exp.dam_alt, "sire_alt": exp.sire_alt,
    }
    for arm in SWAP_ARMS:
        coh = exp.arms[arm]
        arrays[f"Y_{arm}"] = coh.Y
        arrays[f"zyg_{arm}"] = coh.zyg_u
        arrays[f"spread_{arm}"] = coh.module_spread
        arrays[f"roles_{arm}"] = coh.role_div
        for k, v in coh.flags.items():
            arrays[f"flag_{arm}_{k}"] = v

    path = save_raw(NAME, cfg, arrays,
                    extra={"n_backgrounds": exp.n_backgrounds,
                           "arms": list(SWAP_ARMS),
                           "wall_seconds": time.time() - t0})
    log_line(cfg, NAME, f"done in {time.time() - t0:.1f}s -> {path}")
    return path


if __name__ == "__main__":
    cfg = Config()
    if "--smoke-test" in sys.argv:
        cfg = cfg.smoke()
    run(cfg)
