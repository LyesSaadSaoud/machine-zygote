"""
Experiment 5 — parameter robustness.

The central claim (H1: both parents contribute to the newborn phenotype) is
re-tested from scratch at every point of a four-way sweep:

* developmental noise amplitude   x0.25 ... x4
* inter-module coupling (diffusion) x0 ... x4
* mutation amplitude              0 ... 0.20 (normalised locus units)
* developmental duration          x0.25 ... x2

At each sweep point a fresh 2x2 diallel over founders A and B is run with
``n_seeds_robustness`` replicates per cell and the dam/sire effects are
re-estimated.  Sweep points where the effect disappears are reported, not
hidden: that is the point of the experiment.

Only parametric F-tests are used here (5000-fold permutation at every sweep
point would dominate the runtime); the primary analysis in exp01 uses
permutation tests.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config, TRAIT_NAMES, log_line, rng_for, save_raw
from src.germline import FOUNDER_NAMES, founder_germlines, make_spec
from src.reproduction import diallel
from src.statistics import pdvf, two_way_anova

NAME = "exp05_robustness"

SWEEPS = {
    "dev_noise_scale": (0.25, 0.5, 1.0, 2.0, 4.0),
    "coupling_scale": (0.0, 0.5, 1.0, 2.0, 4.0),
    "mutation_sigma": (0.0, 0.025, 0.05, 0.10, 0.20),
    "dev_duration_scale": (0.25, 0.5, 1.0, 1.5, 2.0),
}


def _configure(cfg: Config, knob: str, value: float) -> Config:
    if knob == "dev_noise_scale":
        return replace(cfg, development=replace(cfg.development, noise_scale=value))
    if knob == "coupling_scale":
        return replace(cfg, development=replace(cfg.development,
                                                diffusion_scale=value))
    if knob == "dev_duration_scale":
        return replace(cfg, development=replace(cfg.development,
                                                duration_scale=value))
    if knob == "mutation_sigma":
        return replace(cfg, germline=replace(cfg.germline, mutation_sigma=value))
    raise ValueError(knob)


def run(cfg: Config) -> str:
    t0 = time.time()
    n_rep = cfg.run.n_seeds_robustness
    rows = []

    for knob, values in SWEEPS.items():
        for value in values:
            cfg_k = _configure(cfg, knob, value)
            spec = make_spec(cfg_k)
            F = founder_germlines(spec, len(FOUNDER_NAMES))[:2]
            rng = rng_for(cfg.master_seed, NAME, knob, value)
            coh = diallel(F, spec, cfg_k, rng, n_rep)
            dam, sire = coh.meta["dam"], coh.meta["sire"]
            for ti, tname in enumerate(TRAIT_NAMES):
                y = coh.Y[:, ti]
                if not np.all(np.isfinite(y)) or np.allclose(y, y[0]):
                    rows.append((knob, value, tname, np.nan, np.nan, np.nan,
                                 np.nan, np.nan, np.nan, np.nan, len(coh)))
                    continue
                a = two_way_anova(y, dam, sire, tname)
                h = pdvf(y, dam, sire, tname)
                rows.append((knob, value, tname,
                             a.p_param["dam"], a.p_param["sire"],
                             a.p_param["dam:sire"],
                             a.partial_eta2["dam"], a.partial_eta2["sire"],
                             a.partial_eta2["dam:sire"],
                             h.pdvf_components, len(coh)))
            log_line(cfg, NAME, f"{knob}={value}: {len(coh)} offspring, "
                                f"mean role diversity {coh.role_div.mean():.2f}")

    arrays = {
        "knob": np.array([r[0] for r in rows]),
        "value": np.array([r[1] for r in rows], float),
        "trait": np.array([r[2] for r in rows]),
        "p_dam": np.array([r[3] for r in rows], float),
        "p_sire": np.array([r[4] for r in rows], float),
        "p_interaction": np.array([r[5] for r in rows], float),
        "eta2_dam": np.array([r[6] for r in rows], float),
        "eta2_sire": np.array([r[7] for r in rows], float),
        "eta2_interaction": np.array([r[8] for r in rows], float),
        "pdvf": np.array([r[9] for r in rows], float),
        "n": np.array([r[10] for r in rows], int),
        "trait_names": np.array(TRAIT_NAMES),
    }
    path = save_raw(NAME, cfg, arrays,
                    extra={"sweeps": {k: list(v) for k, v in SWEEPS.items()},
                           "n_rep_per_cell": n_rep,
                           "test": "parametric F (permutation used in exp01)",
                           "wall_seconds": time.time() - t0})
    log_line(cfg, NAME, f"done in {time.time() - t0:.1f}s -> {path}")
    return path


if __name__ == "__main__":
    cfg = Config()
    if "--smoke-test" in sys.argv:
        cfg = cfg.smoke()
    run(cfg)
