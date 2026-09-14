"""
Experiment 3 — baselines and ablations (Conditions 1, 2, 4, 5, 6).

All arms use the same founders, the same sample size and the same evaluation
protocol, so their phenotype distributions are directly comparable:

============================  ====================================================
``biparental``                A x B with full development (the model)
``clone_A`` / ``clone_B``     A x A, B x B                       (Condition 1)
``random_germline``           magnitude-matched random genome    (Condition 4)
``nodev_static``              A x B zygotes, no development      (Condition 5)
``nodev_quasistatic``         A x B zygotes, positional read-out only
``direct_controller``         parental controllers crossed directly (Condition 6)
============================  ====================================================

``nodev_quasistatic`` is the informative developmental ablation:
``nodev_static`` cannot differentiate modules *at all*, so part of its collapse
is structural rather than empirical, and the report says so.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config, TRAIT_NAMES, log_line, rng_for, save_raw
from src.germline import FOUNDER_NAMES, founder_germlines, make_spec
from src.interventions import no_development_offspring, random_germline_offspring
from src.reproduction import (cross, diallel, direct_controller_cross,
                             parental_reference)
from src.zygote import draw_background

NAME = "exp03_baselines"
CONDITIONS = ("biparental", "clone_A", "clone_B", "random_germline",
              "nodev_static", "nodev_quasistatic", "direct_controller")


def run(cfg: Config) -> str:
    t0 = time.time()
    spec = make_spec(cfg)
    F = founder_germlines(spec, len(FOUNDER_NAMES))
    n = cfg.run.n_seeds_baseline
    A, B = F[0], F[1]
    out = {}

    def bg_for(label):
        return draw_background(n, spec, rng_for(cfg.master_seed, NAME, label))

    out["biparental"] = cross(A, B, bg_for("biparental"), spec, cfg)
    out["clone_A"] = cross(A, A, bg_for("clone_A"), spec, cfg)
    out["clone_B"] = cross(B, B, bg_for("clone_B"), spec, cfg)
    out["random_germline"] = random_germline_offspring(
        spec, cfg, rng_for(cfg.master_seed, NAME, "random"), n)
    out["nodev_static"] = no_development_offspring(
        A, B, spec, cfg, rng_for(cfg.master_seed, NAME, "nodev_static"), n,
        mode="nodev_static")
    out["nodev_quasistatic"] = no_development_offspring(
        A, B, spec, cfg, rng_for(cfg.master_seed, NAME, "nodev_quasi"), n,
        mode="nodev_quasistatic")

    # Controller pools for the direct-controller baseline: reference
    # individuals of each parental line, developed normally.
    pool_A = parental_reference(A, n, spec, cfg,
                                rng_for(cfg.master_seed, NAME, "poolA"),
                                keep_soma=True)
    pool_B = parental_reference(B, n, spec, cfg,
                                rng_for(cfg.master_seed, NAME, "poolB"),
                                keep_soma=True)
    out["direct_controller"] = direct_controller_cross(
        pool_A.soma_vec, pool_B.soma_vec, n, spec, cfg,
        rng_for(cfg.master_seed, NAME, "direct"))

    # --- ablated diallels --------------------------------------------------
    # H2 asks whether removing development changes *phenotype organisation*,
    # not merely the location of one cross.  That requires re-running the whole
    # factorial design with development removed, so that PDVF and the parental
    # effect structure can be compared like for like against exp01.
    arrays_dia = {}
    for abl in ("nodev_static", "nodev_quasistatic"):
        dia = diallel(F, spec, cfg, rng_for(cfg.master_seed, NAME, "diallel", abl),
                      cfg.run.n_seeds_diallel, mode=abl)
        arrays_dia[f"diallel_{abl}_Y"] = dia.Y
        arrays_dia[f"diallel_{abl}_dam"] = dia.meta["dam"]
        arrays_dia[f"diallel_{abl}_sire"] = dia.meta["sire"]
        arrays_dia[f"diallel_{abl}_roles"] = dia.role_div
        log_line(cfg, NAME, f"ablated diallel {abl}: n={len(dia)}")

    arrays = {
        "condition_names": np.array(CONDITIONS),
        "trait_names": np.array(TRAIT_NAMES),
        "pool_A_Y": pool_A.Y, "pool_B_Y": pool_B.Y,
        **arrays_dia,
    }
    for cname in CONDITIONS:
        coh = out[cname]
        arrays[f"Y_{cname}"] = coh.Y
        arrays[f"spread_{cname}"] = coh.module_spread
        arrays[f"roles_{cname}"] = coh.role_div
        for k, v in coh.flags.items():
            arrays[f"flag_{cname}_{k}"] = v
        log_line(cfg, NAME, f"{cname}: n={len(coh)} "
                            f"role_diversity={coh.role_div.mean():.2f}")

    path = save_raw(NAME, cfg, arrays,
                    extra={"n_per_condition": n,
                           "conditions": list(CONDITIONS),
                           "wall_seconds": time.time() - t0})
    log_line(cfg, NAME, f"done in {time.time() - t0:.1f}s -> {path}")
    return path


if __name__ == "__main__":
    cfg = Config()
    if "--smoke-test" in sys.argv:
        cfg = cfg.smoke()
    run(cfg)
