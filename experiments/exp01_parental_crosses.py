"""
Experiment 1 — parental crosses (Conditions 1-3).

A complete 4x4 diallel over founders A, B, C, D in which dam and sire roles are
distinct, so ``A x B`` and ``B x A`` are separate cells (Condition 3), the
diagonal supplies the clone controls (Condition 1) and the off-diagonal the
biparental offspring (Condition 2).

Also produces:

* parental *reference* individuals (zygote = founder germline, no
  recombination, no mutation) which define within-line developmental variation
  and the standardisation scaler used by every later analysis;
* stored developmental trajectories for a handful of individuals (Fig. P1-2).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (Config, TRAIT_NAMES, log_line, rng_for, save_raw)
from src.development import develop
from src.germline import (FOUNDER_NAMES, founder_germlines,
                          germline_distance_matrix, make_spec)
from src.reproduction import diallel, parental_reference
from src.zygote import draw_background, recombine

NAME = "exp01_parental_crosses"


def run(cfg: Config) -> str:
    t0 = time.time()
    spec = make_spec(cfg)
    F = founder_germlines(spec, len(FOUNDER_NAMES))
    n_rep = cfg.run.n_seeds_diallel

    log_line(cfg, NAME, f"germline d={spec.d} "
                        f"({int(spec.autosomal_mask.sum())} autosomal, "
                        f"{int(spec.maternal_mask.sum())} maternal)")

    # --- parental reference individuals -----------------------------------
    ref_Y, ref_line, ref_flags = [], [], {}
    ref_soma = []
    for i, nm in enumerate(FOUNDER_NAMES):
        coh = parental_reference(F[i], n_rep, spec, cfg,
                                 rng_for(cfg.master_seed, NAME, "ref", nm),
                                 keep_soma=True)
        ref_Y.append(coh.Y)
        ref_soma.append(coh.soma_vec)
        ref_line.append(np.full(len(coh), i))
        for k, v in coh.flags.items():
            ref_flags.setdefault(k, []).append(v)
    ref_Y = np.concatenate(ref_Y)
    ref_soma = np.concatenate(ref_soma)
    ref_line = np.concatenate(ref_line)
    ref_flags = {k: np.concatenate(v) for k, v in ref_flags.items()}
    log_line(cfg, NAME, f"parental reference individuals: {ref_Y.shape[0]}")

    # --- diallel -----------------------------------------------------------
    coh = diallel(F, spec, cfg, rng_for(cfg.master_seed, NAME, "diallel"), n_rep)
    log_line(cfg, NAME, f"diallel offspring: {len(coh)} "
                        f"({len(FOUNDER_NAMES)}x{len(FOUNDER_NAMES)} cells x {n_rep})")

    # --- developmental trajectories for the schematic figure ---------------
    traj_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    rng_tr = rng_for(cfg.master_seed, NAME, "traj")
    bg = draw_background(len(traj_pairs), spec, rng_tr)
    dam_idx = np.array([p[0] for p in traj_pairs])
    sire_idx = np.array([p[1] for p in traj_pairs])
    zyg = recombine(F[dam_idx], F[sire_idx], bg, spec)
    dev = develop(zyg, spec, cfg, bg.dev_seed, store_traj=True)

    arrays = {
        "Y": coh.Y,
        "dam": coh.meta["dam"],
        "sire": coh.meta["sire"],
        "rep": coh.meta["rep"],
        "zyg_u": coh.zyg_u,
        "frozen": coh.frozen,
        "t_dev": coh.t_dev,
        "module_spread": coh.module_spread,
        "role_diversity": coh.role_div,
        "ref_Y": ref_Y,
        "ref_line": ref_line,
        "ref_soma": ref_soma,
        "founders": F,
        "founder_distance": germline_distance_matrix(F),
        "traj": dev.traj,
        "traj_t": dev.traj_t,
        "traj_dam": dam_idx,
        "traj_sire": sire_idx,
        "traj_frozen": dev.frozen,
        "trait_names": np.array(TRAIT_NAMES),
        "founder_names": np.array(FOUNDER_NAMES),
    }
    for k, v in coh.flags.items():
        arrays[f"flag_{k}"] = v
    for k, v in coh.aux.items():
        arrays[f"aux_{k}"] = v
    for k, v in ref_flags.items():
        arrays[f"ref_flag_{k}"] = v

    path = save_raw(NAME, cfg, arrays,
                    extra={"n_rep": n_rep, "n_founders": len(FOUNDER_NAMES),
                           "wall_seconds": time.time() - t0})
    log_line(cfg, NAME, f"done in {time.time() - t0:.1f}s -> {path}")
    return path


if __name__ == "__main__":
    cfg = Config()
    if "--smoke-test" in sys.argv:
        cfg = cfg.smoke()
    run(cfg)
