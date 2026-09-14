"""
Experiment 4 — three generations, G0 -> G1 -> G2.

A G1 child's germline *is* its zygote genome, so it can supply gametes to G2
exactly as a founder does.  Nothing else changes: the same recombination
operator, the same mutation amplitude, the same development.

There is **no selection**: every offspring produced is retained and no fitness
criterion is applied anywhere.  Any change across generations is recombination
plus mutation plus developmental noise -- drift, not evolution.  The report
states this explicitly.

Measured: parent-offspring resemblance (midparent regression), germline drift
away from the founder set, lineage distance, and developmental stability
(within-family phenotype dispersion) per generation.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config, TRAIT_NAMES, log_line, rng_for, save_raw
from src.germline import FOUNDER_NAMES, founder_germlines, make_spec
from src.reproduction import cross, parental_reference
from src.zygote import draw_background

NAME = "exp04_multigeneration"

#: G1 matings, as (dam founder index, sire founder index).
G1_MATINGS = ((0, 1), (2, 3), (0, 2), (1, 3))


def run(cfg: Config) -> str:
    t0 = time.time()
    spec = make_spec(cfg)
    F = founder_germlines(spec, len(FOUNDER_NAMES))
    n_fam = cfg.run.n_family_per_gen
    n_lin = cfg.run.n_lineages

    rec = {k: [] for k in
           ("lineage", "generation", "family", "individual",
            "dam_id", "sire_id", "Y", "germline", "dam_Y", "sire_Y")}

    for lin in range(n_lin):
        rng = rng_for(cfg.master_seed, NAME, "lineage", lin)

        # ---- G0: founder reference individuals ---------------------------
        g0_g, g0_Y, g0_id = [], [], []
        for fi in range(len(FOUNDER_NAMES)):
            coh = parental_reference(F[fi], n_fam, spec, cfg, rng)
            g0_g.append(np.repeat(F[fi][None, :], n_fam, axis=0))
            g0_Y.append(coh.Y)
            ids = [f"L{lin}G0F{fi}I{i}" for i in range(n_fam)]
            g0_id.extend(ids)
            for i in range(n_fam):
                rec["lineage"].append(lin); rec["generation"].append(0)
                rec["family"].append(fi); rec["individual"].append(ids[i])
                rec["dam_id"].append(""); rec["sire_id"].append("")
                rec["Y"].append(coh.Y[i]); rec["germline"].append(F[fi])
                rec["dam_Y"].append(np.full(len(TRAIT_NAMES), np.nan))
                rec["sire_Y"].append(np.full(len(TRAIT_NAMES), np.nan))
        g0_g = np.concatenate(g0_g)
        g0_Y = np.concatenate(g0_Y)

        # ---- G1 ----------------------------------------------------------
        g1_g, g1_Y, g1_id, g1_fam = [], [], [], []
        for fam, (di, si) in enumerate(G1_MATINGS):
            # the specific G0 individuals acting as parents
            dam_sel = rng.integers(0, n_fam)
            sire_sel = rng.integers(0, n_fam)
            dam_id = f"L{lin}G0F{di}I{dam_sel}"
            sire_id = f"L{lin}G0F{si}I{sire_sel}"
            dam_Y = g0_Y[di * n_fam + dam_sel]
            sire_Y = g0_Y[si * n_fam + sire_sel]

            bg = draw_background(n_fam, spec, rng)
            coh = cross(F[di], F[si], bg, spec, cfg)
            ids = [f"L{lin}G1F{fam}I{i}" for i in range(n_fam)]
            g1_g.append(coh.zyg_u); g1_Y.append(coh.Y)
            g1_id.extend(ids); g1_fam.extend([fam] * n_fam)
            for i in range(n_fam):
                rec["lineage"].append(lin); rec["generation"].append(1)
                rec["family"].append(fam); rec["individual"].append(ids[i])
                rec["dam_id"].append(dam_id); rec["sire_id"].append(sire_id)
                rec["Y"].append(coh.Y[i]); rec["germline"].append(coh.zyg_u[i])
                rec["dam_Y"].append(dam_Y); rec["sire_Y"].append(sire_Y)
        g1_g = np.concatenate(g1_g)
        g1_Y = np.concatenate(g1_Y)

        # ---- G2: G1 individuals reproduce --------------------------------
        n_g1_fam = len(G1_MATINGS)
        for fam in range(n_g1_fam):
            other = (fam + 1) % n_g1_fam
            dam_sel = rng.integers(0, n_fam)
            sire_sel = rng.integers(0, n_fam)
            dam_row = fam * n_fam + dam_sel
            sire_row = other * n_fam + sire_sel
            dam_id, sire_id = g1_id[dam_row], g1_id[sire_row]
            dam_Y, sire_Y = g1_Y[dam_row], g1_Y[sire_row]

            bg = draw_background(n_fam, spec, rng)
            coh = cross(g1_g[dam_row], g1_g[sire_row], bg, spec, cfg)
            for i in range(n_fam):
                rec["lineage"].append(lin); rec["generation"].append(2)
                rec["family"].append(fam)
                rec["individual"].append(f"L{lin}G2F{fam}I{i}")
                rec["dam_id"].append(dam_id); rec["sire_id"].append(sire_id)
                rec["Y"].append(coh.Y[i]); rec["germline"].append(coh.zyg_u[i])
                rec["dam_Y"].append(dam_Y); rec["sire_Y"].append(sire_Y)

        log_line(cfg, NAME, f"lineage {lin} complete "
                            f"({n_fam * (len(FOUNDER_NAMES) + 2 * n_g1_fam)} individuals)")

    arrays = {
        "lineage": np.array(rec["lineage"]),
        "generation": np.array(rec["generation"]),
        "family": np.array(rec["family"]),
        "individual": np.array(rec["individual"]),
        "dam_id": np.array(rec["dam_id"]),
        "sire_id": np.array(rec["sire_id"]),
        "Y": np.array(rec["Y"]),
        "germline": np.array(rec["germline"]),
        "dam_Y": np.array(rec["dam_Y"]),
        "sire_Y": np.array(rec["sire_Y"]),
        "founders": F,
        "trait_names": np.array(TRAIT_NAMES),
        "founder_names": np.array(FOUNDER_NAMES),
        "selection_applied": np.array(False),
        "g1_matings": np.array(G1_MATINGS),
    }
    path = save_raw(NAME, cfg, arrays,
                    extra={"n_lineages": n_lin, "n_family": n_fam,
                           "g1_matings": [list(m) for m in G1_MATINGS],
                           "selection": "none",
                           "wall_seconds": time.time() - t0})
    log_line(cfg, NAME, f"done in {time.time() - t0:.1f}s -> {path}")
    return path


if __name__ == "__main__":
    cfg = Config()
    if "--smoke-test" in sys.argv:
        cfg = cfg.smoke()
    run(cfg)
