"""
Causal interventions (Condition 7) and ablation conditions (Conditions 4-6).

The swap is a genuine intervention, not a re-sample.  For a fixed
:class:`~src.zygote.Background` -- the recombination mask, both gamete mutation
vectors and the developmental-noise seed -- the child is a *deterministic*
function of the two parental germlines.  Replacing one parent while holding the
background fixed therefore isolates ``do(G_sire := G_k)``.

Three matched arms are run for every background:

``base``
    (dam = A_i, sire = B_j);
``swap_sire``
    (dam = A_i, sire = B_k) -- one paternal germline replaced;
``swap_dam``
    (dam = A_k, sire = B_j) -- one maternal germline replaced;
``null_remut``
    (dam = A_i, sire = B_j) with a *fresh* mutation draw but the same
    recombination mask and developmental seed.  This is the control that says
    how much the phenotype moves when nothing causal is changed;
``null_redev``
    (dam = A_i, sire = B_j) with the same genome but a fresh developmental
    noise seed -- the pure developmental-noise floor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .config import Config
from .germline import GermlineSpec, random_germlines
from .reproduction import Cohort, cross, evaluate_zygotes
from .zygote import Background, draw_background, recombine

SWAP_ARMS = ("base", "swap_sire", "swap_dam", "null_remut", "null_redev")


@dataclass
class SwapExperiment:
    arms: Dict[str, Cohort]
    dam_base: int
    sire_base: int
    dam_alt: int
    sire_alt: int
    n_backgrounds: int


def causal_swap(founders: np.ndarray, spec: GermlineSpec, cfg: Config,
                rng: np.random.Generator, dam_base: int = 0, sire_base: int = 1,
                dam_alt: int = 2, sire_alt: int = 3) -> SwapExperiment:
    """Run the matched-background swap design."""
    n = cfg.run.n_backgrounds_swap
    bg = draw_background(n, spec, rng)

    gd, gs = founders[dam_base], founders[sire_base]
    gda, gsa = founders[dam_alt], founders[sire_alt]

    arms: Dict[str, Cohort] = {}
    arms["base"] = cross(gd, gs, bg, spec, cfg)
    arms["swap_sire"] = cross(gd, gsa, bg, spec, cfg)
    arms["swap_dam"] = cross(gda, gs, bg, spec, cfg)

    bg_remut = Background(
        mask=bg.mask.copy(),
        xi_dam=rng.normal(0.0, spec.gcfg.mutation_sigma, size=bg.xi_dam.shape),
        xi_sire=rng.normal(0.0, spec.gcfg.mutation_sigma, size=bg.xi_sire.shape),
        dev_seed=bg.dev_seed.copy(),
    )
    arms["null_remut"] = cross(gd, gs, bg_remut, spec, cfg)

    bg_redev = Background(bg.mask.copy(), bg.xi_dam.copy(), bg.xi_sire.copy(),
                          rng.integers(0, 2 ** 62, size=n, dtype=np.int64))
    arms["null_redev"] = cross(gd, gs, bg_redev, spec, cfg)

    for name, coh in arms.items():
        coh.meta = {"background": np.arange(n), "arm": np.array([name] * n)}
    return SwapExperiment(arms, dam_base, sire_base, dam_alt, sire_alt, n)


# --------------------------------------------------------------------------
# Ablation / baseline conditions
# --------------------------------------------------------------------------
def random_germline_offspring(spec: GermlineSpec, cfg: Config,
                              rng: np.random.Generator, n: int) -> Cohort:
    """Condition 4: inherited germline replaced by a matched random germline.

    Dimension- and magnitude-matched (each locus is ``+-founder_amplitude``),
    then developed and evaluated through exactly the same pipeline.
    """
    zyg = random_germlines(spec, n, rng)
    seeds = rng.integers(0, 2 ** 62, size=n, dtype=np.int64)
    return evaluate_zygotes(zyg, seeds, spec, cfg, mode="full")


def no_development_offspring(dam_g: np.ndarray, sire_g: np.ndarray,
                             spec: GermlineSpec, cfg: Config,
                             rng: np.random.Generator, n: int,
                             mode: str = "nodev_static") -> Cohort:
    """Condition 5: the same zygotes, but developmental dynamics removed."""
    bg = draw_background(n, spec, rng)
    return cross(dam_g, sire_g, bg, spec, cfg, mode=mode)


def paired_shift(base: Cohort, other: Cohort, scaler) -> np.ndarray:
    """Per-individual standardised phenotype shift between two matched arms."""
    return scaler.transform(other.Y) - scaler.transform(base.Y)
