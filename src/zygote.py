"""
Zygote formation  ``Z_C = R(G_A, G_B, xi)``.

The recombination operator is deliberately *explicit* about the sources of
stochasticity, because the causal-swap intervention (exp02) requires the
ability to hold every stochastic draw fixed while changing exactly one parent.

A :class:`Background` bundles all random draws that define a "developmental
background": the recombination mask, both gamete mutation vectors and the
developmental noise seed.  Given a fixed background, ``Z_C`` is a deterministic
function of the two parental germlines, which is what makes the swap a genuine
intervention rather than a re-sample.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .germline import GermlineSpec


@dataclass
class Background:
    """All exogenous randomness of one reproduction event."""

    mask: np.ndarray        # (B, d) bool; True -> locus taken from the dam
    xi_dam: np.ndarray      # (B, d) gamete mutation of the dam
    xi_sire: np.ndarray     # (B, d) gamete mutation of the sire
    dev_seed: np.ndarray    # (B,) int64 seeds for the developmental noise field

    def __len__(self) -> int:
        return self.mask.shape[0]

    def subset(self, idx) -> "Background":
        return Background(self.mask[idx], self.xi_dam[idx], self.xi_sire[idx],
                          self.dev_seed[idx])


def draw_background(n: int, spec: GermlineSpec, rng: np.random.Generator,
                    mutation_sigma: Optional[float] = None) -> Background:
    """Sample ``n`` independent reproduction backgrounds."""
    sigma = spec.gcfg.mutation_sigma if mutation_sigma is None else mutation_sigma
    p = spec.gcfg.recombination_p
    mask = rng.random((n, spec.d)) < p
    xi_dam = rng.normal(0.0, sigma, size=(n, spec.d))
    xi_sire = rng.normal(0.0, sigma, size=(n, spec.d))
    dev_seed = rng.integers(0, 2 ** 62, size=n, dtype=np.int64)
    return Background(mask, xi_dam, xi_sire, dev_seed)


def gametes(g_dam: np.ndarray, g_sire: np.ndarray, bg: Background
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Meiotic products: parental germline plus controlled mutation ``xi``."""
    return (np.clip(np.atleast_2d(g_dam) + bg.xi_dam, -1.0, 1.0),
            np.clip(np.atleast_2d(g_sire) + bg.xi_sire, -1.0, 1.0))


def recombine(g_dam: np.ndarray, g_sire: np.ndarray, bg: Background,
              spec: GermlineSpec) -> np.ndarray:
    """``Z_C = R(G_dam, G_sire, xi)`` in normalised locus coordinates.

    Autosomal loci follow the Bernoulli recombination mask; maternal-channel
    loci are taken from the dam's gamete unconditionally.
    """
    gd, gs = gametes(g_dam, g_sire, bg)
    take_dam = bg.mask | spec.maternal_mask[None, :]
    return np.where(take_dam, gd, gs)


def selfed_zygote(g: np.ndarray, bg: Background, spec: GermlineSpec) -> np.ndarray:
    """Clone / self-cross ``G x G``: the same germline supplies both gametes.

    Note this is *not* a copy of the parent: the two gametes carry independent
    mutations, so within-line variation is generated exactly as in an outcross.
    """
    return recombine(g, g, bg, spec)


def clonal_reference(g: np.ndarray, n: int) -> np.ndarray:
    """The founder germline itself, used as the zygote of the parental
    reference individuals (no recombination, no mutation)."""
    return np.repeat(np.atleast_2d(g), n, axis=0)


def dam_fraction(bg: Background, spec: GermlineSpec) -> np.ndarray:
    """Realised fraction of loci inherited from the dam (diagnostic)."""
    take_dam = bg.mask | spec.maternal_mask[None, :]
    return take_dam.mean(axis=1)
