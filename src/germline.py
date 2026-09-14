"""
Germline state G.

The germline is a vector of *loci* in normalised coordinates ``u in [-1, 1]``.
Each locus has a documented biological range and an inheritance *channel*:

``autosomal``
    inherited from either parent with probability ``recombination_p``;
``maternal``
    always inherited from the dam (a cytoplasmic / maternally deposited
    determinant).  This channel is what makes reciprocal crosses potentially
    non-equivalent; whether it produces a *detectable* effect is an empirical
    question tested in exp01, not an assumption.

Critically, the germline never acts as a controller.  It parameterises a
regulatory dynamical system whose *output over developmental time* is what
configures the soma (see :mod:`development` and :mod:`soma`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import Config, GermlineConfig

FOUNDER_NAMES: Tuple[str, ...] = ("A", "B", "C", "D")


@dataclass(frozen=True)
class Locus:
    name: str
    start: int
    size: int
    lo: float
    hi: float
    channel: str  # "autosomal" | "maternal"

    @property
    def slice(self) -> slice:
        return slice(self.start, self.start + self.size)


class GermlineSpec:
    """Locus table for a given :class:`GermlineConfig`."""

    def __init__(self, gcfg: GermlineConfig):
        self.gcfg = gcfg
        K = gcfg.n_genes
        loci: List[Locus] = []
        cursor = 0

        def add(name: str, size: int, lo: float, hi: float, channel: str = "autosomal"):
            nonlocal cursor
            loci.append(Locus(name, cursor, size, lo, hi, channel))
            cursor += size

        # --- regulatory network of the zygotic GRN -------------------------
        add("W", K * K, -0.50, 0.50)      # regulatory coupling matrix
        add("lam", K, 0.50, 2.00)         # gene product decay rates
        add("b", K, -0.60, 0.60)          # constitutive gene biases
        add("s0", K, -0.80, 0.80)         # zygotic determinant (uniform input)
        add("morph", K, -1.00, 1.00)      # sensitivity to positional morphogen
        # --- global developmental parameters -------------------------------
        add("D_diff", 1, 0.00, 0.50)      # inter-module diffusion of gene products
        add("kappa_J", 1, 0.15, 1.00)     # coupling gain of the developed connectome
        add("fate_temp", 1, 0.50, 2.00)   # softmax temperature (fate thresholds)
        # --- maternally deposited (cytoplasmic) channel --------------------
        add("T_dev", 1, 12.0, 36.0, "maternal")        # developmental duration
        add("sigma_dev", 1, 0.010, 0.060, "maternal")  # developmental noise
        add("morph_steep", 1, 0.60, 2.50, "maternal")  # morphogen gradient steepness

        self.loci: Tuple[Locus, ...] = tuple(loci)
        self.d: int = cursor
        self._by_name: Dict[str, Locus] = {l.name: l for l in self.loci}

        lo = np.empty(self.d)
        hi = np.empty(self.d)
        maternal = np.zeros(self.d, dtype=bool)
        for l in self.loci:
            lo[l.slice] = l.lo
            hi[l.slice] = l.hi
            maternal[l.slice] = (l.channel == "maternal")
        self.lo, self.hi, self.maternal_mask = lo, hi, maternal
        self.autosomal_mask = ~maternal

    # ------------------------------------------------------------------ API
    def __len__(self) -> int:
        return self.d

    def locus(self, name: str) -> Locus:
        return self._by_name[name]

    def denormalise(self, u: np.ndarray) -> np.ndarray:
        """Map normalised coordinates ``u in [-1,1]`` to biological ranges."""
        u = np.clip(u, -1.0, 1.0)
        return self.lo + (u + 1.0) * 0.5 * (self.hi - self.lo)

    def unpack(self, u: np.ndarray) -> Dict[str, np.ndarray]:
        """Return denormalised parameter blocks.  ``u`` may be (d,) or (B, d)."""
        u = np.atleast_2d(u)
        p = self.denormalise(u)
        K = self.gcfg.n_genes
        out = {
            "W": p[:, self.locus("W").slice].reshape(-1, K, K),
            "lam": p[:, self.locus("lam").slice],
            "b": p[:, self.locus("b").slice],
            "s0": p[:, self.locus("s0").slice],
            "morph": p[:, self.locus("morph").slice],
            "D_diff": p[:, self.locus("D_diff").slice][:, 0],
            "kappa_J": p[:, self.locus("kappa_J").slice][:, 0],
            "fate_temp": p[:, self.locus("fate_temp").slice][:, 0],
            "T_dev": p[:, self.locus("T_dev").slice][:, 0],
            "sigma_dev": p[:, self.locus("sigma_dev").slice][:, 0],
            "morph_steep": p[:, self.locus("morph_steep").slice][:, 0],
        }
        return out

    def block_names(self) -> Tuple[str, ...]:
        return tuple(l.name for l in self.loci)


# --------------------------------------------------------------------------
# Founder construction
# --------------------------------------------------------------------------
def _sylvester_hadamard(n: int) -> np.ndarray:
    """Sylvester-construction Hadamard matrix of order ``n`` (a power of two)."""
    if n & (n - 1) != 0:
        raise ValueError("Hadamard order must be a power of two")
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def founder_germlines(spec: GermlineSpec, n_founders: int = 4) -> np.ndarray:
    """Deterministic, documented founder construction.

    Founder *k* is placed at ``amp * h_k`` where ``h_k`` is row ``k+1`` of a
    Sylvester Hadamard matrix truncated to ``d`` loci.  Row 0 (all-ones) is
    skipped.  Consequences:

    * every founder has *exactly* the same per-locus magnitude ``amp``
      (magnitude-matched by construction, so no founder is "stronger");
    * distinct Hadamard rows are orthogonal at full length and near-orthogonal
      after truncation, so founders are maximally spread in germline space;
    * the rule involves no free choice and no post-hoc adjustment.

    The realised pairwise germline distances are *measured* and reported rather
    than assumed (see :func:`germline_distance_matrix`).
    """
    order = 1
    while order < spec.d + n_founders + 1:
        order *= 2
    H = _sylvester_hadamard(order)
    rows = H[1:n_founders + 1, :spec.d]
    return spec.gcfg.founder_amplitude * rows.astype(float)


def random_germlines(spec: GermlineSpec, n: int, rng: np.random.Generator) -> np.ndarray:
    """Magnitude- and dimension-matched random germline control (Condition 4).

    Each locus is set to ``+-amp`` with equal probability.  The resulting
    vectors have *identical* per-locus magnitude and identical norm to the
    founders, so any difference from inherited germlines cannot be attributed
    to a difference in parameter scale.
    """
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n, spec.d))
    return spec.gcfg.founder_amplitude * signs


# --------------------------------------------------------------------------
# Distances
# --------------------------------------------------------------------------
def germline_distance(u1: np.ndarray, u2: np.ndarray) -> float:
    """Root-mean-square distance in normalised locus coordinates."""
    d = np.asarray(u1, float) - np.asarray(u2, float)
    return float(np.sqrt(np.mean(d ** 2)))


def germline_distance_matrix(U: np.ndarray) -> np.ndarray:
    U = np.atleast_2d(U)
    n = U.shape[0]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = germline_distance(U[i], U[j])
    return M


def make_spec(cfg: Config) -> GermlineSpec:
    return GermlineSpec(cfg.germline)
