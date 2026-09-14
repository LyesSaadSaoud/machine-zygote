"""
Phenotype-space metrics: standardisation, distances, transgressive segregation
and lineage measures.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import TRAIT_NAMES


# --------------------------------------------------------------------------
# Standardisation
# --------------------------------------------------------------------------
@dataclass
class Scaler:
    """Fixed centre/scale for trait standardisation.

    The scaler is fitted **once**, on a declared reference set (the parental
    reference individuals), and then applied unchanged everywhere.  Refitting
    per condition would make conditions incomparable.
    """
    centre: np.ndarray
    scale: np.ndarray

    @staticmethod
    def fit(Y: np.ndarray) -> "Scaler":
        Y = np.atleast_2d(Y)
        return Scaler(Y.mean(axis=0), np.clip(Y.std(axis=0, ddof=1), 1e-12, None))

    def transform(self, Y: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(Y) - self.centre[None, :]) / self.scale[None, :]

    def to_dict(self) -> Dict[str, List[float]]:
        return {"centre": self.centre.tolist(), "scale": self.scale.tolist()}


def pooled_sd(groups: Sequence[np.ndarray]) -> float:
    """Pooled within-group standard deviation."""
    num, den = 0.0, 0
    for g in groups:
        g = np.asarray(g, float)
        if g.size > 1:
            num += (g.size - 1) * g.var(ddof=1)
            den += g.size - 1
    return float(np.sqrt(num / den)) if den > 0 else float("nan")


# --------------------------------------------------------------------------
# Distances
# --------------------------------------------------------------------------
def std_distance(Y: np.ndarray, ref: np.ndarray, scaler: Scaler) -> np.ndarray:
    """Euclidean distance in standardised trait space to a single reference."""
    Z = scaler.transform(Y)
    z0 = scaler.transform(np.atleast_2d(ref))[0]
    return np.sqrt(((Z - z0[None, :]) ** 2).sum(axis=1))


def pca(Z: np.ndarray, n_components: int = 2):
    """PCA via SVD on already-standardised data.  Returns (scores, axes, var)."""
    Z = np.atleast_2d(Z)
    mu = Z.mean(axis=0, keepdims=True)
    X = Z - mu
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = (S ** 2) / max(X.shape[0] - 1, 1)
    total = var.sum()
    return (U[:, :n_components] * S[:n_components], Vt[:n_components],
            var[:n_components] / total if total > 0 else var[:n_components], mu)


# --------------------------------------------------------------------------
# H4: transgressive segregation
# --------------------------------------------------------------------------
@dataclass
class TransgressionResult:
    trait: str
    lo: float
    hi: float
    within_sd: float
    hybrid_rate: float
    clone_rate: float
    n_hybrid: int
    n_clone: int
    k_hybrid: int
    k_clone: int

    def to_dict(self) -> Dict:
        return asdict(self)


def transgression_interval(clone_a: np.ndarray, clone_b: np.ndarray,
                           k_sd: float = 2.0) -> Tuple[float, float, float]:
    """Parental interval widened by ``k_sd`` pooled within-line SDs."""
    mu_a, mu_b = float(np.mean(clone_a)), float(np.mean(clone_b))
    s = pooled_sd([clone_a, clone_b])
    return (min(mu_a, mu_b) - k_sd * s, max(mu_a, mu_b) + k_sd * s, s)


def transgression(hybrid: np.ndarray, clone_a: np.ndarray, clone_b: np.ndarray,
                  trait: str = "", k_sd: float = 2.0) -> TransgressionResult:
    """Rate at which hybrids fall outside the widened parental interval.

    The *same* criterion is applied to the clone-line individuals, which
    provides the null rate: clones carry identical developmental noise and
    identical mutation load, so any excess in the hybrids cannot be attributed
    to noise.
    """
    lo, hi, s = transgression_interval(clone_a, clone_b, k_sd)
    hybrid = np.asarray(hybrid, float)
    clones = np.concatenate([np.asarray(clone_a, float), np.asarray(clone_b, float)])
    kh = int(np.sum((hybrid < lo) | (hybrid > hi)))
    kc = int(np.sum((clones < lo) | (clones > hi)))
    return TransgressionResult(
        trait, lo, hi, s,
        kh / hybrid.size if hybrid.size else float("nan"),
        kc / clones.size if clones.size else float("nan"),
        hybrid.size, clones.size, kh, kc)


# --------------------------------------------------------------------------
# Lineage measures
# --------------------------------------------------------------------------
def germline_drift(u_child: np.ndarray, u_founders: np.ndarray) -> np.ndarray:
    """RMS distance of each germline from the nearest founder germline."""
    u_child = np.atleast_2d(u_child)
    d = np.sqrt(((u_child[:, None, :] - u_founders[None, :, :]) ** 2).mean(axis=2))
    return d.min(axis=1)


def founder_ancestry(u_child: np.ndarray, u_founders: np.ndarray) -> np.ndarray:
    """Soft ancestry weights: inverse-distance share of each founder.

    A descriptive summary only; it is not a formal admixture estimator.
    """
    u_child = np.atleast_2d(u_child)
    d = np.sqrt(((u_child[:, None, :] - u_founders[None, :, :]) ** 2).mean(axis=2))
    w = 1.0 / np.clip(d, 1e-9, None)
    return w / w.sum(axis=1, keepdims=True)


def lineage_distance_matrix(U: np.ndarray) -> np.ndarray:
    U = np.atleast_2d(U)
    diff = U[:, None, :] - U[None, :, :]
    return np.sqrt((diff ** 2).mean(axis=2))


def trait_persistence(parent_Y: np.ndarray, child_Y: np.ndarray,
                      scaler: Scaler) -> np.ndarray:
    """Per-trait standardised parent–offspring difference."""
    return scaler.transform(child_Y) - scaler.transform(parent_Y)


def flag_summary(flags: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {k: float(np.mean(v)) for k, v in flags.items()}
