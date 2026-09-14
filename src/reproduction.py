"""
End-to-end reproduction pipeline.

    G_dam, G_sire  --R-->  Z_C  --D-->  S_C^(0)  --Phi-->  Y_C

Everything is vectorised over a batch of individuals and executed in chunks so
that memory stays bounded.  The chunk size never affects a result: the
developmental noise stream of an individual is keyed on its own ``dev_seed``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import Config, TRAIT_NAMES
from .development import DevelopmentResult, develop
from .embodiment import simulate_twins
from .germline import GermlineSpec
from .phenotype import Phenotypes, compute_phenotypes
from .soma import (Soma, differentiate, from_vector, role_diversity, to_vector,
                   vector_scales)
from .zygote import Background, draw_background, recombine


@dataclass
class Cohort:
    """A set of individuals measured under one condition."""

    zyg_u: np.ndarray                    # (B, d) zygote genomes
    Y: np.ndarray                        # (B, n_traits)
    flags: Dict[str, np.ndarray]
    aux: Dict[str, np.ndarray]
    frozen: np.ndarray                   # (B, N, K) frozen developmental state
    t_dev: np.ndarray
    module_spread: np.ndarray
    role_div: np.ndarray
    soma_vec: Optional[np.ndarray] = None
    meta: Dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return self.Y.shape[0]

    def subset(self, idx) -> "Cohort":
        idx = np.asarray(idx)
        return Cohort(
            self.zyg_u[idx], self.Y[idx],
            {k: v[idx] for k, v in self.flags.items()},
            {k: v[idx] for k, v in self.aux.items()},
            self.frozen[idx], self.t_dev[idx], self.module_spread[idx],
            self.role_div[idx],
            None if self.soma_vec is None else self.soma_vec[idx],
            {k: np.asarray(v)[idx] for k, v in self.meta.items()},
        )

    def to_records(self) -> Dict[str, np.ndarray]:
        rec: Dict[str, np.ndarray] = {}
        for i, name in enumerate(TRAIT_NAMES):
            rec[name] = self.Y[:, i]
        for k, v in self.flags.items():
            rec[f"flag_{k}"] = v
        for k, v in self.aux.items():
            rec[k] = v
        rec["t_dev"] = self.t_dev
        rec["module_spread"] = self.module_spread
        rec["role_diversity"] = self.role_div
        for k, v in self.meta.items():
            rec[k] = np.asarray(v)
        return rec


def _concat_cohorts(parts: Sequence[Cohort]) -> Cohort:
    def cat(attr):
        return np.concatenate([getattr(p, attr) for p in parts], axis=0)
    flags = {k: np.concatenate([p.flags[k] for p in parts]) for k in parts[0].flags}
    aux = {k: np.concatenate([p.aux[k] for p in parts]) for k in parts[0].aux}
    soma_vec = (None if parts[0].soma_vec is None
                else np.concatenate([p.soma_vec for p in parts], axis=0))
    return Cohort(cat("zyg_u"), cat("Y"), flags, aux, cat("frozen"),
                  cat("t_dev"), cat("module_spread"), cat("role_div"), soma_vec)


# --------------------------------------------------------------------------
# Core: zygote -> phenotype
# --------------------------------------------------------------------------
def evaluate_zygotes(zyg_u: np.ndarray, dev_seed: np.ndarray, spec: GermlineSpec,
                     cfg: Config, mode: str = "full",
                     keep_soma: bool = False) -> Cohort:
    zyg_u = np.atleast_2d(zyg_u)
    dev_seed = np.asarray(dev_seed, dtype=np.int64)
    n = zyg_u.shape[0]
    cs = max(int(cfg.run.chunk_size), 1)
    parts: List[Cohort] = []
    for lo in range(0, n, cs):
        hi = min(lo + cs, n)
        parts.append(_evaluate_chunk(zyg_u[lo:hi], dev_seed[lo:hi], spec, cfg,
                                     mode, keep_soma))
    return _concat_cohorts(parts) if len(parts) > 1 else parts[0]


def _evaluate_chunk(zyg_u, dev_seed, spec, cfg, mode, keep_soma) -> Cohort:
    par = spec.unpack(zyg_u)
    dev = develop(zyg_u, spec, cfg, dev_seed, mode=mode)
    soma = differentiate(dev.frozen, par["fate_temp"], par["kappa_J"], cfg)
    nom, per = simulate_twins(soma, cfg)
    ph = compute_phenotypes(nom, per, cfg)
    return Cohort(zyg_u, ph.Y, ph.flags, ph.aux, dev.frozen, dev.t_dev,
                  dev.module_spread, role_diversity(soma),
                  to_vector(soma) if keep_soma else None)


def evaluate_somas(soma: Soma, cfg: Config, zyg_u: np.ndarray) -> Cohort:
    """Evaluate controllers that were *not* produced by development
    (direct-controller baseline).  ``zyg_u`` is carried through only as a
    placeholder for bookkeeping."""
    nom, per = simulate_twins(soma, cfg)
    ph = compute_phenotypes(nom, per, cfg)
    B = soma.batch
    return Cohort(zyg_u, ph.Y, ph.flags, ph.aux,
                  np.zeros((B, soma.n_modules, 1)), np.zeros(B),
                  np.zeros(B), role_diversity(soma), to_vector(soma))


# --------------------------------------------------------------------------
# Crosses
# --------------------------------------------------------------------------
def cross(dam_g: np.ndarray, sire_g: np.ndarray, bg: Background,
          spec: GermlineSpec, cfg: Config, mode: str = "full",
          keep_soma: bool = False) -> Cohort:
    """Full reproduction: recombination, development, evaluation."""
    zyg = recombine(dam_g, sire_g, bg, spec)
    return evaluate_zygotes(zyg, bg.dev_seed, spec, cfg, mode, keep_soma)


def parental_reference(g: np.ndarray, n: int, spec: GermlineSpec, cfg: Config,
                       rng: np.random.Generator, keep_soma: bool = False) -> Cohort:
    """Reference individuals of a founder line.

    The zygote *is* the founder germline (no recombination, no mutation), so
    the only source of variation is developmental noise.  This measures the
    within-line developmental variation against which offspring novelty must
    be judged.
    """
    zyg = np.repeat(np.atleast_2d(g), n, axis=0)
    seeds = rng.integers(0, 2 ** 62, size=n, dtype=np.int64)
    return evaluate_zygotes(zyg, seeds, spec, cfg, "full", keep_soma)


def diallel(founders: np.ndarray, spec: GermlineSpec, cfg: Config,
            rng: np.random.Generator, n_rep: int,
            mode: str = "full") -> Cohort:
    """Complete ``n_founders x n_founders`` diallel including reciprocals.

    Cell ``(i, j)`` uses founder ``i`` as the dam and founder ``j`` as the
    sire, so ``(A,B)`` and ``(B,A)`` are distinct cells and the design is
    balanced with ``n_rep`` replicates each.
    """
    n_f = founders.shape[0]
    dam_idx, sire_idx, rep_idx = [], [], []
    for i in range(n_f):
        for j in range(n_f):
            dam_idx.append(np.full(n_rep, i))
            sire_idx.append(np.full(n_rep, j))
            rep_idx.append(np.arange(n_rep))
    dam_idx = np.concatenate(dam_idx)
    sire_idx = np.concatenate(sire_idx)
    rep_idx = np.concatenate(rep_idx)

    bg = draw_background(len(dam_idx), spec, rng)
    coh = cross(founders[dam_idx], founders[sire_idx], bg, spec, cfg, mode)
    coh.meta = {"dam": dam_idx, "sire": sire_idx, "rep": rep_idx}
    return coh


# --------------------------------------------------------------------------
# Condition 6: direct-controller baseline
# --------------------------------------------------------------------------
def direct_controller_cross(pool_a: np.ndarray, pool_b: np.ndarray,
                            n: int, spec: GermlineSpec, cfg: Config,
                            rng: np.random.Generator) -> Cohort:
    """Cross *developed controllers* directly, bypassing germline and development.

    ``pool_a`` / ``pool_b`` are controller parameter vectors of reference
    individuals from the two parental lines.  For each offspring one parent is
    drawn from each pool, parameters are combined by uniform crossover with the
    same per-locus probability used for germline recombination, and Gaussian
    mutation is applied with the same *relative* amplitude
    (``mutation_sigma`` times the pooled per-coordinate standard deviation).
    This is the "ordinary controller inheritance" that developmental heredity
    must be distinguished from.
    """
    pool = np.concatenate([pool_a, pool_b], axis=0)
    scales = vector_scales(pool)
    ia = rng.integers(0, pool_a.shape[0], size=n)
    ib = rng.integers(0, pool_b.shape[0], size=n)
    va, vb = pool_a[ia], pool_b[ib]
    mask = rng.random(va.shape) < spec.gcfg.recombination_p
    child = np.where(mask, va, vb)
    child = child + rng.normal(0.0, 1.0, size=child.shape) * (
        spec.gcfg.mutation_sigma * scales[None, :])
    soma = from_vector(child, spec.gcfg.n_modules)
    coh = evaluate_somas(soma, cfg, np.zeros((n, spec.d)))
    return coh
