"""
Statistical machinery.

Implemented directly on numpy/scipy because ``statsmodels`` is not assumed to
be present.  Everything is deterministic given a seeded generator.

Design notes
------------
* The diallel is *balanced by construction*, so Type I = Type II = Type III
  sums of squares and the classical formulas apply.  Balance is asserted, not
  assumed.
* p-values are always accompanied by an effect size and an interval; the
  reporting layer never presents a p-value alone.
* Permutation tests state their exchangeability assumption explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps


# --------------------------------------------------------------------------
# Basic effect sizes and intervals
# --------------------------------------------------------------------------
def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Bias-corrected standardised mean difference (a − b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float("nan")
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if sp2 <= 0:
        return float("nan")
    d = (a.mean() - b.mean()) / np.sqrt(sp2)
    J = 1.0 - 3.0 / (4.0 * (na + nb) - 9.0)
    return float(d * J)


def bootstrap_ci(x: np.ndarray, stat: Callable[[np.ndarray], float],
                 rng: np.random.Generator, n_boot: int = 5000,
                 alpha: float = 0.05) -> Tuple[float, float, float]:
    """Percentile bootstrap CI of ``stat`` over rows of ``x``."""
    x = np.asarray(x)
    n = x.shape[0]
    if n < 2:
        v = float(stat(x))
        return v, float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    vals = np.array([stat(x[i]) for i in idx])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(stat(x)), float("nan"), float("nan")
    return (float(stat(x)),
            float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def bootstrap_ci_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
                      n_boot: int = 5000, alpha: float = 0.05
                      ) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for the difference of means (a − b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ia = rng.integers(0, a.size, size=(n_boot, a.size))
    ib = rng.integers(0, b.size, size=(n_boot, b.size))
    vals = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return (float(a.mean() - b.mean()),
            float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def permutation_test_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
                          n_perm: int = 5000) -> float:
    """Two-sided permutation p-value for a difference of means."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    na = a.size
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(pooled)
        if abs(p[:na].mean() - p[na:].mean()) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


def paired_permutation_test(d: np.ndarray, rng: np.random.Generator,
                            n_perm: int = 5000) -> float:
    """Two-sided sign-flip permutation p-value for paired differences."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    obs = abs(d.mean())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, d.size))
    null = np.abs((signs * d[None, :]).mean(axis=1))
    return float(((null >= obs - 1e-15).sum() + 1) / (n_perm + 1))


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def two_proportion_permutation(k1: int, n1: int, k2: int, n2: int,
                               rng: np.random.Generator,
                               n_perm: int = 5000) -> Tuple[float, float]:
    """Permutation test for p1 > p2 (two-sided) plus the observed difference."""
    labels = np.concatenate([np.ones(k1), np.zeros(n1 - k1),
                             np.ones(k2), np.zeros(n2 - k2)])
    obs = k1 / n1 - k2 / n2
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(labels)
        if abs(p[:n1].mean() - p[n1:].mean()) >= abs(obs) - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1), float(obs)


# --------------------------------------------------------------------------
# Balanced two-way ANOVA
# --------------------------------------------------------------------------
@dataclass
class AnovaResult:
    trait: str
    n: int
    n_levels_a: int
    n_levels_b: int
    n_per_cell: int
    ss: Dict[str, float]
    df: Dict[str, float]
    ms: Dict[str, float]
    F: Dict[str, float]
    p_param: Dict[str, float]
    partial_eta2: Dict[str, float]
    omega2: Dict[str, float]
    grand_mean: float
    alpha_a: List[float]           # dam main effects (sum-to-zero)
    beta_b: List[float]            # sire main effects (sum-to-zero)
    gamma_ab: List[List[float]]    # interaction effects
    cell_means: List[List[float]]

    def to_dict(self) -> Dict:
        return asdict(self)


def two_way_anova(y: np.ndarray, fa: np.ndarray, fb: np.ndarray,
                  trait: str = "") -> AnovaResult:
    """Balanced two-factor fixed-effects ANOVA with interaction.

    ``fa``/``fb`` are integer level codes.  Raises if the design is unbalanced,
    because the classical partition would then be ambiguous.
    """
    y = np.asarray(y, float)
    fa = np.asarray(fa, int)
    fb = np.asarray(fb, int)
    la = np.unique(fa)
    lb = np.unique(fb)
    a, b = la.size, lb.size
    counts = np.array([[np.sum((fa == i) & (fb == j)) for j in lb] for i in la])
    if counts.min() != counts.max():
        raise ValueError(f"unbalanced design: cell counts {counts.tolist()}")
    n = int(counts[0, 0])
    N = y.size

    gm = y.mean()
    cell = np.array([[y[(fa == i) & (fb == j)].mean() for j in lb] for i in la])
    mean_a = np.array([y[fa == i].mean() for i in la])
    mean_b = np.array([y[fb == j].mean() for j in lb])

    ss_a = n * b * np.sum((mean_a - gm) ** 2)
    ss_b = n * a * np.sum((mean_b - gm) ** 2)
    ss_ab = n * np.sum((cell - mean_a[:, None] - mean_b[None, :] + gm) ** 2)
    fitted = np.array([cell[np.where(la == i)[0][0], np.where(lb == j)[0][0]]
                       for i, j in zip(fa, fb)])
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - gm) ** 2)

    df_a, df_b = a - 1, b - 1
    df_ab = df_a * df_b
    df_res = N - a * b
    ss = {"dam": ss_a, "sire": ss_b, "dam:sire": ss_ab,
          "residual": ss_res, "total": ss_tot}
    df = {"dam": df_a, "sire": df_b, "dam:sire": df_ab,
          "residual": df_res, "total": N - 1}
    ms = {k: (ss[k] / df[k] if df[k] > 0 else float("nan"))
          for k in ("dam", "sire", "dam:sire", "residual")}
    F, p, pe2, w2 = {}, {}, {}, {}
    for k in ("dam", "sire", "dam:sire"):
        F[k] = ms[k] / ms["residual"] if ms["residual"] > 0 else float("nan")
        p[k] = (float(sps.f.sf(F[k], df[k], df_res))
                if np.isfinite(F[k]) and df_res > 0 else float("nan"))
        pe2[k] = ss[k] / (ss[k] + ss_res) if (ss[k] + ss_res) > 0 else float("nan")
        w2[k] = ((ss[k] - df[k] * ms["residual"]) / (ss_tot + ms["residual"])
                 if ss_tot > 0 else float("nan"))

    alpha = mean_a - gm
    beta = mean_b - gm
    gamma = cell - mean_a[:, None] - mean_b[None, :] + gm
    return AnovaResult(trait, N, a, b, n, ss, df, ms, F, p, pe2, w2,
                       float(gm), alpha.tolist(), beta.tolist(),
                       gamma.tolist(), cell.tolist())


def _codes(fa: np.ndarray, fb: np.ndarray):
    la, lb = np.unique(fa), np.unique(fb)
    return np.searchsorted(la, fa), np.searchsorted(lb, fb), la.size, lb.size


def fast_anova_F(y: np.ndarray, ia: np.ndarray, ib: np.ndarray,
                 a: int, b: int) -> Tuple[float, float, float]:
    """Balanced two-way F statistics via bincount (hot path for permutations)."""
    N = y.size
    n = N // (a * b)
    cell = ia * b + ib
    s_cell = np.bincount(cell, y, minlength=a * b)
    s_a = np.bincount(ia, y, minlength=a)
    s_b = np.bincount(ib, y, minlength=b)
    tot = y.sum()
    gm = tot / N

    ss_a = (s_a ** 2).sum() / (n * b) - tot ** 2 / N
    ss_b = (s_b ** 2).sum() / (n * a) - tot ** 2 / N
    ss_cells = (s_cell ** 2).sum() / n - tot ** 2 / N
    ss_ab = ss_cells - ss_a - ss_b
    ss_res = (y ** 2).sum() - (s_cell ** 2).sum() / n

    df_a, df_b = a - 1, b - 1
    df_ab = df_a * df_b
    df_res = N - a * b
    if df_res <= 0 or ss_res <= 0:
        return float("nan"), float("nan"), float("nan")
    mse = ss_res / df_res
    return (ss_a / df_a / mse,
            ss_b / df_b / mse,
            ss_ab / df_ab / mse if df_ab > 0 else float("nan"))


def bootstrap_anova_effects(y: np.ndarray, fa: np.ndarray, fb: np.ndarray,
                            rng: np.random.Generator, n_boot: int = 5000,
                            alpha: float = 0.05) -> Dict[str, np.ndarray]:
    """Within-cell bootstrap CIs for the sum-to-zero ANOVA effect estimates."""
    y = np.asarray(y, float)
    ia, ib, a, b = _codes(np.asarray(fa, int), np.asarray(fb, int))
    cell = ia * b + ib
    groups = [np.where(cell == c)[0] for c in range(a * b)]
    n = groups[0].size

    alphas = np.empty((n_boot, a))
    betas = np.empty((n_boot, b))
    gammas = np.empty((n_boot, a, b))
    for k in range(n_boot):
        idx = np.concatenate([g[rng.integers(0, n, n)] for g in groups])
        yy = y[idx]
        cc = cell[idx]
        s_cell = np.bincount(cc, yy, minlength=a * b) / n
        cm = s_cell.reshape(a, b)
        gm = cm.mean()
        ma = cm.mean(axis=1)
        mb = cm.mean(axis=0)
        alphas[k] = ma - gm
        betas[k] = mb - gm
        gammas[k] = cm - ma[:, None] - mb[None, :] + gm

    q = (100 * alpha / 2, 100 * (1 - alpha / 2))
    return {
        "alpha_lo": np.percentile(alphas, q[0], axis=0),
        "alpha_hi": np.percentile(alphas, q[1], axis=0),
        "beta_lo": np.percentile(betas, q[0], axis=0),
        "beta_hi": np.percentile(betas, q[1], axis=0),
        "gamma_lo": np.percentile(gammas, q[0], axis=0),
        "gamma_hi": np.percentile(gammas, q[1], axis=0),
    }


def _f_statistic(y, fa, fb, which: str) -> float:
    try:
        r = two_way_anova(y, fa, fb)
    except ValueError:
        return float("nan")
    return r.F[which]


def permutation_anova(y: np.ndarray, fa: np.ndarray, fb: np.ndarray, which: str,
                      rng: np.random.Generator, n_perm: int = 5000) -> float:
    """Permutation p-value for one ANOVA term.

    Exchangeability used:

    ``dam``
        under H0 (no dam effect and no interaction) observations sharing a sire
        are exchangeable, so dam labels are permuted *within* each sire group;
    ``sire``
        symmetric;
    ``dam:sire``
        residuals of the additive model are permuted across cells and added to
        the additive fit (ter Braak); this is an approximate test and is
        labelled as such.
    """
    y = np.asarray(y, float)
    ia, ib, a, b = _codes(np.asarray(fa, int), np.asarray(fb, int))
    slot = {"dam": 0, "sire": 1, "dam:sire": 2}[which]
    obs = fast_anova_F(y, ia, ib, a, b)[slot]
    if not np.isfinite(obs):
        return float("nan")

    count = 0
    if which in ("dam", "sire"):
        keep = ib if which == "dam" else ia
        move = ia if which == "dam" else ib
        groups = [np.where(keep == g)[0] for g in np.unique(keep)]
        for _ in range(n_perm):
            perm = move.copy()
            for idx in groups:
                perm[idx] = move[rng.permutation(idx)]
            f = (fast_anova_F(y, perm, ib, a, b)[slot] if which == "dam"
                 else fast_anova_F(y, ia, perm, a, b)[slot])
            if np.isfinite(f) and f >= obs - 1e-12:
                count += 1
    else:
        gm = y.mean()
        ma = np.bincount(ia, y, minlength=a) / (y.size / a)
        mb = np.bincount(ib, y, minlength=b) / (y.size / b)
        add_fit = ma[ia] + mb[ib] - gm
        resid = y - add_fit
        for _ in range(n_perm):
            yp = add_fit + resid[rng.permutation(resid.size)]
            f = fast_anova_F(yp, ia, ib, a, b)[slot]
            if np.isfinite(f) and f >= obs - 1e-12:
                count += 1
    return (count + 1) / (n_perm + 1)


# --------------------------------------------------------------------------
# Parental developmental variance fraction (PDVF)
# --------------------------------------------------------------------------
@dataclass
class PDVF:
    trait: str
    pdvf_r2: float          # (SS_dam + SS_sire + SS_int) / SS_total
    pdvf_components: float  # variance-component estimate
    v_dam: float
    v_sire: float
    v_int: float
    v_residual: float
    ci_low: float = float("nan")
    ci_high: float = float("nan")

    def to_dict(self) -> Dict:
        return asdict(self)


def pdvf(y: np.ndarray, fa: np.ndarray, fb: np.ndarray, trait: str = "") -> PDVF:
    """Parental developmental variance fraction.

    **Not** narrow-sense heritability.  There is no additive genetic model and
    no random sample from a breeding population: the "parents" are four fixed
    founders.  Two estimators are returned:

    ``pdvf_r2``
        the descriptive share of offspring phenotypic sum of squares that the
        parental factors and their interaction account for.  Upward biased.
    ``pdvf_components``
        an unbiased-style estimate from expected mean squares, with negative
        components truncated at zero (which introduces a small upward bias of
        its own; both numbers are reported so the reader can see the spread).
    """
    r = two_way_anova(y, fa, fb, trait)
    n, a, b = r.n_per_cell, r.n_levels_a, r.n_levels_b
    mse = r.ms["residual"]
    v_dam = max((r.ms["dam"] - mse) / (n * b), 0.0)
    v_sire = max((r.ms["sire"] - mse) / (n * a), 0.0)
    v_int = max((r.ms["dam:sire"] - mse) / n, 0.0)
    v_res = max(mse, 0.0)
    tot = v_dam + v_sire + v_int + v_res
    comp = (v_dam + v_sire + v_int) / tot if tot > 0 else float("nan")
    r2 = ((r.ss["dam"] + r.ss["sire"] + r.ss["dam:sire"]) / r.ss["total"]
          if r.ss["total"] > 0 else float("nan"))
    return PDVF(trait, float(r2), float(comp), float(v_dam), float(v_sire),
                float(v_int), float(v_res))


def pdvf_bootstrap(y: np.ndarray, fa: np.ndarray, fb: np.ndarray,
                   rng: np.random.Generator, n_boot: int = 2000,
                   alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile CI for ``pdvf_components`` by resampling within cells."""
    y = np.asarray(y, float)
    fa, fb = np.asarray(fa, int), np.asarray(fb, int)
    cells = {}
    for i in np.unique(fa):
        for j in np.unique(fb):
            cells[(i, j)] = np.where((fa == i) & (fb == j))[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(v, size=v.size, replace=True)
                              for v in cells.values()])
        try:
            vals.append(pdvf(y[idx], fa[idx], fb[idx]).pdvf_components)
        except ValueError:
            continue
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    if vals.size == 0:
        return float("nan"), float("nan")
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


# --------------------------------------------------------------------------
# Regression (parent–offspring resemblance)
# --------------------------------------------------------------------------
@dataclass
class RegressionResult:
    slope: float
    intercept: float
    r2: float
    slope_ci: Tuple[float, float]
    p_perm: float
    n: int

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["slope_ci"] = list(self.slope_ci)
        return d


def linreg_with_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                   n_boot: int = 5000, n_perm: int = 5000) -> RegressionResult:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 3 or np.allclose(x, x[0]):
        return RegressionResult(float("nan"), float("nan"), float("nan"),
                                (float("nan"), float("nan")), float("nan"), n)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    idx = rng.integers(0, n, size=(n_boot, n))
    slopes = np.empty(n_boot)
    for i in range(n_boot):
        xi, yi = x[idx[i]], y[idx[i]]
        if np.allclose(xi, xi[0]):
            slopes[i] = np.nan
        else:
            slopes[i] = np.polyfit(xi, yi, 1)[0]
    slopes = slopes[np.isfinite(slopes)]
    ci = ((float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5)))
          if slopes.size else (float("nan"), float("nan")))

    obs = abs(slope)
    cnt = 0
    for _ in range(n_perm):
        yp = y[rng.permutation(n)]
        s = np.polyfit(x, yp, 1)[0]
        if abs(s) >= obs - 1e-15:
            cnt += 1
    return RegressionResult(float(slope), float(intercept), float(r2), ci,
                            (cnt + 1) / (n_perm + 1), n)


# --------------------------------------------------------------------------
# Multivariate comparison
# --------------------------------------------------------------------------
def centroid_distance_test(A: np.ndarray, B: np.ndarray, rng: np.random.Generator,
                           n_perm: int = 5000) -> Dict[str, float]:
    """Permutation test on the distance between two multivariate centroids.

    Inputs must already be standardised on a common scale.  Also returns a
    multivariate effect size (centroid distance in pooled-SD units).
    """
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    obs = float(np.linalg.norm(A.mean(0) - B.mean(0)))
    pooled = np.vstack([A, B])
    na = A.shape[0]
    cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(pooled.shape[0])
        s = pooled[p]
        if np.linalg.norm(s[:na].mean(0) - s[na:].mean(0)) >= obs - 1e-15:
            cnt += 1
    sd = np.sqrt(0.5 * (A.var(axis=0, ddof=1) + B.var(axis=0, ddof=1))).mean()
    return {"centroid_distance": obs,
            "p_perm": (cnt + 1) / (n_perm + 1),
            "effect_size": float(obs / sd) if sd > 0 else float("nan")}


def levene_like_test(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
                     n_perm: int = 5000) -> Dict[str, float]:
    """Permutation test for a difference in dispersion (Brown–Forsythe style)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    za = np.abs(a - np.median(a))
    zb = np.abs(b - np.median(b))
    p = permutation_test_diff(za, zb, rng, n_perm)
    return {"sd_a": float(a.std(ddof=1)), "sd_b": float(b.std(ddof=1)),
            "variance_ratio": float(a.var(ddof=1) / b.var(ddof=1))
            if b.var(ddof=1) > 0 else float("nan"),
            "p_perm": p}
