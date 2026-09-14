"""
Phenotype operator ``Y_C = Phi(S_C^(0))``.

Six pre-learning traits are measured on the *nominal* twin over the window
``[t_transient, t_total)``; the perturbation trait additionally uses the
perturbed twin.  Nothing is learned, fitted or adapted at any point.

Every individual yields a phenotype.  Individuals whose behaviour is degenerate
(motionless, non-oscillatory, censored recovery) are **flagged, never
discarded** -- the flag rates are themselves reported outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.signal import hilbert

from .config import Config, TRAIT_NAMES
from .embodiment import Traces

#: A spectral peak must carry at least this fraction of total AC power for the
#: newborn to be called oscillatory.
OSCILLATION_POWER_FRACTION = 0.05
#: Mean speed below this fraction of ``v_max`` counts as motionless.
MOTION_FLOOR_FRACTION = 0.02


@dataclass
class Phenotypes:
    Y: np.ndarray                 # (B, 6)
    flags: Dict[str, np.ndarray]  # each (B,) bool
    aux: Dict[str, np.ndarray]

    @property
    def batch(self) -> int:
        return self.Y.shape[0]

    def as_dict(self) -> Dict[str, np.ndarray]:
        d = {name: self.Y[:, i] for i, name in enumerate(TRAIT_NAMES)}
        d.update({f"flag_{k}": v for k, v in self.flags.items()})
        d.update(self.aux)
        return d


def _orbit_distance(query: np.ndarray, cloud: np.ndarray,
                    block: int = 64) -> np.ndarray:
    """Minimum Euclidean distance from each query state to a reference orbit.

    ``query`` is (B, T, D), ``cloud`` is (B, C, D).  Evaluated in blocks over
    the batch to bound memory.
    """
    B, T, D = query.shape
    out = np.empty((B, T))
    for lo in range(0, B, block):
        hi = min(lo + block, B)
        q = query[lo:hi]                       # (b, T, D)
        c = cloud[lo:hi]                       # (b, C, D)
        d2 = (np.sum(q ** 2, axis=2)[:, :, None]
              - 2.0 * np.einsum("btd,bcd->btc", q, c)
              + np.sum(c ** 2, axis=2)[:, None, :])
        out[lo:hi] = np.sqrt(np.maximum(d2.min(axis=2), 0.0))
    return out


def _dominant_frequency(r: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Dominant frequency of the module activities and its power fraction.

    The power spectra of all modules are averaged, so the estimate does not
    depend on which module happens to dominate, and antiphase modules do not
    cancel.
    """
    B, T, N = r.shape
    x = r - r.mean(axis=1, keepdims=True)
    win = np.hanning(T)[None, :, None]
    nfft = int(2 ** np.ceil(np.log2(max(T, 8)))) * 4
    F = np.fft.rfft(x * win, n=nfft, axis=1)
    P = (np.abs(F) ** 2).mean(axis=2)               # (B, nfft//2+1)
    freqs = np.fft.rfftfreq(nfft, d=dt)
    fmin = 2.0 / (T * dt)                           # need >= 2 cycles in window
    valid = freqs >= fmin
    Pv = np.where(valid[None, :], P, 0.0)
    k = np.argmax(Pv, axis=1)
    total = Pv.sum(axis=1) + 1e-30

    # parabolic refinement in log-power
    kk = np.clip(k, 1, P.shape[1] - 2)
    y0 = np.log(Pv[np.arange(B), kk - 1] + 1e-30)
    y1 = np.log(Pv[np.arange(B), kk] + 1e-30)
    y2 = np.log(Pv[np.arange(B), kk + 1] + 1e-30)
    denom = (y0 - 2 * y1 + y2)
    safe = np.abs(denom) > 1e-12
    delta = np.where(safe, 0.5 * (y0 - y2) / np.where(safe, denom, 1.0), 0.0)
    delta = np.clip(delta, -0.5, 0.5)
    df = freqs[1] - freqs[0]
    f_peak = freqs[kk] + delta * df

    band = np.zeros(B)
    for j in range(-3, 4):
        idx = np.clip(kk + j, 0, P.shape[1] - 1)
        band += Pv[np.arange(B), idx]
    return f_peak, band / total


def _coherence(r: np.ndarray) -> np.ndarray:
    """Amplitude-weighted phase coherence across modules (Kuramoto order).

    Amplitude weighting prevents silent modules from injecting random phases.
    """
    x = r - r.mean(axis=1, keepdims=True)
    ana = hilbert(x, axis=1)
    amp = np.abs(ana)
    phase = np.angle(ana)
    num = np.abs((amp * np.exp(1j * phase)).sum(axis=2))
    den = amp.sum(axis=2) + 1e-30
    return (num / den).mean(axis=1)


def _recovery_time(nom_state: np.ndarray, per_state: np.ndarray, t: np.ndarray,
                   cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phase-invariant perturbation recovery time.

    A generic impulse to a limit-cycle system produces a permanent phase shift,
    so twin state-distance never decays and any recovery criterion based on it
    censors almost everybody (this was verified in the pre-registration pilot).
    The measured quantity is therefore the **transverse** deviation: the
    distance from the perturbed state to the nominal orbit, represented as the
    point cloud of nominal states over the pre-impulse steady window.

    Because the nominal twin is itself still moving through the (position
    dependent) sensory field, its own transverse distance to that cloud is not
    exactly zero; it is subtracted as a baseline, so the reported quantity is
    the *excess* deviation caused by the impulse.  Distances are normalised by
    the radius of gyration of the orbit, making the criterion scale-free.

    Recovery is the first time after the impulse at which the excess deviation
    falls below ``recovery_threshold`` times its post-impulse peak and remains
    below for ``recovery_hold``.  Individuals that never satisfy this are
    censored at ``recovery_censor`` and flagged; they are never dropped.
    """
    e = cfg.embodiment
    dt_s = float(t[1] - t[0])
    pre = (t >= e.t_transient) & (t < e.t_perturb)
    post = t >= e.t_perturb

    cloud_all = nom_state[:, pre, :]
    n_pre = cloud_all.shape[1]
    n_cloud = min(160, n_pre)
    sel = np.linspace(0, n_pre - 1, n_cloud).astype(int)
    cloud = cloud_all[:, sel, :]
    centroid = cloud.mean(axis=1, keepdims=True)
    radius = np.sqrt(np.mean(np.sum((cloud - centroid) ** 2, axis=2), axis=1)) + 1e-9

    d_per = _orbit_distance(per_state[:, post, :], cloud)
    d_nom = _orbit_distance(nom_state[:, post, :], cloud)
    excess = np.maximum(d_per - d_nom, 0.0) / radius[:, None]

    tp = t[post]
    hold = max(int(round(e.recovery_hold / dt_s)), 1)
    peak = excess.max(axis=1)
    ineffective = peak < 1e-6
    thr = e.recovery_threshold * peak
    below = excess <= thr[:, None]

    B, T = excess.shape
    rec = np.full(B, e.recovery_censor)
    censored = np.ones(B, bool)
    run = np.zeros(B, dtype=int)
    for i in range(T):
        run = np.where(below[:, i], run + 1, 0)
        hit = censored & (run >= hold)
        if hit.any():
            rec[hit] = tp[i] - (hold - 1) * dt_s - e.t_perturb
            censored[hit] = False
    rec = np.clip(rec, 0.0, e.recovery_censor)
    rec[ineffective] = 0.0
    censored[ineffective] = False
    return rec, censored, peak


def compute_phenotypes(nom: Traces, per: Traces, cfg: Config) -> Phenotypes:
    e = cfg.embodiment
    t = nom.t
    dt_s = float(t[1] - t[0])
    mask = t >= e.t_transient

    speed = nom.v[:, mask].mean(axis=1)
    turn = nom.omega[:, mask].mean(axis=1)
    acts = nom.activities()[:, mask, :]
    f_peak, pfrac = _dominant_frequency(acts, dt_s)
    coh = _coherence(acts)
    rec, censored, peak = _recovery_time(nom.state, per.state, t, cfg)
    radius = np.sqrt((nom.xy ** 2).sum(axis=2)).max(axis=1)

    Y = np.stack([speed, f_peak, turn, rec, coh, radius], axis=1)

    flags = {
        "motionless": speed < MOTION_FLOOR_FRACTION * e.v_max,
        "non_oscillatory": pfrac < OSCILLATION_POWER_FRACTION,
        "recovery_censored": censored,
        "nonfinite": ~np.all(np.isfinite(Y), axis=1),
    }
    aux = {
        "spectral_power_fraction": pfrac,
        "perturbation_peak": peak,
        "path_length": np.abs(nom.v[:, mask]).sum(axis=1) * dt_s,
    }
    return Phenotypes(Y, flags, aux)


def standardise(Y: np.ndarray, centre: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """z-score traits with externally supplied centre/scale (never re-fitted)."""
    return (np.asarray(Y, float) - centre[None, :]) / np.clip(scale, 1e-12, None)[None, :]
