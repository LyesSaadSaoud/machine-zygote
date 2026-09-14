"""
Endogenous development  ``S_C^(0) = D(Z_C, U, E_0)``.

The generic undifferentiated soma ``U``
-------------------------------------
Every child receives the *same* soma: ``N`` modules whose regulatory state is
identically zero and whose only distinguishing feature is a positional
coordinate ``p_m in [-1, 1]`` (a body axis).  No module has a role, a time
constant or a connection at birth of development.

The developmental dynamics
--------------------------
Each module runs the *same* zygotic regulatory network -- there is one genome
per individual, not one per module.  Symmetry is broken endogenously by
positional information (a morphogen read through the germline-encoded
sensitivity vector) and by inter-module diffusion of gene products::

    dz_i^(m)/dt = -lam_i z_i^(m)
                  + tanh( sum_j W_ij z_j^(m) + b_i + s_i^(m) )
                  + D * (z_i^(m-1) + z_i^(m+1) - 2 z_i^(m))
                  + sigma_dev * dW_t

    s_i^(m) = s0_i + morph_i * tanh(morph_steep * p_m)

Integration runs until the individual's own (maternally inherited) duration
``T_dev``, after which the state is *frozen*.  The frozen state is the only
thing the soma readout may see.

Ablations (Condition 5)
-----------------------
``nodev_static``
    literal "no development": the frozen state is the zygotic determinant
    ``s0`` copied to every module.  No dynamics, no positional information.
``nodev_quasistatic``
    a stricter control: positional information *is* read out, but no recurrent
    regulatory dynamics, no diffusion, no noise and no developmental time,
    ``z^(m) = tanh(b + s^(m)) / lam``.  This isolates the contribution of the
    dynamical process from that of mere positional read-out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .config import Config
from .germline import GermlineSpec

DEV_MODES = ("full", "nodev_static", "nodev_quasistatic")


@dataclass
class DevelopmentResult:
    frozen: np.ndarray               # (B, N, K) frozen regulatory state
    t_dev: np.ndarray                # (B,) realised developmental duration
    traj: Optional[np.ndarray]       # (B, T', N, K) subsampled trajectory or None
    traj_t: Optional[np.ndarray]     # (T',) times
    saturation: np.ndarray           # (B,) mean |tanh-argument| saturation index
    module_spread: np.ndarray        # (B,) RMS between-module dispersion of frozen z
    mode: str


def module_positions(n_modules: int) -> np.ndarray:
    """Positional coordinate of the generic soma modules (the body axis)."""
    if n_modules == 1:
        return np.zeros(1)
    return np.linspace(-1.0, 1.0, n_modules)


def generic_soma_state(batch: int, n_modules: int, n_genes: int) -> np.ndarray:
    """The undifferentiated soma ``U``: identically zero regulatory state."""
    return np.zeros((batch, n_modules, n_genes))


def develop(zyg_u: np.ndarray, spec: GermlineSpec, cfg: Config,
            dev_seed: np.ndarray, mode: str = "full",
            store_traj: bool = False) -> DevelopmentResult:
    """Run development for a batch of zygotes.

    Parameters
    ----------
    zyg_u : (B, d) array
        Zygote genomes in normalised locus coordinates.
    dev_seed : (B,) int array
        Per-individual seed for the developmental noise field.  Supplying the
        same seeds for two batches makes the noise realisation identical, which
        the causal-swap intervention relies on.
    """
    if mode not in DEV_MODES:
        raise ValueError(f"unknown development mode {mode!r}")

    zyg_u = np.atleast_2d(zyg_u)
    B = zyg_u.shape[0]
    K = spec.gcfg.n_genes
    N = spec.gcfg.n_modules
    par = spec.unpack(zyg_u)

    W, lam, b = par["W"], par["lam"], par["b"]
    s0, morph = par["s0"], par["morph"]
    D_diff = par["D_diff"] * cfg.development.diffusion_scale
    sigma_dev = par["sigma_dev"] * cfg.development.noise_scale
    steep = par["morph_steep"]
    t_dev = par["T_dev"] * cfg.development.duration_scale

    p = module_positions(N)                                   # (N,)
    morphogen = np.tanh(steep[:, None] * p[None, :])          # (B, N)
    s = s0[:, None, :] + morph[:, None, :] * morphogen[:, :, None]   # (B, N, K)

    # ---------------------------------------------------------- ablations
    if mode == "nodev_static":
        frozen = np.repeat(s0[:, None, :], N, axis=1)
        return DevelopmentResult(frozen, np.zeros(B), None, None,
                                 np.zeros(B), _spread(frozen), mode)
    if mode == "nodev_quasistatic":
        frozen = np.tanh(b[:, None, :] + s) / lam[:, None, :]
        return DevelopmentResult(frozen, np.zeros(B), None, None,
                                 np.zeros(B), _spread(frozen), mode)

    # ---------------------------------------------------------- full model
    dt = cfg.development.dt
    stop_step = np.maximum(1, np.rint(t_dev / dt).astype(np.int64))
    n_steps = int(stop_step.max())

    z = generic_soma_state(B, N, K)              # U: generic, identical, zero
    frozen = np.zeros_like(z)
    done = np.zeros(B, dtype=bool)

    # Independent noise stream per individual, reproducible from dev_seed alone
    # (never from batch position), which is what the causal-swap design needs.
    noise_field = _BlockNoise(np.asarray(dev_seed, dtype=np.int64), (N, K))

    stride = cfg.development.trace_stride
    traj = [] if store_traj else None
    traj_t = [] if store_traj else None

    sat_acc = np.zeros(B)
    sqrt_dt = np.sqrt(dt)

    for step in range(n_steps):
        arg = np.einsum("bij,bnj->bni", W, z) + b[:, None, :] + s
        sat_acc += np.mean(np.abs(arg), axis=(1, 2))
        drift = -lam[:, None, :] * z + np.tanh(arg)
        if N > 1:
            lap = (np.roll(z, 1, axis=1) + np.roll(z, -1, axis=1) - 2.0 * z)
            # no-flux (Neumann) boundaries
            lap[:, 0, :] = z[:, 1, :] - z[:, 0, :]
            lap[:, -1, :] = z[:, -2, :] - z[:, -1, :]
            drift = drift + D_diff[:, None, None] * lap

        noise = noise_field.step(step)
        z = z + dt * drift + sigma_dev[:, None, None] * sqrt_dt * noise
        z = np.clip(z, -20.0, 20.0)   # numerical guard; never reached in practice

        reached = (~done) & (stop_step <= step + 1)
        if reached.any():
            frozen[reached] = z[reached]
            done |= reached

        if store_traj and (step % stride == 0):
            traj.append(z.copy())
            traj_t.append((step + 1) * dt)

    if not done.all():                       # safety: should be impossible
        frozen[~done] = z[~done]

    saturation = sat_acc / max(n_steps, 1)
    out_traj = np.stack(traj, axis=1) if store_traj else None
    out_t = np.asarray(traj_t) if store_traj else None
    return DevelopmentResult(frozen, t_dev, out_traj, out_t,
                             saturation, _spread(frozen), mode)


class _BlockNoise:
    """Per-individual Gaussian noise field, generated in blocks for speed.

    Individual ``i`` always sees the stream ``default_rng(dev_seed[i])``,
    independent of the batch it is simulated in and of the batch size, so two
    runs that share ``dev_seed`` share the exact same developmental noise.
    """

    def __init__(self, seeds: np.ndarray, shape, block: int = 256):
        self.streams = [np.random.default_rng(int(s)) for s in seeds]
        self.shape = tuple(shape)
        self.block = int(block)
        self._buf = None
        self._base = -1

    def step(self, step: int) -> np.ndarray:
        if self._buf is None or not (self._base <= step < self._base + self.block):
            self._base = (step // self.block) * self.block
            self._buf = np.stack(
                [st.standard_normal((self.block,) + self.shape) for st in self.streams],
                axis=0,
            )
        return self._buf[:, step - self._base]


def _spread(frozen: np.ndarray) -> np.ndarray:
    """RMS dispersion of the frozen state *between* modules.

    Zero means the modules never differentiated from one another.
    """
    return np.sqrt(np.mean((frozen - frozen.mean(axis=1, keepdims=True)) ** 2,
                           axis=(1, 2)))
