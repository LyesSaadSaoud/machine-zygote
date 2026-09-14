"""
Newborn embodiment and the standardised evaluation environment ``E_eval``.

Each developed soma drives a differential-drive body::

    tau_u du_m/dt = -u_m + w_m r_m + sum_{n!=m} J_mn r_n + bias_m + I_m - a_m
    tau_a,m da_m/dt = g_m r_m - a_m
    r_m = tanh(u_m)

    I_m = sens_gain_m * L(x, y),   L = exp(-|| (x,y) - light ||^2 / (2 sigma_L^2))

    wheel_L = v_max * clip(slope * sum_m motL_m r_m, 0, 1)
    wheel_R = v_max * clip(slope * sum_m motR_m r_m, 0, 1)
    v = (wheel_L + wheel_R)/2,     omega = (wheel_R - wheel_L)/wheel_base
    dx = v cos(theta),  dy = v sin(theta),  dtheta = omega

*No parameter is updated during evaluation.*  There is no plasticity, no
reward, no adaptation of weights: the newborn is measured exactly as it was
born.  The initial condition is identical for every individual in every
condition, so all behavioural differences are attributable to the developed
soma.

Perturbation protocol
---------------------
Every individual is simulated twice, as an identical *twin pair*: one nominal,
one receiving a fixed impulse ``perturb_magnitude * cos(pi (m+1/2)/N)`` added
to ``u`` at ``t_perturb``.  Because the newborn is a limit-cycle system, a
generic impulse produces a permanent *phase* shift, so plain state distance
between the twins never returns to zero.  Recovery is therefore measured as
the *transverse* distance from the perturbed state to the nominal orbit (see
:func:`phenotype._recovery_time`), which is phase-invariant by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .config import Config
from .soma import Soma


@dataclass
class Traces:
    t: np.ndarray          # (T,) sample times
    state: np.ndarray      # (B, T, 2N) full neural state [u | a]
    v: np.ndarray          # (B, T) forward speed
    omega: np.ndarray      # (B, T) angular velocity
    xy: np.ndarray         # (B, T, 2) position
    perturbed: np.ndarray  # (B,) bool

    @property
    def n_modules(self) -> int:
        return self.state.shape[2] // 2

    def activities(self) -> np.ndarray:
        """Module outputs ``r = tanh(u)``."""
        return np.tanh(self.state[:, :, :self.n_modules])

    def energy(self) -> np.ndarray:
        """``mean_m (u^2 + a^2)``."""
        return 2.0 * np.mean(self.state ** 2, axis=2)


def initial_state(B: int, N: int, cfg: Config):
    """Standardised, identical newborn initial condition."""
    pat = np.cos(np.pi * (np.arange(N) + 0.5) / N)
    u = np.broadcast_to(cfg.embodiment.u0_amplitude * pat, (B, N)).copy()
    a = np.zeros((B, N))
    pose = np.zeros((B, 3))
    return u, a, pose


def perturbation_pattern(N: int) -> np.ndarray:
    return np.cos(np.pi * (np.arange(N) + 0.5) / N)


def tile_soma(s: Soma, reps: int) -> Soma:
    return Soma(*[np.concatenate([getattr(s, f.name)] * reps, axis=0)
                  for f in s.__dataclass_fields__.values()])


def _light(xy: np.ndarray, cfg: Config) -> np.ndarray:
    e = cfg.embodiment
    d2 = (xy[:, 0] - e.light_x) ** 2 + (xy[:, 1] - e.light_y) ** 2
    return np.exp(-d2 / (2.0 * e.light_sigma ** 2))


def simulate(soma: Soma, cfg: Config, perturb: np.ndarray) -> Traces:
    """Integrate a batch of newborns with Heun's method.

    ``perturb`` is a boolean array marking which members of the batch receive
    the impulse at ``t_perturb``.
    """
    e = cfg.embodiment
    B, N = soma.batch, soma.n_modules
    dt = e.dt
    n_steps = int(round(e.t_total / dt))
    stride = e.trace_stride
    n_out = n_steps // stride + 1

    u, a, pose = initial_state(B, N, cfg)
    perturb = np.asarray(perturb, bool)
    pat = perturbation_pattern(N)
    perturb_step = int(round(e.t_perturb / dt))

    t_out = np.empty(n_out)
    st_out = np.empty((B, n_out, 2 * N))
    v_out = np.empty((B, n_out))
    w_out = np.empty((B, n_out))
    xy_out = np.empty((B, n_out, 2))

    w_self, g, tau_a = soma.w, soma.g, soma.tau_a
    bias, J = soma.bias, soma.J
    sens, motL, motR = soma.sens_gain, soma.mot_L, soma.mot_R
    inv_tau_u = 1.0 / e.tau_u
    inv_tau_a = 1.0 / tau_a

    def deriv(u_, a_, pose_):
        r = np.tanh(u_)
        drive = (w_self * r
                 + np.einsum("bmn,bn->bm", J, r)
                 + bias
                 + sens * _light(pose_[:, :2], cfg)[:, None])
        du = (-u_ + drive - a_) * inv_tau_u
        da = (g * r - a_) * inv_tau_a
        dl = np.sum(motL * r, axis=1)
        dr = np.sum(motR * r, axis=1)
        # Rectified motor pools: wheels can only be driven forward, as a
        # muscle can only pull.  A silent motor pool therefore produces a
        # motionless newborn rather than a fixed half-speed cruise.
        wl = e.v_max * np.clip(e.motor_slope * dl, 0.0, 1.0)
        wr = e.v_max * np.clip(e.motor_slope * dr, 0.0, 1.0)
        v = 0.5 * (wl + wr)
        om = (wr - wl) / e.wheel_base
        th = pose_[:, 2]
        dpose = np.stack([v * np.cos(th), v * np.sin(th), om], axis=1)
        return du, da, dpose, r, v, om

    k = 0
    for step in range(n_steps + 1):
        du1, da1, dp1, r, v, om = deriv(u, a, pose)

        if step % stride == 0 and k < n_out:
            t_out[k] = step * dt
            st_out[:, k, :N] = u
            st_out[:, k, N:] = a
            v_out[:, k] = v
            w_out[:, k] = om
            xy_out[:, k] = pose[:, :2]
            k += 1

        if step == n_steps:
            break

        if step == perturb_step and perturb.any():
            u = u + perturb.astype(float)[:, None] * (e.perturb_magnitude * pat)[None, :]
            du1, da1, dp1, r, v, om = deriv(u, a, pose)

        u2 = u + dt * du1
        a2 = a + dt * da1
        p2 = pose + dt * dp1
        du2, da2, dp2, _, _, _ = deriv(u2, a2, p2)
        u = u + 0.5 * dt * (du1 + du2)
        a = a + 0.5 * dt * (da1 + da2)
        pose = pose + 0.5 * dt * (dp1 + dp2)

        if not np.all(np.isfinite(u)):
            u = np.nan_to_num(u, nan=0.0, posinf=1e3, neginf=-1e3)
            a = np.nan_to_num(a, nan=0.0, posinf=1e3, neginf=-1e3)

    return Traces(t_out[:k], st_out[:, :k], v_out[:, :k], w_out[:, :k],
                  xy_out[:, :k], perturb)


def simulate_twins(soma: Soma, cfg: Config):
    """Simulate the nominal and perturbed twin of every individual.

    Returns ``(nominal_traces, perturbed_traces)`` with matched batch order.
    """
    B = soma.batch
    doubled = tile_soma(soma, 2)
    perturb = np.concatenate([np.zeros(B, bool), np.ones(B, bool)])
    tr = simulate(doubled, cfg, perturb)
    nom = Traces(tr.t, tr.state[:B], tr.v[:B], tr.omega[:B], tr.xy[:B],
                 tr.perturbed[:B])
    per = Traces(tr.t, tr.state[B:], tr.v[B:], tr.omega[B:], tr.xy[B:],
                 tr.perturbed[B:])
    return nom, per
