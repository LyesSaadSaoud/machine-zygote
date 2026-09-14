"""
Soma differentiation: frozen regulatory state -> operational controller.

The read-out map ``rho`` implemented here is **fixed, universal and
non-heritable**.  It is identical for every individual in every condition and
contains no free parameters that any germline can address.  This is what
prevents the germline from being a disguised controller: a germline can only
influence the newborn controller by changing the *frozen developmental state*,
never by changing how that state is interpreted.

Read-out (per module ``m``, frozen state ``z^(m) in R^K``, ``K = 8``):

    fate weights   pi_m       = softmax(FATE_GAIN*(z^(m)[0:5] - mean_m z[0:5])
                                          / fate_temp)
    self-excite    w_m        = w_lo  + (w_hi-w_lo)  * sigmoid(z^(m)[5])
    adapt gain     g_m        = g_lo  + (g_hi-g_lo)  * sigmoid(z^(m)[6])
    adapt time     tau_a,m    = ta_lo + (ta_hi-ta_lo)* sigmoid(z^(m)[7])
                                * (1 + memory_slow * pi_m[memory])
    connectome     J_mn       = kappa_J * <z^(m), z^(n)> / K
                                * (0.5 + 1.5 * pi_n[inter]),  m != n,  J_mm = 0

The connectome rule is a chemoaffinity analogue: modules whose expression
profiles align couple positively, modules with opposed profiles couple
negatively.  Connectivity is therefore a *developmental product*, not an
inherited matrix.

The isolated-unit parameter ranges are chosen so that a unit is generically on
the oscillatory side of its Hopf bifurcation (``g > w - 1`` and
``tau_a > tau_u/(w-1)`` hold over the whole box).  This is a modelling choice
made a priori to avoid a floor effect of universally motionless newborns; it
is *not* tuned to any hypothesis, and the robustness sweep varies the coupling
that shapes behaviour on top of it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .config import Config, N_ROLES, ROLE_NAMES

# Fixed read-out constants (non-heritable).
W_LO, W_HI = 1.30, 2.00        # self-excitation
G_LO, G_HI = 1.00, 3.00        # adaptation gain
TA_LO, TA_HI = 1.50, 4.00      # adaptation time constant
MEMORY_SLOW = 1.5              # memory fate slows adaptation
FATE_GAIN = 5.0                # fixed gain on the (mean-subtracted) fate logits
INTER_COUPLING = (0.5, 1.5)    # interneuron fate scales outgoing coupling
BIAS_SCALE = 0.20              # tonic drive contributed by the memory fate

ROLE_IDX: Dict[str, int] = {r: i for i, r in enumerate(ROLE_NAMES)}


@dataclass
class Soma:
    """Developed newborn controller ``S_C^(0)`` (frozen; no learning)."""

    pi: np.ndarray        # (B, N, n_roles) graded fate weights
    w: np.ndarray         # (B, N) self-excitation
    g: np.ndarray         # (B, N) adaptation gain
    tau_a: np.ndarray     # (B, N) adaptation time constant
    bias: np.ndarray      # (B, N) tonic drive
    J: np.ndarray         # (B, N, N) developed connectome
    sens_gain: np.ndarray  # (B, N) sensor coupling
    mot_L: np.ndarray     # (B, N) left-motor read-out weights (sum to 1)
    mot_R: np.ndarray     # (B, N) right-motor read-out weights (sum to 1)

    @property
    def batch(self) -> int:
        return self.pi.shape[0]

    @property
    def n_modules(self) -> int:
        return self.pi.shape[1]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _softmax(x: np.ndarray, temp: np.ndarray) -> np.ndarray:
    z = x / np.clip(temp, 1e-6, None)[:, None, None]
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def differentiate(frozen: np.ndarray, fate_temp: np.ndarray,
                  kappa_J: np.ndarray, cfg: Config) -> Soma:
    """Apply the fixed read-out to a batch of frozen developmental states."""
    frozen = np.asarray(frozen, float)
    B, N, K = frozen.shape
    if K < N_ROLES + 3:
        raise ValueError("n_genes must be at least n_roles + 3")

    # Lateral-inhibition style read-out: a module's fate is decided by how its
    # regulatory state differs from the body-wide mean, not by its absolute
    # level.  If development leaves all modules identical the logits vanish and
    # every module stays uncommitted (uniform pi) -- the correct behaviour for
    # the no-development ablation.
    fate_logits = frozen[:, :, :N_ROLES]
    fate_logits = FATE_GAIN * (fate_logits - fate_logits.mean(axis=1, keepdims=True))
    pi = _softmax(fate_logits, np.asarray(fate_temp, float))
    w = W_LO + (W_HI - W_LO) * _sigmoid(frozen[:, :, N_ROLES + 0])
    g = G_LO + (G_HI - G_LO) * _sigmoid(frozen[:, :, N_ROLES + 1])
    tau_a = TA_LO + (TA_HI - TA_LO) * _sigmoid(frozen[:, :, N_ROLES + 2])
    tau_a = tau_a * (1.0 + MEMORY_SLOW * pi[:, :, ROLE_IDX["memory"]])
    bias = BIAS_SCALE * (pi[:, :, ROLE_IDX["memory"]] - 1.0 / N_ROLES)

    # Developed connectome (chemoaffinity analogue).
    gram = np.einsum("bmk,bnk->bmn", frozen, frozen) / float(K)
    inter = pi[:, :, ROLE_IDX["inter"]]
    lo, hi = INTER_COUPLING
    scale = lo + hi * inter                              # (B, N) per presynaptic module
    fan_in = max(N - 1, 1)                               # mean-field normalisation
    J = (np.asarray(kappa_J, float)[:, None, None] * gram * scale[:, None, :]
         / fan_in)
    eye = np.eye(N, dtype=bool)
    J[:, eye] = 0.0

    sens_gain = cfg.embodiment.sensor_gain * pi[:, :, ROLE_IDX["sensor"]]

    mot_L = _normalise(pi[:, :, ROLE_IDX["motor_L"]])
    mot_R = _normalise(pi[:, :, ROLE_IDX["motor_R"]])
    return Soma(pi, w, g, tau_a, bias, J, sens_gain, mot_L, mot_R)


def _normalise(x: np.ndarray) -> np.ndarray:
    s = x.sum(axis=1, keepdims=True)
    return x / np.clip(s, 1e-12, None)


# --------------------------------------------------------------------------
# Direct-controller baseline support (Condition 6)
# --------------------------------------------------------------------------
def to_vector(soma: Soma) -> np.ndarray:
    """Flatten a developed controller into a parameter vector.

    Layout: [pi (N*n_roles) | w (N) | g (N) | tau_a (N) | bias (N) |
             sens_gain (N) | J (N*N)].  ``mot_L`` / ``mot_R`` are recomputed
    from ``pi`` so that the crossed controller stays a valid read-out.
    """
    B, N, R = soma.pi.shape
    return np.concatenate([
        soma.pi.reshape(B, N * R),
        soma.w, soma.g, soma.tau_a, soma.bias, soma.sens_gain,
        soma.J.reshape(B, N * N),
    ], axis=1)


def from_vector(vec: np.ndarray, n_modules: int) -> Soma:
    """Inverse of :func:`to_vector`, with fate weights re-normalised."""
    vec = np.atleast_2d(vec)
    B = vec.shape[0]
    N, R = n_modules, N_ROLES
    o = 0
    pi = vec[:, o:o + N * R].reshape(B, N, R); o += N * R
    pi = np.clip(pi, 1e-9, None)
    pi = pi / pi.sum(axis=-1, keepdims=True)
    w = vec[:, o:o + N]; o += N
    g = vec[:, o:o + N]; o += N
    tau_a = vec[:, o:o + N]; o += N
    bias = vec[:, o:o + N]; o += N
    sens = vec[:, o:o + N]; o += N
    J = vec[:, o:o + N * N].reshape(B, N, N); o += N * N
    eye = np.eye(N, dtype=bool)
    J = J.copy()
    J[:, eye] = 0.0
    return Soma(pi, w, g, tau_a, bias, J, sens,
                _normalise(pi[:, :, ROLE_IDX["motor_L"]]),
                _normalise(pi[:, :, ROLE_IDX["motor_R"]]))


def vector_scales(vec: np.ndarray) -> np.ndarray:
    """Per-coordinate spread used to scale mutation in the direct-controller
    baseline, so that its mutation load is comparable to the germline one."""
    return np.std(np.atleast_2d(vec), axis=0) + 1e-12


def role_argmax(soma: Soma) -> np.ndarray:
    """Dominant fate per module (reporting / visualisation only)."""
    return np.argmax(soma.pi, axis=-1)


def differentiation_entropy(soma: Soma) -> np.ndarray:
    """Mean Shannon entropy of the fate distributions (0 = fully committed)."""
    p = np.clip(soma.pi, 1e-12, None)
    return float_mean(-(p * np.log(p)).sum(axis=-1))


def float_mean(x: np.ndarray) -> np.ndarray:
    return np.mean(x, axis=1)


def role_diversity(soma: Soma) -> np.ndarray:
    """Number of distinct dominant fates realised in the body (1..n_roles)."""
    am = role_argmax(soma)
    return np.array([len(np.unique(row)) for row in am], dtype=float)
