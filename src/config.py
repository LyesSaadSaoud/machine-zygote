"""
Configuration, deterministic seed derivation and run provenance.

Design rule
-----------
Every numerical parameter of the Machine Zygote model is declared here.  A run
is therefore *completely* described by ``(MASTER_SEED, Config)``.  Experiment
scripts are forbidden from hard-coding model constants; they may only override
fields of :class:`Config`.

All parameter values in this file were fixed in ``configs/preregistration.md``
before any hypothesis test was executed.  See ``supplement/METHODS.md`` for the
audit trail.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Master seed.  Every other seed in the project is derived from this integer.
# --------------------------------------------------------------------------
MASTER_SEED: int = 20260823

# --------------------------------------------------------------------------
# Fixed symbolic vocabularies
# --------------------------------------------------------------------------
#: Functional roles a soma module can differentiate towards.  Fate assignment
#: is *graded* (a softmax weight vector), never a hard categorical label, so
#: that no child can fail to possess a role.
ROLE_NAMES: Tuple[str, ...] = ("sensor", "motor_L", "motor_R", "inter", "memory")
N_ROLES: int = len(ROLE_NAMES)

#: Pre-learning phenotype vector Y = [y1..y6].
TRAIT_NAMES: Tuple[str, ...] = (
    "speed_mean",      # y1  mean spontaneous forward speed        [len/time]
    "gait_freq",       # y2  dominant oscillation frequency        [1/time]
    "turn_bias",       # y3  signed left/right turning bias        [rad/time]
    "recovery_time",   # y4  perturbation recovery time            [time]
    "coherence",       # y5  inter-module phase coherence          [0..1]
    "explore_radius",  # y6  maximum displacement from start       [len]
)
N_TRAITS: int = len(TRAIT_NAMES)

#: Traits whose sign is arbitrary w.r.t. body midline; used only for reporting.
SIGNED_TRAITS: Tuple[str, ...] = ("turn_bias",)


# --------------------------------------------------------------------------
# Model configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GermlineConfig:
    """Structure of the germline / zygotic regulatory genome."""

    n_genes: int = 8          # K regulatory genes: 5 fate logits + w + g + tau_a
    n_modules: int = 8        # N modules in the generic soma U
    founder_amplitude: float = 0.7   # |u| of every founder locus (normalised units)
    mutation_sigma: float = 0.05     # xi ~ N(0, sigma) in normalised locus units
    recombination_p: float = 0.5     # P(locus inherited from the dam)


@dataclass(frozen=True)
class DevelopmentConfig:
    """Endogenous developmental process D(Z, U, E0)."""

    dt: float = 0.05
    t_max: float = 40.0        # integration horizon; individual T_dev <= t_max
    noise_scale: float = 1.0   # multiplier on the germline sigma_dev locus (sweep knob)
    diffusion_scale: float = 1.0    # multiplier on the germline D locus (sweep knob)
    duration_scale: float = 1.0     # multiplier on the germline T_dev locus (sweep knob)
    trace_stride: int = 4      # subsampling of stored developmental trajectories


@dataclass(frozen=True)
class EmbodimentConfig:
    """Standardised newborn evaluation environment E_eval (no learning)."""

    dt: float = 0.05
    t_total: float = 200.0
    t_transient: float = 30.0      # discarded before trait measurement
    t_perturb: float = 100.0       # impulse applied to the perturbed twin
    perturb_magnitude: float = 0.5
    recovery_periods: float = 3.0  # causal MA window, in units of the gait period
    recovery_window_min: float = 6.0   # floor on that window [time]
    recovery_window_max: float = 40.0  # ceiling on that window [time]
    recovery_threshold: float = 0.10   # fraction of post-impulse peak deviation
    recovery_hold: float = 8.0     # deviation must stay below threshold this long [time]
    recovery_censor: float = 90.0  # censoring value if recovery never achieved

    tau_u: float = 1.0            # membrane time constant (fixed, not heritable)
    motor_slope: float = 1.5      # kappa_m in wheel = v_max*clip(kappa_m*drive,0,1)
    v_max: float = 0.5
    wheel_base: float = 0.30
    sensor_gain: float = 1.0
    light_x: float = 5.0
    light_y: float = 5.0
    light_sigma: float = 8.0

    trace_stride: int = 4

    # Newborn initial condition: identical for every individual, deterministic.
    u0_amplitude: float = 0.05


@dataclass(frozen=True)
class RunConfig:
    """Sample sizes and execution parameters."""

    n_seeds_diallel: int = 40      # replicate zygotes per cross cell
    n_backgrounds_swap: int = 60   # matched backgrounds for the causal-swap design
    n_seeds_baseline: int = 80     # replicates per baseline condition
    n_lineages: int = 6            # independent 3-generation lineages
    n_family_per_gen: int = 8      # offspring per mating in the lineage experiment
    n_seeds_robustness: int = 30   # replicates per sweep point
    n_bootstrap: int = 5000
    n_permutation: int = 5000
    chunk_size: int = 256          # individuals simulated per vectorised batch
    smoke_test: bool = False


@dataclass(frozen=True)
class Config:
    germline: GermlineConfig = field(default_factory=GermlineConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)
    embodiment: EmbodimentConfig = field(default_factory=EmbodimentConfig)
    run: RunConfig = field(default_factory=RunConfig)
    master_seed: int = MASTER_SEED

    # ---------------------------------------------------------------- utils
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        """Stable hash of the full configuration (for provenance records)."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def smoke(self) -> "Config":
        """Reduced-cost configuration used by ``--smoke-test``.

        Only sample sizes and integration horizons are reduced.  No model
        parameter that could change a scientific conclusion is altered, and
        smoke-test outputs are written to separate files.
        """
        return replace(
            self,
            development=replace(self.development, t_max=20.0),
            embodiment=replace(self.embodiment, t_total=80.0, t_transient=20.0,
                               t_perturb=40.0, recovery_censor=35.0,
                               recovery_window_max=20.0, recovery_hold=5.0),
            run=replace(
                self.run,
                n_seeds_diallel=4,
                n_backgrounds_swap=6,
                n_seeds_baseline=8,
                n_lineages=2,
                n_family_per_gen=4,
                n_seeds_robustness=4,
                n_bootstrap=200,
                n_permutation=200,
                chunk_size=64,
                smoke_test=True,
            ),
        )


DEFAULT_CONFIG = Config()


# --------------------------------------------------------------------------
# Deterministic seed derivation
# --------------------------------------------------------------------------
def derive_seed(master_seed: int, *labels: Any) -> int:
    """Derive a stable 63-bit seed from a master seed and string labels.

    Uses BLAKE2b rather than :func:`hash` so that the mapping is identical
    across processes, platforms and Python versions.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(str(int(master_seed)).encode("utf-8"))
    for lab in labels:
        h.update(b"\x00")
        h.update(str(lab).encode("utf-8"))
    return int.from_bytes(h.digest(), "big") & ((1 << 63) - 1)


def rng_for(master_seed: int, *labels: Any) -> np.random.Generator:
    """Return an independent PCG64 generator for a named experiment stream."""
    return np.random.default_rng(derive_seed(master_seed, *labels))


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _git_dirty() -> Optional[bool]:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except Exception:
        pass
    return None


def package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in ("numpy", "scipy", "matplotlib", "pandas"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = "not installed"
    return versions


def hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
    }
    try:  # Linux only; best effort
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    info["mem_total"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo", "r") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    return info


def provenance(cfg: Config, experiment: str, extra: Optional[Dict[str, Any]] = None
               ) -> Dict[str, Any]:
    """Full provenance record attached to every raw output file."""
    rec: Dict[str, Any] = {
        "experiment": experiment,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "packages": package_versions(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "master_seed": cfg.master_seed,
        "config_digest": cfg.digest(),
        "config": cfg.to_dict(),
        "hardware": hardware_info(),
        "smoke_test": cfg.run.smoke_test,
    }
    if extra:
        rec["extra"] = extra
    return rec


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
RAW_DIR = os.path.join(OUT_DIR, "raw")
PROC_DIR = os.path.join(OUT_DIR, "processed")
FIG_DIR = os.path.join(OUT_DIR, "figures")
LOG_DIR = os.path.join(OUT_DIR, "logs")


def ensure_dirs() -> None:
    for d in (OUT_DIR, RAW_DIR, PROC_DIR, FIG_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)


def tag(cfg: Config) -> str:
    """Filename suffix separating smoke-test artefacts from full runs."""
    return "_smoke" if cfg.run.smoke_test else ""


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return v if np.isfinite(v) else None
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _sanitise(o: Any) -> Any:
    """Recursively convert numpy scalars/arrays and non-finite floats.

    Non-finite values become ``null`` so that the summary is valid JSON.  They
    are *not* silently turned into zeros: a missing statistic stays missing.
    """
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _sanitise(o.tolist())
    if isinstance(o, dict):
        return {str(k): _sanitise(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitise(v) for v in o]
    return o


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_sanitise(obj), fh, indent=2, allow_nan=False)


def load_json(path: str) -> Any:
    with open(path, "r") as fh:
        return json.load(fh)


def save_raw(name: str, cfg: Config, arrays: Dict[str, Any],
             extra: Optional[Dict[str, Any]] = None) -> str:
    """Persist replicate-level arrays plus a full provenance record."""
    ensure_dirs()
    base = os.path.join(RAW_DIR, f"{name}{tag(cfg)}")
    np.savez_compressed(base + ".npz", **{k: np.asarray(v)
                                          for k, v in arrays.items()})
    save_json(base + "_provenance.json", provenance(cfg, name, extra))
    return base + ".npz"


def load_raw(name: str, cfg: Config) -> Dict[str, np.ndarray]:
    path = os.path.join(RAW_DIR, f"{name}{tag(cfg)}.npz")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def raw_exists(name: str, cfg: Config) -> bool:
    return os.path.exists(os.path.join(RAW_DIR, f"{name}{tag(cfg)}.npz"))


def log_line(cfg: Config, experiment: str, message: str) -> None:
    ensure_dirs()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    line = f"[{stamp}] {experiment}: {message}"
    print(line, flush=True)
    with open(os.path.join(LOG_DIR, f"run{tag(cfg)}.log"), "a") as fh:
        fh.write(line + "\n")
