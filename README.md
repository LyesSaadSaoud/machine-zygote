# Machine Zygote

**A germline–soma architecture for developmental machine heredity — reproducible codebase**

This repository implements and tests a machine-heredity architecture in which
two parent systems carry separate **germline** states, their contributions are
recombined into a **zygote**, and the zygote does *not* encode a finished
controller. Instead it drives an endogenous **developmental** process acting on
an initially generic soma; the resulting **newborn** is measured before any
learning takes place.

> **This is a computational proof of concept.** No physical system was built,
> instrumented or measured. Nothing here constitutes physical validation, and
> controller crossover is *not* claimed to be physical heredity.

---

## Quick start

```bash
pip install -r requirements.txt        # or: conda env create -f environment.yml

python scripts/run_all.py              # ~6 min on one CPU core
python scripts/analyze_all.py
python scripts/make_figures.py
```

Fast check of the whole pipeline (minutes → seconds, separate `*_smoke` outputs):

```bash
python scripts/run_all.py --smoke-test
python scripts/analyze_all.py --smoke-test
python scripts/make_figures.py --smoke-test
```

Tests (no pytest required; `python -m pytest tests/` also works):

```bash
python tests/run_tests.py
```

Or simply `make all`.

Results land in:

| path | contents |
|---|---|
| `outputs/raw/*.npz` | every replicate-level measurement, plus a `_provenance.json` next to each |
| `outputs/processed/*.json` | per-experiment statistics |
| `outputs/figures/` | Fig. P1-1 … P1-8 as PDF **and** 600 dpi PNG |
| `outputs/paper1_summary.json` | machine-readable summary of every result |
| `outputs/PAPER1_RESULTS.md` | human-readable report with exact values |

---

## The model in one page

An individual is `M = (G, S)`: a germline `G` and a soma `S`. The germline is a
vector of 102 loci (99 autosomal, 3 maternally transmitted), each with a
declared biological range.

**Reproduction** `Z_C = R(G_A, G_B, ξ)` — each parent forms a gamete (its
germline plus Gaussian mutation `ξ`); autosomal loci are inherited from either
gamete with p = 0.5; the three maternal-channel loci (developmental duration,
developmental noise amplitude, morphogen steepness) always come from the dam,
so reciprocal crosses *can* differ.

**Development** `S_C⁽⁰⁾ = D(Z_C, U, E₀)` — the generic soma `U` is 8 modules
with identically zero regulatory state, distinguished only by a position on a
body axis. Every module runs the *same* zygotic regulatory network

```
dz_i/dt = −λ_i z_i + tanh( Σ_j W_ij z_j + b_i + s_i(p) ) + D·∇²z + σ_dev·dW
s_i(p)  = s0_i + morph_i · tanh(steep · p)
```

Symmetry is broken endogenously, by positional information and diffusion, not
by any per-module parameter. At the individual's own developmental duration the
state is **frozen**.

**Differentiation** — a *fixed, universal, non-heritable* read-out turns the
frozen state into a controller: graded fate weights over five roles (sensor,
left motor, right motor, interneuron, memory), self-excitation, adaptation
gain, adaptation time constant, and a connectome derived from the similarity of
module expression profiles. Because the read-out has no heritable parameters,
the germline can only influence the newborn *through development*.

**Newborn** — the developed soma drives a differential-drive body in a
standardised environment. There is no learning rule, no reward and no parameter
update; `tests/test_determinism.py` audits this both numerically and at the AST
level. Six pre-learning traits are measured: mean speed, gait frequency,
turning bias, perturbation recovery time, inter-module coherence, exploration
radius.

---

## Experiments

| script | conditions |
|---|---|
| `exp01_parental_crosses` | 4×4 diallel (clones, biparental, reciprocals) + parental reference lines |
| `exp02_causal_swap` | germline swap on matched backgrounds, with re-mutation and re-development nulls |
| `exp03_baselines` | random germline, no-development (two variants), direct-controller crossover, plus ablated diallels |
| `exp04_multigeneration` | G0 → G1 → G2 lineages, **no selection** |
| `exp05_robustness` | sweeps over developmental noise, coupling, mutation amplitude, developmental duration |

---

## Reproducibility

* One master seed (`MASTER_SEED = 20260823` in `src/config.py`); every other
  seed is derived from it by BLAKE2b over string labels, so seeds are stable
  across processes and platforms.
* Each individual's developmental noise stream is keyed on its own seed, never
  on its position in a batch: **chunk size cannot change a result**, and a test
  asserts it.
* Every raw file is written with a provenance record: Python version, package
  versions, git commit and dirty flag, master seed, full config, config digest,
  hardware, and timestamp.
* Re-running any experiment reproduces the previous output bitwise.

## Integrity

The rules this repository was built under are in
[`configs/preregistration.md`](configs/preregistration.md), written before any
hypothesis statistic was computed, including full disclosure of the three
changes made during the non-degeneracy pilot.

* No result is hard-coded; every number in the report is computed from the raw
  arrays at analysis time.
* No seed, individual or cross is discarded. Degenerate individuals are
  **flagged and retained**, and flag rates are reported.
* Failed hypotheses are printed as `HYPOTHESIS NOT SUPPORTED` with the
  statistics that failed them.
* `PDVF` (parental developmental variance fraction) is deliberately *not*
  called narrow-sense heritability, and the report says why.
* Causal language is confined to `exp02`, where every other stochastic draw is
  held fixed.

## Layout

```
src/          germline, zygote, development, soma, embodiment, phenotype,
              reproduction, interventions, metrics, statistics, config
experiments/  exp01 … exp05
scripts/      run_all.py, analyze_all.py, make_figures.py
tests/        germline, development, reproduction, determinism + runner
configs/      preregistration.md
supplement/   METHODS.md
outputs/      raw/ processed/ figures/ logs/
```

## Licence

MIT — see [`LICENSE`](LICENSE).
