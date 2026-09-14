# Pre-registration — Machine Zygote, Paper 1

This document was written **after** the model was made non-degenerate and
**before** any hypothesis statistic was computed. It records what was frozen,
what was changed during the pilot and why, and how each hypothesis will be
judged. It is part of the audit trail; nothing below was edited after results
were inspected.

---

## 1. What is frozen

The complete model is `src/config.py` (`Config`, `MASTER_SEED = 20260823`)
together with the fixed read-out constants in `src/soma.py`. Config digest is
recorded in every raw output file. No experiment script may override a model
parameter except through the declared robustness sweep of exp05.

Germline dimension `d = 102` loci (99 autosomal, 3 maternal-channel).
Founders A–D are rows 1–4 of a Sylvester Hadamard matrix truncated to `d`,
scaled by `founder_amplitude = 0.7`. This rule has no free parameter and was
not adjusted.

## 2. Pilot changes (full disclosure)

Three changes were made after non-degeneracy pilots (n = 32, 48, 64
individuals) and before any hypothesis test. Each was made because a trait had
**no dynamic range**, not because a hypothesis was failing; no factorial
model, no swap contrast and no baseline comparison had been fitted at the time.

| # | Observation in pilot | Change | Rationale |
|---|---|---|---|
| 1 | `speed_mean` had sd = 0.003 on mean 0.253: a sigmoidal motor map makes the time-average of a zero-mean oscillation equal `v_max/2` for every individual. | Motor read-out changed to rectified: `wheel = v_max·clip(slope·drive, 0, 1)`. | A floor/ceiling artefact of the read-out, not a property of the organism. Rectification is also the more physical model (a muscle pulls, it does not push). |
| 2 | Fate weights were near-uniform (mean dominant-fate diversity 1.9 / 5): the between-module spread of the frozen state (~0.2) is small compared with the softmax temperature, so no module ever committed. | Fate logits are mean-subtracted across modules and multiplied by a fixed gain `FATE_GAIN = 5.0`. | Without usable dynamic range in the fate map the soma cannot differentiate at all, so *every* condition would collapse and no hypothesis would be testable. Mean subtraction is a lateral-inhibition analogue and it preserves the correct degenerate behaviour: if development leaves all modules identical the logits vanish and fates stay uniform. |
| 3 | 78 % of individuals were censored on `recovery_time`. Diagnosis: an impulse to a limit cycle produces a permanent *phase* shift, so no moving average of the twin state difference decays; the criterion was measuring residual ripple. | Recovery redefined as the **transverse** deviation — distance from the perturbed state to the nominal orbit point cloud, minus the nominal twin's own distance to that cloud, normalised by the orbit radius of gyration. | The previous definition was dynamically wrong, not merely noisy. The replacement is the standard phase-invariant construction. |

After change 3 the censoring rate was 25 % in the pilot. Residual censoring is
treated as a real dynamical outcome, reported, and never removed from the data.

**No further changes to the model are permitted.** Everything below was run
once, at the sample sizes stated, with the master seed stated.

## 3. Sample sizes (fixed here, before running)

| Experiment | Design | n |
|---|---|---|
| exp01 | 4 × 4 diallel (dam × sire, reciprocals distinct) | 40 offspring per cell → 640 |
| exp01 | parental reference individuals per founder line | 40 → 160 |
| exp02 | matched backgrounds, 5 arms each | 60 → 300 |
| exp03 | 6 conditions | 80 each → 480 |
| exp04 | independent 3-generation lineages | 6 lineages |
| exp05 | 4 swept parameters × 5 levels × 2 cross types | 30 seeds → 1200 |

Bootstrap resamples 5000; permutations 5000; α = 0.05 with Holm correction
across the six traits within each hypothesis family.

## 4. Hypotheses and decision rules

Each hypothesis is judged by a rule stated **now**. "Supported" requires the
stated rule to be met; otherwise the report prints `HYPOTHESIS NOT SUPPORTED`
together with the statistics.

**H1 — biparental contribution.** In the diallel, both the dam main effect and
the sire main effect are significant (Holm-corrected permutation p < 0.05) for
at least one common trait, with partial η² reported and bootstrap CIs on the
effect estimates. *Supported iff* ≥ 1 trait shows both main effects
significant.

**H2 — developmental dependence.** Removing the developmental dynamics changes
phenotype organisation. Judged on two ablations against the full model:
`nodev_static` and the stricter `nodev_quasistatic`. *Supported iff* the
multivariate phenotype distribution differs (permutation test on standardised
centroid distance, p < 0.05) **and** the between-cross variance structure
changes for ≥ 1 trait. The `nodev_static` arm is partly structural (with all
modules identical no fate can be assigned); this is stated as a limitation, and
`nodev_quasistatic` is the informative comparison.

**H3 — pre-learning inheritance.** By construction there is no learning
anywhere in the pipeline. The empirical content of H3 is that the H1 effects
are measured on the newborn: *supported iff* H1 holds, with the code path
audited by `tests/test_no_learning.py`.

**H4 — recombination novelty.** An offspring is *transgressive* on a trait if
its value lies outside `[min(μ_A, μ_B) − 2s, max(μ_A, μ_B) + 2s]`, where μ are
the clone-line means and `s` is the pooled within-clone-line SD. *Supported iff*
the transgression rate of A×B offspring significantly exceeds the rate for
clone-line individuals scored against the same interval (two-proportion
permutation test, Holm-corrected). Random noise alone cannot pass this because
the clone arm carries exactly the same developmental noise.

**H5 — causal swap.** With every other stochastic draw held fixed, replacing
one parental germline changes the phenotype more than re-drawing the mutation
vector with the same parents. *Supported iff* the paired standardised
phenotype shift under swap exceeds the matched no-swap control (paired
permutation test, p < 0.05) for ≥ 1 trait after Holm correction.

## 5. Quantity naming

The parental variance fraction is called **PDVF** (parental developmental
variance fraction). It is *not* narrow-sense heritability: there is no additive
genetic model, no breeding population, no environmental variance partition in
the quantitative-genetic sense, and the "parents" are four fixed founders
rather than a random sample of a population. Both an ANOVA R²-style estimate
and a variance-component estimate are reported.

## 6. Prohibited actions

No result may be produced by: discarding seeds, selecting favourable crosses,
re-running with a different master seed, adjusting a model constant, or
reporting a correlation as causal without the exp02 intervention. If a
hypothesis fails it is reported as failed.
