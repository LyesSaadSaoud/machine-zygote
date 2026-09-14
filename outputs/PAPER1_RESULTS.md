# Machine Zygote — Paper 1 results package

Generated 2026-08-23T11:43:09+00:00 · master seed `20260823` · config digest `8270e321ac1fc1c7`

This is a **computational proof of concept**. Nothing here is a physical experiment, no physical robot was built or measured, and no claim of physical validation is made.

## Hypothesis outcomes

| Hypothesis | Claim | Outcome |
|---|---|---|
| H1 | both parental germlines contribute to newborn phenotype | **SUPPORTED** |
| H2 | removing development changes phenotype organisation | **HYPOTHESIS NOT SUPPORTED** |
| H3 | the difference exists before any learning | **SUPPORTED** |
| H4 | recombination yields transgressive offspring | **SUPPORTED** |
| H5 | swapping one germline causally changes traits | **SUPPORTED** |

HYPOTHESIS NOT SUPPORTED: H2. The statistics behind each verdict are below; nothing was removed or re-run to change these outcomes.

## Design and sample sizes

* Germline: 102 loci (99 autosomal, 3 maternal-channel), founders A, B, C, D.
* Diallel: 4×4 cells × 40 replicates = 640 offspring, plus 160 parental reference individuals.
* Causal swap: 60 matched backgrounds × 5 arms.
* Baselines: 7 conditions.
* Lineages: 6 independent 3-generation lineages, 576 individuals.
* Robustness: 4 knobs × 5 levels × 30 replicates per cell.
* Total individuals simulated across all experiments: 6076.

Mean dominant-fate diversity of biparental newborns: 3.18 of 5 roles; mean between-module dispersion of the frozen state 0.196.

Flag rates in the diallel (individuals are flagged, never discarded): `motionless` 0.0%, `non_oscillatory` 0.0%, `recovery_censored` 26.9%, `nonfinite` 0.0%.

## H1 — biparental contribution

Figures: `outputs/figures/fig_P1-3_phenotype_space.pdf`, `outputs/figures/fig_P1-4_factorial_effects.pdf`

| trait | F(dam) | p(dam) Holm | η²p dam | F(sire) | p(sire) Holm | η²p sire | F(int) | p(int) Holm | PDVF [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| speed_mean | 11.03 | 0.0012 | 0.050 | 17.25 | 0.0012 | 0.077 | 18.86 | 0.0012 | 0.379 [0.343, 0.459] |
| gait_freq | 70.09 | 0.0012 | 0.252 | 69.95 | 0.0012 | 0.252 | 12.07 | 0.0012 | 0.533 [0.484, 0.602] |
| turn_bias | 13.14 | 0.0012 | 0.059 | 30.98 | 0.0012 | 0.130 | 12.95 | 0.0012 | 0.360 [0.303, 0.455] |
| recovery_time | 1.59 | 0.2026 | 0.008 | 1.91 | 0.1300 | 0.009 | 4.22 | 0.0012 | 0.082 [0.063, 0.178] |
| coherence | 21.12 | 0.0012 | 0.092 | 16.25 | 0.0012 | 0.072 | 16.31 | 0.0012 | 0.377 [0.349, 0.449] |
| explore_radius | 49.87 | 0.0012 | 0.193 | 24.69 | 0.0012 | 0.106 | 7.17 | 0.0012 | 0.378 [0.342, 0.457] |

Both parental main effects are significant after Holm correction for **5 of 6** traits: speed_mean, gait_freq, turn_bias, coherence, explore_radius.

`PDVF` is the *parental developmental variance fraction*. It is **not** narrow-sense heritability h²: there is no additive genetic model, no breeding population and no quantitative-genetic variance partition; the parents are four fixed founders. Both a descriptive R²-style value and a variance-component value are stored in the JSON summary.

### Reciprocal cross (A×B vs B×A)

| trait | mean A×B | mean B×A | difference [95% CI] | g | p Holm |
|---|---|---|---|---|---|
| speed_mean | 0.1626 | 0.1558 | 0.0068 [-0.0097, 0.0231] | 0.17 | 1.0000 |
| gait_freq | 0.1001 | 0.0972 | 0.0029 [-0.0028, 0.0090] | 0.21 | 1.0000 |
| turn_bias | 6.673e-05 | 0.0291 | -0.0290 [-0.0850, 0.0294] | -0.21 | 1.0000 |
| recovery_time | 41.0500 | 41.9100 | -0.8600 [-15.2252, 13.4157] | -0.02 | 1.0000 |
| coherence | 0.6941 | 0.6202 | 0.0739 [-0.0173, 0.1616] | 0.35 | 0.5819 |
| explore_radius | 17.5329 | 12.4396 | 5.0934 [0.0168, 10.2736] | 0.43 | 0.3215 |

0 of 6 traits differ between reciprocal crosses. The model contains 3 maternal-channel loci (developmental duration, developmental noise amplitude, morphogen steepness), so a reciprocal difference is possible by construction; whether it is detectable is the empirical result above.

## H2 — developmental dependence

Figure: `outputs/figures/fig_P1-6_baselines.pdf`

| condition | n | mean role diversity | centroid distance from biparental (SD units) | p | traits differing |
|---|---|---|---|---|---|
| clone_A | 80 | 4.22 | 1.907 | 2.000e-04 | 5/6 |
| clone_B | 80 | 3.80 | 3.063 | 2.000e-04 | 5/6 |
| random_germline | 80 | 2.76 | 1.263 | 2.000e-04 | 1/6 |
| nodev_static | 80 | 1.00 | 2.764 | 2.000e-04 | 3/6 |
| nodev_quasistatic | 80 | 2.73 | 0.561 | 0.1540 | 0/6 |
| direct_controller | 80 | 4.10 | 1.964 | 2.000e-04 | 3/6 |
| biparental (model) | 80 | 2.94 | — | — | — |

Effect of removing development on the **parental variance structure** (full diallel re-run under each ablation):

| ablation | trait | PDVF ablated | PDVF full | change |
|---|---|---|---|---|
| nodev_static | speed_mean | 0.533 | 0.379 | 0.154 |
| nodev_static | gait_freq | 0.681 | 0.533 | 0.148 |
| nodev_static | turn_bias | n/a | 0.360 | None |
| nodev_static | recovery_time | 0.083 | 0.082 | 1.479e-04 |
| nodev_static | coherence | 0.498 | 0.377 | 0.122 |
| nodev_static | explore_radius | 0.639 | 0.378 | 0.261 |
| nodev_quasistatic | speed_mean | 0.548 | 0.379 | 0.169 |
| nodev_quasistatic | gait_freq | 0.760 | 0.533 | 0.227 |
| nodev_quasistatic | turn_bias | 0.688 | 0.360 | 0.328 |
| nodev_quasistatic | recovery_time | 0.085 | 0.082 | 0.002 |
| nodev_quasistatic | coherence | 0.410 | 0.377 | 0.033 |
| nodev_quasistatic | explore_radius | 0.525 | 0.378 | 0.147 |

**HYPOTHESIS NOT SUPPORTED (H2).** The pre-registered rule was a conjunction and the two halves came apart:

* the phenotype *centroid* of `nodev_quasistatic` offspring is **not** distinguishable from the full model (distance 0.561 SD units, permutation p = 0.1540; 0 of 6 traits differ after correction) — this half of the rule fails;
* the parental *variance structure* does change, and substantially: PDVF rises under the ablation on most traits (table above) — this half of the rule is met.

Read plainly: in this model the dynamical developmental process is **not** what makes offspring resemble their parents. Removing it leaves the average newborn where it was and *increases* the share of variance attributable to the parents, because the regulatory dynamics and their noise contribute non-parental variance of their own. Development here shapes how much of the phenotype is parentally determined, not whether it is. The conjunction was written before the data existed and is not relaxed after the fact.

*Caveat.* nodev_static is partly structural: with all modules identical no fate can be assigned, so its collapse is expected by construction and is reported separately. Its centroid does differ from the full model (distance 2.764, p = 2.000e-04), and its mean dominant-fate diversity is exactly 1.00 of 5 roles, as the construction requires.

## H3 — pre-learning inheritance

There is no learning rule, no reward, no plasticity and no parameter update anywhere between development freezing and phenotype measurement. Every trait in this report is measured on the newborn with the developed soma held constant; `tests/test_no_learning.py` asserts this at the level of the simulation state. H3 therefore holds exactly when H1 holds, and its verdict tracks H1.

## H4 — recombination novelty (transgressive segregation)

An offspring is transgressive on a trait when it falls outside the parental clone-line means widened by 2 pooled within-line SDs. The identical criterion is applied to clone-line individuals, giving the noise-only rate.

| trait | interval | within-line SD | hybrid rate | clone rate | difference | p Holm |
|---|---|---|---|---|---|---|
| speed_mean | [0.1076, 0.2200] | 0.0162 | 15.0% | 1.2% | 13.7 pp | 0.0200 |
| gait_freq | [0.0746, 0.1110] | 0.0044 | 17.5% | 0.0% | 17.5 pp | 0.0012 |
| turn_bias | [-0.3753, 0.1666] | 0.0857 | 11.2% | 2.5% | 8.8 pp | 0.2368 |
| recovery_time | [-16.6563, 138.3563] | 32.5157 | 0.0% | 0.0% | 0.0 pp | 1.0000 |
| coherence | [0.2819, 1.0331] | 0.0791 | 0.0% | 0.0% | 0.0 pp | 1.0000 |
| explore_radius | [-4.1925, 41.0286] | 4.6213 | 6.2% | 0.0% | 6.2 pp | 0.2368 |

Significant transgressive excess on: speed_mean, gait_freq.

## H5 — causal germline swap

Figure: `outputs/figures/fig_P1-5_causal_swap.pdf`

Base cross AxB; alternative dam C, alternative sire D; 60 backgrounds with the recombination mask, both mutation vectors and the developmental-noise seed held fixed.

| trait | shift, swap sire | shift, swap dam | shift, re-mutation (null) | shift, re-development (null) | excess (sire) p Holm | excess (dam) p Holm |
|---|---|---|---|---|---|---|
| speed_mean | 0.879 | 0.796 | 0.308 | 0.326 | 0.0012 | 0.0012 |
| gait_freq | 1.255 | 2.564 | 0.342 | 0.258 | 0.0012 | 0.0012 |
| turn_bias | 0.733 | 1.494 | 0.470 | 0.429 | 0.0308 | 0.0012 |
| recovery_time | 0.992 | 1.231 | 0.958 | 1.002 | 0.8080 | 0.0564 |
| coherence | 0.894 | 0.798 | 0.355 | 0.304 | 0.0012 | 0.0012 |
| explore_radius | 0.986 | 1.045 | 0.332 | 0.280 | 0.0012 | 0.0012 |

Shifts are mean |Δ| in standardised trait units relative to the matched base individual.

Significant causal effects: speed_mean::swap_sire, gait_freq::swap_sire, turn_bias::swap_sire, coherence::swap_sire, explore_radius::swap_sire, speed_mean::swap_dam, gait_freq::swap_dam, turn_bias::swap_dam, coherence::swap_dam, explore_radius::swap_dam.

## Three generations (no selection)

Figure: `outputs/figures/fig_P1-7_lineage.pdf`

No selection operator exists anywhere in the code. Cross-generation change is recombination, mutation and developmental noise, i.e. drift -- not evolution.

| generation | n | germline drift from nearest founder | mean within-family phenotype SD (SD units) |
|---|---|---|---|
| G0 | 192 | 0.0000 | 0.386 |
| G1 | 192 | 0.6597 | 1.078 |
| G2 | 192 | 0.7367 | 1.023 |

Midparent–offspring regression slopes (standardised units):

| trait | G1 slope [95% CI] | G1 p Holm | G2 slope [95% CI] | G2 p Holm |
|---|---|---|---|---|
| speed_mean | -0.095 [-0.326, 0.134] | 1.0000 | 0.471 [0.252, 0.682] | 0.0030 |
| gait_freq | 1.570 [0.875, 2.406] | 0.0030 | 0.529 [0.228, 0.821] | 0.0030 |
| turn_bias | 0.054 [-0.140, 0.250] | 1.0000 | 0.329 [0.062, 0.613] | 0.0255 |
| recovery_time | -0.133 [-0.325, 0.058] | 0.6537 | -0.156 [-0.339, 0.043] | 0.0900 |
| coherence | -0.290 [-0.578, -0.001] | 0.1799 | 0.406 [0.221, 0.587] | 0.0030 |
| explore_radius | -0.259 [-0.872, 0.340] | 1.0000 | 0.257 [0.084, 0.442] | 0.0255 |

5 of 6 traits show significant midparent-offspring resemblance in G2, against 1 in G1. The asymmetry is expected rather than surprising: a G1 individual's parents are two founder-line reference individuals whose own phenotypes differ only by developmental noise around a fixed line mean, so the midparent value carries almost no heritable information. G1 parents, by contrast, are genuinely genetically variable, which is why the G2 regression has something to detect. This is a statement about the design, not evidence of a change in the mechanism across generations.

## Robustness

Figure: `outputs/figures/fig_P1-8_robustness.pdf`

The H1 effect (both parental main effects significant on at least one trait) is recovered at 100.0% of the 20 sweep points.

| knob | values | sweep points retaining the effect |
|---|---|---|
| coupling_scale | 0.000, 0.500, 1.000, 2.000, 4.000 | 5/5 |
| dev_duration_scale | 0.250, 0.500, 1.000, 1.500, 2.000 | 5/5 |
| dev_noise_scale | 0.250, 0.500, 1.000, 2.000, 4.000 | 5/5 |
| mutation_sigma | 0.000, 0.025, 0.050, 0.100, 0.200 | 5/5 |

No sweep point loses the effect.

## Figures

* `outputs/figures/fig_P1-1_schematic.pdf` / `.png` (600 dpi) — conceptual schematic G_A + G_B -> Z -> D -> S -> Y
* `outputs/figures/fig_P1-2_development.pdf` / `.png` (600 dpi) — developmental trajectories of the soma modules
* `outputs/figures/fig_P1-3_phenotype_space.pdf` / `.png` (600 dpi) — parent and offspring phenotype space (PCA)
* `outputs/figures/fig_P1-4_factorial_effects.pdf` / `.png` (600 dpi) — factorial parental contribution plots
* `outputs/figures/fig_P1-5_causal_swap.pdf` / `.png` (600 dpi) — causal germline-swap intervention
* `outputs/figures/fig_P1-6_baselines.pdf` / `.png` (600 dpi) — clone / biparental / random / no-development / direct-controller baselines
* `outputs/figures/fig_P1-7_lineage.pdf` / `.png` (600 dpi) — three-generation lineage
* `outputs/figures/fig_P1-8_robustness.pdf` / `.png` (600 dpi) — robustness and parameter sweep

## Integrity statement

* Simulation only; no physical system was built or measured.
* No result was hard-coded; every number above is computed from the `.npz` files in `outputs/raw/` by `scripts/analyze_all.py`.
* No seed, individual or cross was discarded. Degenerate individuals are flagged and retained, and flag rates are reported.
* Model constants were frozen in `configs/preregistration.md` before any hypothesis statistic was computed; the three pilot changes made before freezing are disclosed there in full.
* Causal language is used only for the exp02 intervention, in which every other stochastic draw is held fixed.
* The direct-controller baseline is *ordinary controller inheritance*; it is reported as a comparison, and no claim is made that developmental heredity here is physical heredity.
* Multigeneration results involve no selection and are therefore drift and recombination, not evolution.

