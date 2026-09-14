"""
Generate every figure of the paper.

matplotlib only (no seaborn).  Colour is avoided: series are separated by
grey level, line style, marker and hatch so that the figures survive
greyscale printing.

Usage::

    python scripts/make_figures.py [--smoke-test]
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (Config, FIG_DIR, OUT_DIR, TRAIT_NAMES, ensure_dirs,
                        load_json, load_raw, log_line, tag)
from src.metrics import Scaler, pca

NAME = "make_figures"

GREYS = ["0.05", "0.35", "0.55", "0.72", "0.85"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
LINES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
HATCHES = ["", "///", "...", "xxx", "\\\\\\", "+++"]

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

TRAIT_LABELS = {
    "speed_mean": "speed",
    "gait_freq": "gait freq.",
    "turn_bias": "turn bias",
    "recovery_time": "recovery",
    "coherence": "coherence",
    "explore_radius": "explor. radius",
}


def _save(fig, name: str, cfg: Config) -> List[str]:
    ensure_dirs()
    base = os.path.join(FIG_DIR, f"{name}{tag(cfg)}")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=600)
    plt.close(fig)
    log_line(cfg, NAME, f"wrote {os.path.basename(base)}.pdf/.png")
    return [base + ".pdf", base + ".png"]


def _despine(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ==========================================================================
# Fig. P1-1  conceptual schematic
# ==========================================================================
def fig_schematic(cfg: Config):
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.set_xlim(-0.15, 10.7)
    ax.set_ylim(-1.05, 3.85)
    ax.axis("off")

    def box(x, y, w, h, title, sub, fc="0.97"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.06,rounding_size=0.08",
                                    linewidth=0.9, edgecolor="0.1", facecolor=fc))
        ax.text(x + w / 2, y + h - 0.24, title, ha="center", va="center",
                fontsize=8.5, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.18, sub, ha="center", va="center",
                fontsize=7, color="0.25", linespacing=1.35)

    def arrow(x0, y0, x1, y1, label=""):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                     arrowstyle="-|>", mutation_scale=8,
                                     linewidth=0.9, color="0.1"))
        if label:
            ax.text((x0 + x1) / 2, max(y0, y1) + 0.12, label, ha="center",
                    va="bottom", fontsize=7.5, style="italic")

    box(0.1, 1.75, 1.55, 1.15, r"$G_A$", "germline\nparent A")
    box(0.1, 0.15, 1.55, 1.15, r"$G_B$", "germline\nparent B")
    box(2.35, 0.95, 1.55, 1.15, r"$Z_C$", "zygote\n" + r"$\mathcal{R}(G_A,G_B,\xi)$")
    box(4.6, 0.95, 1.75, 1.15, r"$\mathcal{D}$", "development\non generic\nsoma $U$")
    box(7.05, 0.95, 1.6, 1.15, r"$S_C^{(0)}$", "newborn\nsoma\n(frozen)")
    box(9.05, 0.95, 1.4, 1.15, r"$Y_C$", "phenotype\n(6 traits)")

    arrow(1.65, 2.33, 2.35, 1.75)
    arrow(1.65, 0.73, 2.35, 1.30)
    arrow(3.90, 1.53, 4.60, 1.53)
    arrow(6.35, 1.53, 7.05, 1.53)
    arrow(8.65, 1.53, 9.05, 1.53, r"$\Phi$")

    ax.add_patch(FancyArrowPatch((5.48, 0.30), (5.48, 0.95), arrowstyle="-|>",
                                 mutation_scale=7, linewidth=0.7, color="0.4"))
    ax.text(5.48, 0.16, r"generic soma $U$, standard environment $E_0$",
            ha="center", va="top", fontsize=7.0, color="0.3")
    ax.text(5.27, 3.55,
            "no learning anywhere on this path — the phenotype is measured "
            "on the newborn",
            ha="center", va="center", fontsize=7.4, style="italic", color="0.15")
    ax.text(-0.15, -0.92,
            "The germline never acts as a controller: it parameterises a "
            "regulatory dynamical system whose developmental output configures "
            "the soma.",
            ha="left", va="bottom", fontsize=6.9, color="0.35")
    return _save(fig, "fig_P1-1_schematic", cfg)


# ==========================================================================
# Fig. P1-2  developmental trajectories
# ==========================================================================
def fig_development(cfg: Config):
    d = load_raw("exp01_parental_crosses", cfg)
    traj, tt = d["traj"], d["traj_t"]          # (I, T, N, K)
    frozen = d["traj_frozen"]
    names = [str(s) for s in d["founder_names"]]
    dam, sire = d["traj_dam"], d["traj_sire"]
    I, T, N, K = traj.shape
    show = min(I, 3)

    fig, axes = plt.subplots(2, show, figsize=(2.35 * show, 3.5),
                             gridspec_kw={"height_ratios": [1.35, 1.0]})
    axes = np.atleast_2d(axes)
    for i in range(show):
        ax = axes[0, i]
        for m in range(N):
            ax.plot(tt, traj[i, :, m, 0], color=str(0.06 + 0.72 * m / max(N - 1, 1)),
                    linewidth=0.8)
        ax.set_title(f"{names[dam[i]]}×{names[sire[i]]}")
        ax.set_xlabel("developmental time")
        if i == 0:
            ax.set_ylabel("gene 1 expression\n(one line per module)")
        _despine(ax)

        ax2 = axes[1, i]
        im = ax2.imshow(frozen[i].T, aspect="auto", cmap="Greys",
                        interpolation="nearest")
        ax2.set_xlabel("module (body axis)")
        if i == 0:
            ax2.set_ylabel("frozen gene")
        ax2.set_xticks(range(N))
        ax2.set_yticks(range(K))
        ax2.tick_params(length=2)
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
    fig.suptitle("Endogenous differentiation of an initially identical soma",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, "fig_P1-2_development", cfg)


# ==========================================================================
# Fig. P1-3  phenotype space
# ==========================================================================
def fig_phenotype_space(cfg: Config, S: Dict):
    d = load_raw("exp01_parental_crosses", cfg)
    Y, dam, sire = d["Y"], d["dam"], d["sire"]
    ref_Y, ref_line = d["ref_Y"], d["ref_line"]
    names = [str(s) for s in d["founder_names"]]
    sc = Scaler(np.array(S["diallel"]["scaler"]["centre"]),
                np.array(S["diallel"]["scaler"]["scale"]))

    Zall = np.vstack([sc.transform(ref_Y), sc.transform(Y)])
    scores, axes_, var, mu = pca(Zall, 2)
    n_ref = ref_Y.shape[0]
    ref_s, off_s = scores[:n_ref], scores[n_ref:]

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax = axs[0]
    for k in range(len(names)):
        m = ref_line == k
        ax.scatter(ref_s[m, 0], ref_s[m, 1], s=26, marker=MARKERS[k],
                   facecolor="none", edgecolor="0.05", linewidth=0.9,
                   label=f"parent {names[k]}", zorder=3)
    hyb = (dam == 0) & (sire == 1)
    rec = (dam == 1) & (sire == 0)
    cln = dam == sire
    ax.scatter(off_s[cln, 0], off_s[cln, 1], s=8, marker=".", color="0.72",
               label="clone offspring", zorder=1)
    ax.scatter(off_s[hyb, 0], off_s[hyb, 1], s=13, marker="x", color="0.15",
               linewidth=0.7, label="A×B", zorder=2)
    ax.scatter(off_s[rec, 0], off_s[rec, 1], s=13, marker="+", color="0.45",
               linewidth=0.7, label="B×A", zorder=2)
    ax.set_xlabel(f"PC1 ({100 * var[0]:.0f}% var.)")
    ax.set_ylabel(f"PC2 ({100 * var[1]:.0f}% var.)")
    ax.set_title("Phenotype space (standardised traits)")
    ax.legend(frameon=False, loc="best", handletextpad=0.4)
    _despine(ax)

    ax = axs[1]
    pos = np.arange(len(TRAIT_NAMES))
    width = 0.38
    Zh = sc.transform(Y[hyb])
    for k, (lab, sel) in enumerate([("A×A", (dam == 0) & (sire == 0)),
                                    ("B×B", (dam == 1) & (sire == 1))]):
        Zc = sc.transform(Y[sel])
        ax.bar(pos + (k - 0.5) * width, Zc.mean(0), width,
               yerr=Zc.std(0, ddof=1) / np.sqrt(Zc.shape[0]),
               facecolor="none", edgecolor="0.1", linewidth=0.8,
               hatch=HATCHES[k + 1], label=lab, error_kw={"linewidth": 0.7})
    ax.errorbar(pos, Zh.mean(0), yerr=Zh.std(0, ddof=1) / np.sqrt(Zh.shape[0]),
                fmt="o", color="0.05", markersize=3.5, linewidth=0.8,
                label="A×B", zorder=4)
    ax.axhline(0, color="0.6", linewidth=0.6)
    ax.set_xticks(pos)
    ax.set_xticklabels([TRAIT_LABELS[t] for t in TRAIT_NAMES], rotation=35,
                       ha="right")
    ax.set_ylabel("trait (SD units, parental scale)")
    ax.set_title("Clone lines vs biparental offspring")
    ax.legend(frameon=False)
    _despine(ax)
    fig.tight_layout()
    return _save(fig, "fig_P1-3_phenotype_space", cfg)


# ==========================================================================
# Fig. P1-4  factorial parental contributions
# ==========================================================================
def fig_factorial(cfg: Config, S: Dict):
    d = load_raw("exp01_parental_crosses", cfg)
    names = [str(s) for s in d["founder_names"]]
    per = S["diallel"]["per_trait"]
    nf = len(names)

    fig, axs = plt.subplots(2, len(TRAIT_NAMES) // 2 + len(TRAIT_NAMES) % 2,
                            figsize=(7.2, 4.4))
    axs = axs.ravel()
    for i, rec in enumerate(per):
        ax = axs[i]
        cm = np.array(rec["anova"]["cell_means"])
        for j in range(nf):
            ax.plot(np.arange(nf), cm[j], marker=MARKERS[j], markersize=3.2,
                    linestyle=LINES[j % len(LINES)],
                    color=GREYS[j % len(GREYS)], label=f"dam {names[j]}")
        ax.set_title(f"{TRAIT_LABELS[rec['trait']]}\n"
                     f"$\\eta^2_p$ dam {rec['anova']['partial_eta2']['dam']:.2f}, "
                     f"sire {rec['anova']['partial_eta2']['sire']:.2f}",
                     fontsize=7.6)
        ax.set_xticks(np.arange(nf))
        ax.set_xticklabels([f"sire {n}" for n in names], fontsize=6.4,
                           rotation=25, ha="right")
        ax.tick_params(length=2)
        _despine(ax)
    for k in range(len(per), len(axs)):
        axs[k].axis("off")
    axs[0].legend(frameon=False, fontsize=6.2, ncol=2, handlelength=1.6)
    fig.suptitle("Factorial parental contributions (cell means of the 4×4 diallel)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "fig_P1-4_factorial_effects", cfg)


# ==========================================================================
# Fig. P1-5  causal swap
# ==========================================================================
def fig_causal_swap(cfg: Config, S: Dict):
    sw = S["swap"]
    arms = ["swap_sire", "swap_dam", "null_remut", "null_redev"]
    labels = ["swap sire\n(intervention)", "swap dam\n(intervention)",
              "re-mutation\n(null)", "re-development\n(null)"]
    pos = np.arange(len(TRAIT_NAMES))
    width = 0.2

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0),
                            gridspec_kw={"width_ratios": [2.1, 1]})
    ax = axs[0]
    for k, arm in enumerate(arms):
        vals = [r[arm]["mean_shift"] if arm.startswith("swap")
                else r[f"{arm}_mean_shift"] for r in sw["per_trait"]]
        ax.bar(pos + (k - 1.5) * width, vals, width, facecolor="none",
               edgecolor="0.05", linewidth=0.8, hatch=HATCHES[k],
               label=labels[k])
    for i, r in enumerate(sw["per_trait"]):
        for k, arm in enumerate(("swap_sire", "swap_dam")):
            if r[arm].get("significant"):
                y = r[arm]["mean_shift"]
                ax.text(pos[i] + (k - 1.5) * width, y, "*", ha="center",
                        va="bottom", fontsize=9)
    ax.set_xticks(pos)
    ax.set_xticklabels([TRAIT_LABELS[t] for t in TRAIT_NAMES], rotation=35,
                       ha="right")
    ax.set_ylabel("mean |Δ phenotype| (SD units)")
    ax.set_title(f"Germline swap on matched backgrounds "
                 f"(base {sw['base_cross']}, n={sw['n_backgrounds']})")
    ax.legend(frameon=False, ncol=2)
    _despine(ax)

    ax = axs[1]
    mv = sw["multivariate"]
    keys = ["swap_sire", "swap_dam", "null_remut", "null_redev"]
    means = [mv[k]["mean_distance"] for k in keys]
    errs = [mv[k]["sd"] / np.sqrt(sw["n_backgrounds"]) for k in keys]
    ax.bar(np.arange(len(keys)), means, 0.6, yerr=errs, facecolor="none",
           edgecolor="0.05", linewidth=0.8,
           hatch=[HATCHES[i] for i in range(len(keys))],
           error_kw={"linewidth": 0.7})
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(["swap\nsire", "swap\ndam", "re-mut\n(null)",
                        "re-dev\n(null)"], fontsize=6.6)
    ax.set_ylabel("multivariate distance (SD units)")
    ax.set_title("Whole-phenotype shift")
    _despine(ax)
    fig.tight_layout()
    return _save(fig, "fig_P1-5_causal_swap", cfg)


# ==========================================================================
# Fig. P1-6  baselines
# ==========================================================================
def fig_baselines(cfg: Config, S: Dict):
    d = load_raw("exp03_baselines", cfg)
    conds = [str(c) for c in d["condition_names"]]
    sc = Scaler(np.array(S["diallel"]["scaler"]["centre"]),
                np.array(S["diallel"]["scaler"]["scale"]))
    short = {"biparental": "bipar.", "clone_A": "clone A", "clone_B": "clone B",
             "random_germline": "random", "nodev_static": "no-dev\n(static)",
             "nodev_quasistatic": "no-dev\n(quasi)",
             "direct_controller": "direct\ncontroller"}

    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6))
    axs = axs.ravel()
    for ti, tname in enumerate(TRAIT_NAMES):
        ax = axs[ti]
        data = [sc.transform(d[f"Y_{c}"])[:, ti] for c in conds]
        bp = ax.boxplot(data, widths=0.6, showfliers=False, patch_artist=True)
        for k, box in enumerate(bp["boxes"]):
            box.set(facecolor="none", edgecolor="0.1", linewidth=0.7,
                    hatch=HATCHES[k % len(HATCHES)])
        for part in ("whiskers", "caps", "medians"):
            for art in bp[part]:
                art.set(color="0.1", linewidth=0.7)
        ax.set_xticks(np.arange(1, len(conds) + 1))
        ax.set_xticklabels([short[c] for c in conds], rotation=45, ha="right",
                           fontsize=5.8)
        ax.set_title(TRAIT_LABELS[tname], fontsize=8)
        if ti % 3 == 0:
            ax.set_ylabel("SD units")
        ax.tick_params(length=2)
        _despine(ax)
    fig.suptitle("Condition comparison: clone, biparental, random germline, "
                 "no-development, direct-controller", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, "fig_P1-6_baselines", cfg)


# ==========================================================================
# Fig. P1-7  lineage
# ==========================================================================
def fig_lineage(cfg: Config, S: Dict):
    d = load_raw("exp04_multigeneration", cfg)
    Y, gen, lin, fam = d["Y"], d["generation"], d["lineage"], d["family"]
    dam_Y, sire_Y = d["dam_Y"], d["sire_Y"]
    sc = Scaler(np.array(S["diallel"]["scaler"]["centre"]),
                np.array(S["diallel"]["scaler"]["scale"]))
    Z = sc.transform(Y)

    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.55,
                          wspace=0.35)

    # --- lineage tree of one lineage --------------------------------------
    ax = fig.add_subplot(gs[0, :2])
    l0 = lin == lin.min()
    for g in (0, 1, 2):
        m = l0 & (gen == g)
        fams = np.unique(fam[m])
        for f in fams:
            sel = m & (fam == f)
            x = np.linspace(f - 0.32, f + 0.32, int(sel.sum()))
            ax.scatter(x, np.full(x.size, 2 - g), s=9, marker="o",
                       facecolor=str(0.15 + 0.3 * g), edgecolor="none")
        ax.text(-0.95, 2 - g, f"G{g}", fontsize=8, va="center")
    # true pedigree edges, read from the stored mating table
    matings = d["g1_matings"]
    nfam = matings.shape[0]
    for f in range(nfam):
        for parent in matings[f]:
            ax.annotate("", xy=(f, 1.0), xytext=(int(parent), 2.0),
                        arrowprops=dict(arrowstyle="-", linewidth=0.5,
                                        color="0.6"))
        for parent in (f, (f + 1) % nfam):
            ax.annotate("", xy=(f, 0.0), xytext=(parent, 1.0),
                        arrowprops=dict(arrowstyle="-", linewidth=0.5,
                                        color="0.6"))
    ax.set_xlim(-1.1, 3.7)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"family {i}" for i in range(4)], fontsize=6.5)
    ax.set_yticks([])
    ax.set_title("Three-generation lineage (one of "
                 f"{S['multigeneration']['n_lineages']}; no selection applied)",
                 fontsize=8.5)
    _despine(ax)
    ax.spines["left"].set_visible(False)

    # --- germline drift ----------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    pg = S["multigeneration"]["per_generation"]
    gvals = [pg[str(g)]["germline_drift_mean"] for g in (0, 1, 2)]
    gerr = [pg[str(g)]["germline_drift_sd"] for g in (0, 1, 2)]
    ax.errorbar([0, 1, 2], gvals, yerr=gerr, fmt="o-", color="0.1",
                markersize=3.5, linewidth=0.9, capsize=2)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["G0", "G1", "G2"])
    ax.set_ylabel("germline distance to\nnearest founder")
    ax.set_title("Drift", fontsize=8.5)
    _despine(ax)

    # --- midparent regressions --------------------------------------------
    reg = S["multigeneration"]["midparent_regression"]
    for j, ti in enumerate([0, 1, 4]):
        ax = fig.add_subplot(gs[1, j])
        for g, mk, col in ((1, "o", "0.6"), (2, "s", "0.1")):
            m = (gen == g) & np.all(np.isfinite(dam_Y), axis=1)
            mid = 0.5 * (sc.transform(dam_Y[m])[:, ti]
                         + sc.transform(sire_Y[m])[:, ti])
            ax.scatter(mid, Z[m][:, ti], s=5, marker=mk, facecolor="none",
                       edgecolor=col, linewidth=0.5, label=f"G{g}")
            r = reg[str(g)][ti]
            if np.isfinite(r["slope"]):
                xs = np.linspace(mid.min(), mid.max(), 10)
                ax.plot(xs, r["slope"] * xs + r["intercept"], color=col,
                        linewidth=0.9)
        ax.set_xlabel("midparent (SD units)", fontsize=7)
        ax.set_ylabel(f"offspring {TRAIT_LABELS[TRAIT_NAMES[ti]]}", fontsize=7)
        if j == 0:
            ax.legend(frameon=False, fontsize=6.2)
        _despine(ax)
    return _save(fig, "fig_P1-7_lineage", cfg)


# ==========================================================================
# Fig. P1-8  robustness
# ==========================================================================
def fig_robustness(cfg: Config, S: Dict):
    rb = S["robustness"]
    knobs = list(rb["knobs"].keys())
    fig, axs = plt.subplots(2, len(knobs), figsize=(7.4, 4.2), sharey="row")
    if len(knobs) == 1:
        axs = axs.reshape(2, 1)
    pretty = {"dev_noise_scale": "developmental noise (×)",
              "coupling_scale": "inter-module coupling (×)",
              "mutation_sigma": "mutation amplitude",
              "dev_duration_scale": "developmental duration (×)"}

    for j, k in enumerate(knobs):
        pts = rb["knobs"][k]["points"]
        xs = [p["value"] for p in pts]

        ax = axs[0, j]
        n_sig = [p["n_traits_both_parents_significant"] for p in pts]
        ax.plot(xs, n_sig, marker="o", markersize=3.4, color="0.1",
                linewidth=0.9)
        ax.axhline(0, color="0.7", linewidth=0.6, linestyle=":")
        ax.set_ylim(-0.3, len(TRAIT_NAMES) + 0.3)
        ax.set_title(pretty.get(k, k), fontsize=7.6)
        if j == 0:
            ax.set_ylabel("traits with both\nparental effects")
        _despine(ax)

        ax = axs[1, j]
        for ti, tname in enumerate(TRAIT_NAMES):
            vals = [next((q["pdvf"] for q in p["per_trait"]
                          if q["trait"] == tname), np.nan) for p in pts]
            ax.plot(xs, vals, marker=MARKERS[ti % len(MARKERS)], markersize=2.8,
                    linestyle=LINES[ti % len(LINES)],
                    color=GREYS[ti % len(GREYS)], linewidth=0.8,
                    label=TRAIT_LABELS[tname])
        ax.set_xlabel(pretty.get(k, k), fontsize=7)
        if j == 0:
            ax.set_ylabel("PDVF")
        _despine(ax)
    axs[1, 0].legend(frameon=False, fontsize=5.6, ncol=2, handlelength=1.6)
    fig.suptitle("Robustness of the biparental effect across parameter sweeps",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "fig_P1-8_robustness", cfg)


# ==========================================================================
def main(cfg: Config):
    ensure_dirs()
    summary_path = os.path.join(OUT_DIR, f"paper1_summary{tag(cfg)}.json")
    if not os.path.exists(summary_path):
        raise SystemExit("run scripts/analyze_all.py first")
    S = load_json(summary_path)

    made = []
    made += fig_schematic(cfg)
    made += fig_development(cfg)
    made += fig_phenotype_space(cfg, S)
    made += fig_factorial(cfg, S)
    made += fig_causal_swap(cfg, S)
    made += fig_baselines(cfg, S)
    made += fig_lineage(cfg, S)
    made += fig_robustness(cfg, S)
    log_line(cfg, NAME, f"{len(made) // 2} figures written to {FIG_DIR}")
    return made


if __name__ == "__main__":
    cfg = Config()
    if "--smoke-test" in sys.argv:
        cfg = cfg.smoke()
    main(cfg)
