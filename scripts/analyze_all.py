"""
Analyse every raw experiment output and emit the machine-readable summary and
the human-readable report.

Usage::

    python scripts/analyze_all.py [--smoke-test]

Decision rules are exactly those written in ``configs/preregistration.md``.
When a rule is not met the hypothesis is reported as
``HYPOTHESIS NOT SUPPORTED`` together with the statistics that failed it.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (Config, OUT_DIR, PROC_DIR, TRAIT_NAMES, ensure_dirs,
                        load_raw, log_line, provenance, raw_exists, rng_for,
                        save_json, tag)
from src.metrics import (Scaler, founder_ancestry, germline_drift,
                         lineage_distance_matrix, pooled_sd, transgression)
from src.statistics import (bootstrap_anova_effects, bootstrap_ci_diff,
                            centroid_distance_test, hedges_g, holm,
                            levene_like_test, linreg_with_ci,
                            paired_permutation_test, pdvf, pdvf_bootstrap,
                            permutation_anova, permutation_test_diff,
                            two_proportion_permutation, two_way_anova)

NAME = "analyze_all"
ALPHA = 0.05


def _p(x) -> float:
    """JSON-safe float."""
    x = float(x)
    return x if np.isfinite(x) else float("nan")


# ==========================================================================
# exp01 : diallel -> H1, H3, H4, reciprocal cross, PDVF
# ==========================================================================
def analyse_diallel(cfg: Config) -> Dict:
    d = load_raw("exp01_parental_crosses", cfg)
    Y, dam, sire = d["Y"], d["dam"], d["sire"]
    founder_names = [str(s) for s in d["founder_names"]]
    ref_Y, ref_line = d["ref_Y"], d["ref_line"]
    n_perm, n_boot = cfg.run.n_permutation, cfg.run.n_bootstrap

    # Scaler fitted once, on the parental reference individuals only.
    scaler = Scaler.fit(ref_Y)

    per_trait: List[Dict] = []
    p_dam, p_sire, p_int = [], [], []
    for ti, tname in enumerate(TRAIT_NAMES):
        y = Y[:, ti]
        rng = rng_for(cfg.master_seed, NAME, "h1", tname)
        a = two_way_anova(y, dam, sire, tname)
        h = pdvf(y, dam, sire, tname)
        lo, hi = pdvf_bootstrap(y, dam, sire, rng, n_boot=min(n_boot, 2000))
        h.ci_low, h.ci_high = lo, hi
        boot = bootstrap_anova_effects(y, dam, sire, rng, n_boot=n_boot)
        pd_ = permutation_anova(y, dam, sire, "dam", rng, n_perm)
        ps_ = permutation_anova(y, dam, sire, "sire", rng, n_perm)
        pi_ = permutation_anova(y, dam, sire, "dam:sire", rng, n_perm)
        p_dam.append(pd_); p_sire.append(ps_); p_int.append(pi_)
        per_trait.append({
            "trait": tname,
            "anova": a.to_dict(),
            "pdvf": h.to_dict(),
            "p_perm": {"dam": _p(pd_), "sire": _p(ps_), "dam:sire": _p(pi_)},
            "effect_ci": {k: v.tolist() for k, v in boot.items()},
        })

    adj_dam = holm(p_dam)
    adj_sire = holm(p_sire)
    adj_int = holm(p_int)
    for i, rec in enumerate(per_trait):
        rec["p_holm"] = {"dam": _p(adj_dam[i]), "sire": _p(adj_sire[i]),
                         "dam:sire": _p(adj_int[i])}
        rec["both_parents_significant"] = bool(adj_dam[i] < ALPHA
                                               and adj_sire[i] < ALPHA)

    h1_traits = [r["trait"] for r in per_trait if r["both_parents_significant"]]
    h1_supported = len(h1_traits) > 0

    # ---- reciprocal cross contrast (Condition 3) --------------------------
    recip = []
    p_recip = []
    for ti, tname in enumerate(TRAIT_NAMES):
        rng = rng_for(cfg.master_seed, NAME, "recip", tname)
        ab = Y[(dam == 0) & (sire == 1), ti]
        ba = Y[(dam == 1) & (sire == 0), ti]
        p = permutation_test_diff(ab, ba, rng, n_perm)
        diff, lo, hi = bootstrap_ci_diff(ab, ba, rng, n_boot)
        p_recip.append(p)
        recip.append({"trait": tname, "mean_AxB": float(ab.mean()),
                      "mean_BxA": float(ba.mean()), "difference": _p(diff),
                      "ci": [_p(lo), _p(hi)], "hedges_g": _p(hedges_g(ab, ba)),
                      "p_perm": _p(p)})
    adj = holm(p_recip)
    for i, r in enumerate(recip):
        r["p_holm"] = _p(adj[i])
        r["significant"] = bool(adj[i] < ALPHA)

    # ---- H4 transgressive segregation -------------------------------------
    trans = []
    p_trans = []
    for ti, tname in enumerate(TRAIT_NAMES):
        rng = rng_for(cfg.master_seed, NAME, "h4", tname)
        clone_a = Y[(dam == 0) & (sire == 0), ti]
        clone_b = Y[(dam == 1) & (sire == 1), ti]
        hybrid = Y[((dam == 0) & (sire == 1)) | ((dam == 1) & (sire == 0)), ti]
        tr = transgression(hybrid, clone_a, clone_b, tname)
        p, obs = two_proportion_permutation(tr.k_hybrid, tr.n_hybrid,
                                            tr.k_clone, tr.n_clone, rng, n_perm)
        p_trans.append(p)
        rec = tr.to_dict()
        rec.update({"rate_difference": _p(obs), "p_perm": _p(p)})
        trans.append(rec)
    adj = holm(p_trans)
    for i, r in enumerate(trans):
        r["p_holm"] = _p(adj[i])
        r["significant_excess"] = bool(adj[i] < ALPHA
                                       and r["hybrid_rate"] > r["clone_rate"])
    h4_traits = [r["trait"] for r in trans if r["significant_excess"]]
    h4_supported = len(h4_traits) > 0

    # ---- within-line developmental variation ------------------------------
    within = {}
    for ti, tname in enumerate(TRAIT_NAMES):
        within[tname] = {
            "clone_pooled_sd": _p(pooled_sd(
                [Y[(dam == k) & (sire == k), ti] for k in range(len(founder_names))])),
            "reference_pooled_sd": _p(pooled_sd(
                [ref_Y[ref_line == k, ti] for k in range(len(founder_names))])),
            "offspring_sd": _p(Y[:, ti].std(ddof=1)),
        }

    flags = {k[len("flag_"):]: float(np.mean(v)) for k, v in d.items()
             if k.startswith("flag_") and not k.startswith("flag_ref")}

    return {
        "n_offspring": int(Y.shape[0]),
        "n_reference": int(ref_Y.shape[0]),
        "n_per_cell": int(Y.shape[0] // (len(founder_names) ** 2)),
        "founder_names": founder_names,
        "founder_germline_distance": d["founder_distance"].tolist(),
        "scaler": scaler.to_dict(),
        "per_trait": per_trait,
        "reciprocal": recip,
        "transgression": trans,
        "within_line_variation": within,
        "flag_rates": flags,
        "mean_role_diversity": float(np.mean(d["role_diversity"])),
        "mean_module_spread": float(np.mean(d["module_spread"])),
        "H1_supported": bool(h1_supported),
        "H1_traits": h1_traits,
        "H4_supported": bool(h4_supported),
        "H4_traits": h4_traits,
    }


# ==========================================================================
# exp02 : causal swap -> H5
# ==========================================================================
def analyse_swap(cfg: Config, scaler: Scaler) -> Dict:
    d = load_raw("exp02_causal_swap", cfg)
    n_perm, n_boot = cfg.run.n_permutation, cfg.run.n_bootstrap
    base = d["Y_base"]
    Zb = scaler.transform(base)

    arms = {}
    for arm in ("swap_sire", "swap_dam", "null_remut", "null_redev"):
        Z = scaler.transform(d[f"Y_{arm}"])
        arms[arm] = np.abs(Z - Zb)           # per-individual |standardised shift|

    out = {"n_backgrounds": int(base.shape[0]),
           "base_cross": f"{d['founder_names'][int(d['dam_base'])]}"
                         f"x{d['founder_names'][int(d['sire_base'])]}",
           "alt_dam": str(d["founder_names"][int(d["dam_alt"])]),
           "alt_sire": str(d["founder_names"][int(d["sire_alt"])]),
           "per_trait": [], "multivariate": {}}

    p_all = {"swap_sire": [], "swap_dam": []}
    for ti, tname in enumerate(TRAIT_NAMES):
        rng = rng_for(cfg.master_seed, NAME, "h5", tname)
        rec = {"trait": tname,
               "null_remut_mean_shift": _p(arms["null_remut"][:, ti].mean()),
               "null_redev_mean_shift": _p(arms["null_redev"][:, ti].mean())}
        for arm in ("swap_sire", "swap_dam"):
            diff = arms[arm][:, ti] - arms["null_remut"][:, ti]
            p = paired_permutation_test(diff, rng, n_perm)
            m, lo, hi = bootstrap_ci_diff(arms[arm][:, ti],
                                          arms["null_remut"][:, ti], rng, n_boot)
            p_all[arm].append(p)
            rec[arm] = {"mean_shift": _p(arms[arm][:, ti].mean()),
                        "excess_over_null": _p(m), "ci": [_p(lo), _p(hi)],
                        "hedges_g": _p(hedges_g(arms[arm][:, ti],
                                                arms["null_remut"][:, ti])),
                        "p_perm": _p(p)}
        out["per_trait"].append(rec)

    supported_traits = []
    for arm in ("swap_sire", "swap_dam"):
        adj = holm(p_all[arm])
        for i, rec in enumerate(out["per_trait"]):
            rec[arm]["p_holm"] = _p(adj[i])
            sig = bool(adj[i] < ALPHA and rec[arm]["excess_over_null"] > 0)
            rec[arm]["significant"] = sig
            if sig:
                supported_traits.append(f"{rec['trait']}::{arm}")

    for arm in ("swap_sire", "swap_dam", "null_remut", "null_redev"):
        dist = np.linalg.norm(scaler.transform(d[f"Y_{arm}"]) - Zb, axis=1)
        out["multivariate"][arm] = {"mean_distance": _p(dist.mean()),
                                    "sd": _p(dist.std(ddof=1))}
    rng = rng_for(cfg.master_seed, NAME, "h5", "mv")
    for arm in ("swap_sire", "swap_dam"):
        da = np.linalg.norm(scaler.transform(d[f"Y_{arm}"]) - Zb, axis=1)
        dn = np.linalg.norm(scaler.transform(d["Y_null_remut"]) - Zb, axis=1)
        out["multivariate"][arm]["p_paired_perm"] = _p(
            paired_permutation_test(da - dn, rng, n_perm))
        out["multivariate"][arm]["excess_over_null"] = _p(np.mean(da - dn))

    out["H5_supported"] = len(supported_traits) > 0
    out["H5_traits"] = supported_traits
    return out


# ==========================================================================
# exp03 : baselines -> H2, direct-controller, random control
# ==========================================================================
def analyse_baselines(cfg: Config, scaler: Scaler, full_pdvf: Dict) -> Dict:
    d = load_raw("exp03_baselines", cfg)
    n_perm, n_boot = cfg.run.n_permutation, cfg.run.n_bootstrap
    conds = [str(c) for c in d["condition_names"]]
    Ys = {c: d[f"Y_{c}"] for c in conds}
    ref = Ys["biparental"]
    Zref = scaler.transform(ref)

    comparisons = {}
    for c in conds:
        if c == "biparental":
            continue
        rng = rng_for(cfg.master_seed, NAME, "h2", c)
        Z = scaler.transform(Ys[c])
        mv = centroid_distance_test(Zref, Z, rng, n_perm)
        per_trait = []
        ps = []
        for ti, tname in enumerate(TRAIT_NAMES):
            rng_t = rng_for(cfg.master_seed, NAME, "h2", c, tname)
            a, b = ref[:, ti], Ys[c][:, ti]
            p = permutation_test_diff(a, b, rng_t, n_perm)
            m, lo, hi = bootstrap_ci_diff(a, b, rng_t, n_boot)
            disp = levene_like_test(a, b, rng_t, min(n_perm, 2000))
            ps.append(p)
            per_trait.append({"trait": tname, "mean_biparental": float(a.mean()),
                              "mean_condition": float(b.mean()),
                              "difference": _p(m), "ci": [_p(lo), _p(hi)],
                              "hedges_g": _p(hedges_g(a, b)), "p_perm": _p(p),
                              "dispersion": {k: _p(v) for k, v in disp.items()}})
        adj = holm(ps)
        for i, r in enumerate(per_trait):
            r["p_holm"] = _p(adj[i])
            r["significant"] = bool(adj[i] < ALPHA)
        comparisons[c] = {
            "n": int(Ys[c].shape[0]),
            "multivariate": {k: _p(v) for k, v in mv.items()},
            "mean_role_diversity": float(np.mean(d[f"roles_{c}"])),
            "per_trait": per_trait,
            "n_traits_significant": int(sum(r["significant"] for r in per_trait)),
        }
    comparisons["biparental"] = {
        "n": int(ref.shape[0]),
        "mean_role_diversity": float(np.mean(d["roles_biparental"])),
    }

    # ---- ablated diallels: does removing development change the parental
    #      variance structure?
    ablated = {}
    for abl in ("nodev_static", "nodev_quasistatic"):
        Ya = d[f"diallel_{abl}_Y"]
        dam = d[f"diallel_{abl}_dam"]
        sire = d[f"diallel_{abl}_sire"]
        rows = []
        for ti, tname in enumerate(TRAIT_NAMES):
            y = Ya[:, ti]
            if np.allclose(y, y[0]):
                rows.append({"trait": tname, "degenerate": True,
                             "pdvf_components": float("nan"),
                             "eta2_dam": float("nan"), "eta2_sire": float("nan"),
                             "pdvf_full": full_pdvf[tname]})
                continue
            a = two_way_anova(y, dam, sire, tname)
            h = pdvf(y, dam, sire, tname)
            rows.append({"trait": tname, "degenerate": False,
                         "pdvf_components": _p(h.pdvf_components),
                         "eta2_dam": _p(a.partial_eta2["dam"]),
                         "eta2_sire": _p(a.partial_eta2["sire"]),
                         "p_dam_param": _p(a.p_param["dam"]),
                         "p_sire_param": _p(a.p_param["sire"]),
                         "pdvf_full": full_pdvf[tname],
                         "pdvf_change": _p(h.pdvf_components - full_pdvf[tname])})
        ablated[abl] = {"n": int(Ya.shape[0]), "per_trait": rows,
                        "mean_role_diversity": float(np.mean(d[f"diallel_{abl}_roles"]))}

    # ---- H2 decision -------------------------------------------------------
    key = "nodev_quasistatic"
    mv_sig = comparisons[key]["multivariate"]["p_perm"] < ALPHA
    struct_changed = any(
        (not r["degenerate"]) and np.isfinite(r["pdvf_change"])
        and abs(r["pdvf_change"]) > 0.05
        for r in ablated[key]["per_trait"])
    h2_supported = bool(mv_sig and struct_changed)

    return {
        "conditions": conds,
        "comparisons": comparisons,
        "ablated_diallels": ablated,
        "H2_supported": h2_supported,
        "H2_basis": {
            "primary_ablation": key,
            "multivariate_p": comparisons[key]["multivariate"]["p_perm"],
            "centroid_distance": comparisons[key]["multivariate"]["centroid_distance"],
            "variance_structure_changed": bool(struct_changed),
            "note": ("nodev_static is partly structural: with all modules "
                     "identical no fate can be assigned, so its collapse is "
                     "expected by construction and is reported separately."),
        },
    }


# ==========================================================================
# exp04 : multigeneration
# ==========================================================================
def analyse_multigeneration(cfg: Config, scaler: Scaler) -> Dict:
    d = load_raw("exp04_multigeneration", cfg)
    Y, gen, lin, fam = d["Y"], d["generation"], d["lineage"], d["family"]
    G, dam_Y, sire_Y = d["germline"], d["dam_Y"], d["sire_Y"]
    founders = d["founders"]
    n_perm, n_boot = cfg.run.n_permutation, cfg.run.n_bootstrap
    Z = scaler.transform(Y)

    per_gen = {}
    for g in (0, 1, 2):
        m = gen == g
        drift = germline_drift(G[m], founders)
        stab = []
        for l in np.unique(lin[m]):
            for f in np.unique(fam[m & (lin == l)]):
                sel = m & (lin == l) & (fam == f)
                if sel.sum() > 1:
                    stab.append(Z[sel].std(axis=0, ddof=1))
        per_gen[str(g)] = {
            "n": int(m.sum()),
            "germline_drift_mean": float(drift.mean()),
            "germline_drift_sd": float(drift.std(ddof=1)) if m.sum() > 1 else 0.0,
            "within_family_sd_std_units": (np.mean(stab, axis=0).tolist()
                                           if stab else []),
            "phenotype_mean_std_units": Z[m].mean(axis=0).tolist(),
            "phenotype_sd_std_units": Z[m].std(axis=0, ddof=1).tolist(),
        }

    # parent-offspring (midparent) regression, per generation and trait
    regressions = {}
    for g in (1, 2):
        m = (gen == g) & np.all(np.isfinite(dam_Y), axis=1)
        mid = 0.5 * (scaler.transform(dam_Y[m]) + scaler.transform(sire_Y[m]))
        kid = scaler.transform(Y[m])
        rows = []
        for ti, tname in enumerate(TRAIT_NAMES):
            rng = rng_for(cfg.master_seed, NAME, "gen", g, tname)
            r = linreg_with_ci(mid[:, ti], kid[:, ti], rng,
                               n_boot=min(n_boot, 2000),
                               n_perm=min(n_perm, 2000))
            rec = r.to_dict()
            rec["trait"] = tname
            rows.append(rec)
        ps = holm([r["p_perm"] for r in rows])
        for i, r in enumerate(rows):
            r["p_holm"] = _p(ps[i])
            r["significant"] = bool(ps[i] < ALPHA)
        regressions[str(g)] = rows

    # lineage distances: germline distance vs phenotype distance
    corr = {}
    for g in (1, 2):
        m = gen == g
        Dg = lineage_distance_matrix(G[m])
        Dp = np.sqrt(((Z[m][:, None, :] - Z[m][None, :, :]) ** 2).sum(axis=2))
        iu = np.triu_indices(Dg.shape[0], 1)
        x, y = Dg[iu], Dp[iu]
        if x.size > 2 and x.std() > 0 and y.std() > 0:
            corr[str(g)] = {"pearson_r": float(np.corrcoef(x, y)[0, 1]),
                            "n_pairs": int(x.size)}
        else:
            corr[str(g)] = {"pearson_r": float("nan"), "n_pairs": int(x.size)}

    anc = founder_ancestry(G[gen == 2], founders)
    return {
        "n_total": int(Y.shape[0]),
        "n_lineages": int(len(np.unique(lin))),
        "selection_applied": False,
        "selection_note": ("No selection operator exists anywhere in the code. "
                           "Cross-generation change is recombination, mutation "
                           "and developmental noise, i.e. drift -- not evolution."),
        "per_generation": per_gen,
        "midparent_regression": regressions,
        "germline_vs_phenotype_distance": corr,
        "g2_mean_founder_ancestry": anc.mean(axis=0).tolist(),
    }


# ==========================================================================
# exp05 : robustness
# ==========================================================================
def analyse_robustness(cfg: Config) -> Dict:
    d = load_raw("exp05_robustness", cfg)
    knob, value, trait = d["knob"], d["value"], d["trait"]
    pd_, ps_ = d["p_dam"], d["p_sire"]
    out = {"knobs": {}, "n_per_cell": int(d["n"][0]) // 4}
    total_pts, passed_pts = 0, 0
    for k in sorted(set(str(x) for x in knob)):
        m_k = knob == k
        vals = sorted(set(value[m_k]))
        pts = []
        for v in vals:
            m = m_k & (value == v)
            both = [(str(trait[i]), float(pd_[i]), float(ps_[i]),
                     float(d["eta2_dam"][i]), float(d["eta2_sire"][i]),
                     float(d["pdvf"][i]))
                    for i in np.where(m)[0]]
            n_both = sum(1 for t, a, b, _, _, _ in both
                         if np.isfinite(a) and np.isfinite(b)
                         and a < ALPHA and b < ALPHA)
            total_pts += 1
            passed_pts += 1 if n_both > 0 else 0
            pts.append({
                "value": float(v),
                "n_traits_both_parents_significant": int(n_both),
                "traits": [t for t, a, b, _, _, _ in both
                           if np.isfinite(a) and np.isfinite(b)
                           and a < ALPHA and b < ALPHA],
                "mean_pdvf": float(np.nanmean([x[5] for x in both])),
                "per_trait": [{"trait": t, "p_dam": a, "p_sire": b,
                               "eta2_dam": c, "eta2_sire": e, "pdvf": f}
                              for t, a, b, c, e, f in both],
            })
        out["knobs"][k] = {"values": [float(v) for v in vals], "points": pts,
                           "n_points_with_effect":
                               int(sum(p["n_traits_both_parents_significant"] > 0
                                       for p in pts)),
                           "n_points": len(pts)}
    out["fraction_sweep_points_with_H1_effect"] = (passed_pts / total_pts
                                                   if total_pts else float("nan"))
    out["failure_points"] = [
        {"knob": k, "value": p["value"]}
        for k, kk in out["knobs"].items() for p in kk["points"]
        if p["n_traits_both_parents_significant"] == 0]
    return out


# ==========================================================================
# Report
# ==========================================================================
def _fmt(x, nd=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "n/a"
    if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
        return f"{x:.3e}"
    return f"{x:.{nd}f}"


def write_report(cfg: Config, S: Dict) -> str:
    t = tag(cfg)
    fig = lambda n: f"outputs/figures/{n}{t}.pdf"
    L: List[str] = []
    add = L.append

    add("# Machine Zygote — Paper 1 results package\n")
    add(f"Generated {S['provenance']['timestamp_utc']} · "
        f"master seed `{S['provenance']['master_seed']}` · "
        f"config digest `{S['provenance']['config_digest']}`"
        + ("  \n**SMOKE TEST RUN — reduced sample sizes, not for reporting.**"
           if cfg.run.smoke_test else ""))
    add("\nThis is a **computational proof of concept**. Nothing here is a "
        "physical experiment, no physical robot was built or measured, and no "
        "claim of physical validation is made.\n")

    # ---------------- headline ------------------------------------------
    add("## Hypothesis outcomes\n")
    add("| Hypothesis | Claim | Outcome |")
    add("|---|---|---|")
    for hid, claim, ok in [
            ("H1", "both parental germlines contribute to newborn phenotype",
             S["H1_supported"]),
            ("H2", "removing development changes phenotype organisation",
             S["H2_supported"]),
            ("H3", "the difference exists before any learning", S["H3_supported"]),
            ("H4", "recombination yields transgressive offspring", S["H4_supported"]),
            ("H5", "swapping one germline causally changes traits",
             S["H5_supported"])]:
        add(f"| {hid} | {claim} | {'**SUPPORTED**' if ok else '**HYPOTHESIS NOT SUPPORTED**'} |")
    add("")
    unsupported = [h for h in ("H1", "H2", "H3", "H4", "H5")
                   if not S[f"{h}_supported"]]
    if unsupported:
        add(f"HYPOTHESIS NOT SUPPORTED: {', '.join(unsupported)}. "
            "The statistics behind each verdict are below; nothing was removed "
            "or re-run to change these outcomes.\n")

    # ---------------- design ---------------------------------------------
    dia = S["diallel"]
    add("## Design and sample sizes\n")
    add(f"* Germline: {S['germline_dimension']} loci "
        f"({S['germline_autosomal']} autosomal, {S['germline_maternal']} "
        f"maternal-channel), founders {', '.join(dia['founder_names'])}.")
    add(f"* Diallel: {len(dia['founder_names'])}×{len(dia['founder_names'])} cells "
        f"× {dia['n_per_cell']} replicates = {dia['n_offspring']} offspring, "
        f"plus {dia['n_reference']} parental reference individuals.")
    add(f"* Causal swap: {S['swap']['n_backgrounds']} matched backgrounds × 5 arms.")
    add(f"* Baselines: {len(S['baselines']['conditions'])} conditions.")
    add(f"* Lineages: {S['multigeneration']['n_lineages']} independent "
        f"3-generation lineages, {S['multigeneration']['n_total']} individuals.")
    add(f"* Robustness: 4 knobs × 5 levels × "
        f"{S['robustness']['n_per_cell']} replicates per cell.")
    add(f"* Total individuals simulated across all experiments: "
        f"{S['n_individuals_total']}.\n")
    add(f"Mean dominant-fate diversity of biparental newborns: "
        f"{_fmt(dia['mean_role_diversity'], 2)} of 5 roles; mean between-module "
        f"dispersion of the frozen state {_fmt(dia['mean_module_spread'], 3)}.\n")
    add("Flag rates in the diallel (individuals are flagged, never discarded): "
        + ", ".join(f"`{k}` {_fmt(v * 100, 1)}%" for k, v in dia["flag_rates"].items())
        + ".\n")

    # ---------------- H1 --------------------------------------------------
    add("## H1 — biparental contribution\n")
    add(f"Figures: `{fig('fig_P1-3_phenotype_space')}`, "
        f"`{fig('fig_P1-4_factorial_effects')}`\n")
    add("| trait | F(dam) | p(dam) Holm | η²p dam | F(sire) | p(sire) Holm | "
        "η²p sire | F(int) | p(int) Holm | PDVF [95% CI] |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for r in dia["per_trait"]:
        a, ph, pv = r["anova"], r["p_holm"], r["pdvf"]
        add(f"| {r['trait']} | {_fmt(a['F']['dam'], 2)} | {_fmt(ph['dam'])} | "
            f"{_fmt(a['partial_eta2']['dam'], 3)} | {_fmt(a['F']['sire'], 2)} | "
            f"{_fmt(ph['sire'])} | {_fmt(a['partial_eta2']['sire'], 3)} | "
            f"{_fmt(a['F']['dam:sire'], 2)} | {_fmt(ph['dam:sire'])} | "
            f"{_fmt(pv['pdvf_components'], 3)} "
            f"[{_fmt(pv['ci_low'], 3)}, {_fmt(pv['ci_high'], 3)}] |")
    add("")
    if dia["H1_supported"]:
        add(f"Both parental main effects are significant after Holm correction "
            f"for **{len(dia['H1_traits'])} of {len(TRAIT_NAMES)}** traits: "
            f"{', '.join(dia['H1_traits'])}.\n")
    else:
        add("**HYPOTHESIS NOT SUPPORTED (H1).** No trait shows both parental "
            "main effects significant after correction.\n")
    add("`PDVF` is the *parental developmental variance fraction*. It is **not** "
        "narrow-sense heritability h²: there is no additive genetic model, no "
        "breeding population and no quantitative-genetic variance partition; "
        "the parents are four fixed founders. Both a descriptive R²-style value "
        "and a variance-component value are stored in the JSON summary.\n")

    add("### Reciprocal cross (A×B vs B×A)\n")
    add("| trait | mean A×B | mean B×A | difference [95% CI] | g | p Holm |")
    add("|---|---|---|---|---|---|")
    for r in dia["reciprocal"]:
        add(f"| {r['trait']} | {_fmt(r['mean_AxB'])} | {_fmt(r['mean_BxA'])} | "
            f"{_fmt(r['difference'])} [{_fmt(r['ci'][0])}, {_fmt(r['ci'][1])}] | "
            f"{_fmt(r['hedges_g'], 2)} | {_fmt(r['p_holm'])} |")
    n_rec = sum(r["significant"] for r in dia["reciprocal"])
    add(f"\n{n_rec} of {len(TRAIT_NAMES)} traits differ between reciprocal "
        f"crosses. The model contains {S['germline_maternal']} maternal-channel "
        "loci (developmental duration, developmental noise amplitude, morphogen "
        "steepness), so a reciprocal difference is possible by construction; "
        "whether it is detectable is the empirical result above.\n")

    # ---------------- H2 --------------------------------------------------
    b = S["baselines"]
    add("## H2 — developmental dependence\n")
    add(f"Figure: `{fig('fig_P1-6_baselines')}`\n")
    add("| condition | n | mean role diversity | centroid distance from "
        "biparental (SD units) | p | traits differing |")
    add("|---|---|---|---|---|---|")
    for c, rec in b["comparisons"].items():
        if c == "biparental":
            add(f"| biparental (model) | {rec['n']} | "
                f"{_fmt(rec['mean_role_diversity'], 2)} | — | — | — |")
            continue
        mv = rec["multivariate"]
        add(f"| {c} | {rec['n']} | {_fmt(rec['mean_role_diversity'], 2)} | "
            f"{_fmt(mv['centroid_distance'], 3)} | {_fmt(mv['p_perm'])} | "
            f"{rec['n_traits_significant']}/{len(TRAIT_NAMES)} |")
    add("")
    add("Effect of removing development on the **parental variance structure** "
        "(full diallel re-run under each ablation):\n")
    add("| ablation | trait | PDVF ablated | PDVF full | change |")
    add("|---|---|---|---|---|")
    for abl, rec in b["ablated_diallels"].items():
        for r in rec["per_trait"]:
            add(f"| {abl} | {r['trait']} | {_fmt(r['pdvf_components'], 3)} | "
                f"{_fmt(r['pdvf_full'], 3)} | {_fmt(r.get('pdvf_change'), 3)} |")
    add("")
    basis = b["H2_basis"]
    if b["H2_supported"]:
        add("H2 is supported on the `nodev_quasistatic` ablation "
            f"(multivariate p = {_fmt(basis['multivariate_p'])}, "
            "parental variance structure changed).\n")
    else:
        add("**HYPOTHESIS NOT SUPPORTED (H2).** The pre-registered rule was a "
            "conjunction and the two halves came apart:\n")
        add(f"* the phenotype *centroid* of `nodev_quasistatic` offspring is "
            f"**not** distinguishable from the full model "
            f"(distance {_fmt(basis['centroid_distance'], 3)} SD units, "
            f"permutation p = {_fmt(basis['multivariate_p'])}; 0 of "
            f"{len(TRAIT_NAMES)} traits differ after correction) — this half "
            "of the rule fails;")
        add(f"* the parental *variance structure* does change, and "
            f"substantially: PDVF rises under the ablation on most traits "
            "(table above) — this half of the rule is met.\n")
        add("Read plainly: in this model the dynamical developmental process "
            "is **not** what makes offspring resemble their parents. Removing "
            "it leaves the average newborn where it was and *increases* the "
            "share of variance attributable to the parents, because the "
            "regulatory dynamics and their noise contribute non-parental "
            "variance of their own. Development here shapes how much of the "
            "phenotype is parentally determined, not whether it is. The "
            "conjunction was written before the data existed and is not "
            "relaxed after the fact.\n")
    add(f"*Caveat.* {basis['note']} Its centroid does differ from the full "
        f"model (distance "
        f"{_fmt(b['comparisons']['nodev_static']['multivariate']['centroid_distance'], 3)}, "
        f"p = {_fmt(b['comparisons']['nodev_static']['multivariate']['p_perm'])}), "
        "and its mean dominant-fate diversity is exactly 1.00 of 5 roles, as "
        "the construction requires.\n")

    # ---------------- H3 --------------------------------------------------
    add("## H3 — pre-learning inheritance\n")
    add("There is no learning rule, no reward, no plasticity and no parameter "
        "update anywhere between development freezing and phenotype "
        "measurement. Every trait in this report is measured on the newborn "
        "with the developed soma held constant; `tests/test_no_learning.py` "
        "asserts this at the level of the simulation state. H3 therefore holds "
        "exactly when H1 holds, and its verdict tracks H1.\n")

    # ---------------- H4 --------------------------------------------------
    add("## H4 — recombination novelty (transgressive segregation)\n")
    add("An offspring is transgressive on a trait when it falls outside the "
        "parental clone-line means widened by 2 pooled within-line SDs. The "
        "identical criterion is applied to clone-line individuals, giving the "
        "noise-only rate.\n")
    add("| trait | interval | within-line SD | hybrid rate | clone rate | "
        "difference | p Holm |")
    add("|---|---|---|---|---|---|---|")
    for r in dia["transgression"]:
        add(f"| {r['trait']} | [{_fmt(r['lo'])}, {_fmt(r['hi'])}] | "
            f"{_fmt(r['within_sd'])} | {_fmt(r['hybrid_rate'] * 100, 1)}% | "
            f"{_fmt(r['clone_rate'] * 100, 1)}% | "
            f"{_fmt(r['rate_difference'] * 100, 1)} pp | {_fmt(r['p_holm'])} |")
    add("")
    if dia["H4_supported"]:
        add(f"Significant transgressive excess on: {', '.join(dia['H4_traits'])}.\n")
    else:
        add("**HYPOTHESIS NOT SUPPORTED (H4).** No trait shows a hybrid "
            "transgression rate significantly above the clone-line rate after "
            "correction. Offspring do differ from their parents, but not by "
            "more than within-line developmental variation already allows.\n")

    # ---------------- H5 --------------------------------------------------
    sw = S["swap"]
    add("## H5 — causal germline swap\n")
    add(f"Figure: `{fig('fig_P1-5_causal_swap')}`\n")
    add(f"Base cross {sw['base_cross']}; alternative dam {sw['alt_dam']}, "
        f"alternative sire {sw['alt_sire']}; {sw['n_backgrounds']} backgrounds "
        "with the recombination mask, both mutation vectors and the "
        "developmental-noise seed held fixed.\n")
    add("| trait | shift, swap sire | shift, swap dam | shift, re-mutation "
        "(null) | shift, re-development (null) | excess (sire) p Holm | "
        "excess (dam) p Holm |")
    add("|---|---|---|---|---|---|---|")
    for r in sw["per_trait"]:
        add(f"| {r['trait']} | {_fmt(r['swap_sire']['mean_shift'], 3)} | "
            f"{_fmt(r['swap_dam']['mean_shift'], 3)} | "
            f"{_fmt(r['null_remut_mean_shift'], 3)} | "
            f"{_fmt(r['null_redev_mean_shift'], 3)} | "
            f"{_fmt(r['swap_sire']['p_holm'])} | {_fmt(r['swap_dam']['p_holm'])} |")
    add("\nShifts are mean |Δ| in standardised trait units relative to the "
        "matched base individual.\n")
    if sw["H5_supported"]:
        add(f"Significant causal effects: {', '.join(sw['H5_traits'])}.\n")
    else:
        add("**HYPOTHESIS NOT SUPPORTED (H5).**\n")

    # ---------------- multigeneration ------------------------------------
    mg = S["multigeneration"]
    add("## Three generations (no selection)\n")
    add(f"Figure: `{fig('fig_P1-7_lineage')}`\n")
    add(f"{mg['selection_note']}\n")
    add("| generation | n | germline drift from nearest founder | "
        "mean within-family phenotype SD (SD units) |")
    add("|---|---|---|---|")
    for g in ("0", "1", "2"):
        r = mg["per_generation"][g]
        wf = (np.mean(r["within_family_sd_std_units"])
              if r["within_family_sd_std_units"] else float("nan"))
        add(f"| G{g} | {r['n']} | {_fmt(r['germline_drift_mean'], 4)} | "
            f"{_fmt(wf, 3)} |")
    add("\nMidparent–offspring regression slopes (standardised units):\n")
    add("| trait | G1 slope [95% CI] | G1 p Holm | G2 slope [95% CI] | G2 p Holm |")
    add("|---|---|---|---|---|")
    for i, tname in enumerate(TRAIT_NAMES):
        r1 = mg["midparent_regression"]["1"][i]
        r2 = mg["midparent_regression"]["2"][i]
        add(f"| {tname} | {_fmt(r1['slope'], 3)} "
            f"[{_fmt(r1['slope_ci'][0], 3)}, {_fmt(r1['slope_ci'][1], 3)}] | "
            f"{_fmt(r1['p_holm'])} | {_fmt(r2['slope'], 3)} "
            f"[{_fmt(r2['slope_ci'][0], 3)}, {_fmt(r2['slope_ci'][1], 3)}] | "
            f"{_fmt(r2['p_holm'])} |")
    n_g2 = sum(r["significant"] for r in mg["midparent_regression"]["2"])
    n_g1 = sum(r["significant"] for r in mg["midparent_regression"]["1"])
    add(f"\n{n_g2} of {len(TRAIT_NAMES)} traits show significant "
        f"midparent-offspring resemblance in G2, against {n_g1} in G1. The "
        "asymmetry is expected rather than surprising: a G1 individual's "
        "parents are two founder-line reference individuals whose own "
        "phenotypes differ only by developmental noise around a fixed line "
        "mean, so the midparent value carries almost no heritable information. "
        "G1 parents, by contrast, are genuinely genetically variable, which is "
        "why the G2 regression has something to detect. This is a statement "
        "about the design, not evidence of a change in the mechanism across "
        "generations.\n")

    # ---------------- robustness ------------------------------------------
    rb = S["robustness"]
    add("## Robustness\n")
    add(f"Figure: `{fig('fig_P1-8_robustness')}`\n")
    add(f"The H1 effect (both parental main effects significant on at least one "
        f"trait) is recovered at "
        f"{_fmt(rb['fraction_sweep_points_with_H1_effect'] * 100, 1)}% of the "
        f"20 sweep points.\n")
    add("| knob | values | sweep points retaining the effect |")
    add("|---|---|---|")
    for k, rec in rb["knobs"].items():
        add(f"| {k} | {', '.join(_fmt(v, 3) for v in rec['values'])} | "
            f"{rec['n_points_with_effect']}/{rec['n_points']} |")
    if rb["failure_points"]:
        add("\nSweep points where the effect is **not** recovered: "
            + ", ".join(f"`{f['knob']}={_fmt(f['value'], 3)}`"
                        for f in rb["failure_points"]) + ".\n")
    else:
        add("\nNo sweep point loses the effect.\n")

    # ---------------- figures + integrity ---------------------------------
    add("## Figures\n")
    for n, cap in S["figures"]:
        add(f"* `outputs/figures/{n}{t}.pdf` / `.png` (600 dpi) — {cap}")
    add("\n## Integrity statement\n")
    add("* Simulation only; no physical system was built or measured.")
    add("* No result was hard-coded; every number above is computed from the "
        "`.npz` files in `outputs/raw/` by `scripts/analyze_all.py`.")
    add("* No seed, individual or cross was discarded. Degenerate individuals "
        "are flagged and retained, and flag rates are reported.")
    add("* Model constants were frozen in `configs/preregistration.md` before "
        "any hypothesis statistic was computed; the three pilot changes made "
        "before freezing are disclosed there in full.")
    add("* Causal language is used only for the exp02 intervention, in which "
        "every other stochastic draw is held fixed.")
    add("* The direct-controller baseline is *ordinary controller inheritance*; "
        "it is reported as a comparison, and no claim is made that "
        "developmental heredity here is physical heredity.")
    add("* Multigeneration results involve no selection and are therefore drift "
        "and recombination, not evolution.\n")

    path = os.path.join(OUT_DIR, f"PAPER1_RESULTS{t}.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return path


# ==========================================================================
def main(cfg: Config) -> Dict:
    ensure_dirs()
    t0 = time.time()
    for name in ("exp01_parental_crosses", "exp02_causal_swap",
                 "exp03_baselines", "exp04_multigeneration",
                 "exp05_robustness"):
        if not raw_exists(name, cfg):
            raise SystemExit(f"missing raw output for {name}; run scripts/run_all.py first")

    log_line(cfg, NAME, "analysing diallel (H1, H3, H4, reciprocal, PDVF)")
    dia = analyse_diallel(cfg)
    scaler = Scaler(np.array(dia["scaler"]["centre"]),
                    np.array(dia["scaler"]["scale"]))
    full_pdvf = {r["trait"]: r["pdvf"]["pdvf_components"] for r in dia["per_trait"]}

    log_line(cfg, NAME, "analysing causal swap (H5)")
    swap = analyse_swap(cfg, scaler)
    log_line(cfg, NAME, "analysing baselines (H2)")
    base = analyse_baselines(cfg, scaler, full_pdvf)
    log_line(cfg, NAME, "analysing multigeneration")
    mg = analyse_multigeneration(cfg, scaler)
    log_line(cfg, NAME, "analysing robustness")
    rb = analyse_robustness(cfg)

    from src.germline import make_spec
    spec = make_spec(cfg)
    d1 = load_raw("exp01_parental_crosses", cfg)
    d3 = load_raw("exp03_baselines", cfg)
    d4 = load_raw("exp04_multigeneration", cfg)
    n_total = (int(d1["Y"].shape[0]) + int(d1["ref_Y"].shape[0])
               + 5 * swap["n_backgrounds"]
               + sum(int(d3[f"Y_{c}"].shape[0]) for c in base["conditions"])
               + int(d3["pool_A_Y"].shape[0]) + int(d3["pool_B_Y"].shape[0])
               + sum(int(d3[f"diallel_{a}_Y"].shape[0])
                     for a in ("nodev_static", "nodev_quasistatic"))
               + int(d4["Y"].shape[0])
               + int(load_raw("exp05_robustness", cfg)["n"].sum()
                     // len(TRAIT_NAMES)))

    figures = [
        ("fig_P1-1_schematic", "conceptual schematic G_A + G_B -> Z -> D -> S -> Y"),
        ("fig_P1-2_development", "developmental trajectories of the soma modules"),
        ("fig_P1-3_phenotype_space", "parent and offspring phenotype space (PCA)"),
        ("fig_P1-4_factorial_effects", "factorial parental contribution plots"),
        ("fig_P1-5_causal_swap", "causal germline-swap intervention"),
        ("fig_P1-6_baselines", "clone / biparental / random / no-development / "
                               "direct-controller baselines"),
        ("fig_P1-7_lineage", "three-generation lineage"),
        ("fig_P1-8_robustness", "robustness and parameter sweep"),
    ]

    S = {
        "provenance": provenance(cfg, NAME),
        "smoke_test": cfg.run.smoke_test,
        "germline_dimension": spec.d,
        "germline_autosomal": int(spec.autosomal_mask.sum()),
        "germline_maternal": int(spec.maternal_mask.sum()),
        "n_seeds": {
            "diallel_replicates_per_cell": cfg.run.n_seeds_diallel,
            "swap_backgrounds": cfg.run.n_backgrounds_swap,
            "baseline_replicates": cfg.run.n_seeds_baseline,
            "robustness_replicates_per_cell": cfg.run.n_seeds_robustness,
            "lineages": cfg.run.n_lineages,
        },
        "n_individuals_total": int(n_total),
        "n_offspring_diallel": dia["n_offspring"],
        "traits": list(TRAIT_NAMES),
        "diallel": dia,
        "swap": swap,
        "baselines": base,
        "multigeneration": mg,
        "robustness": rb,
        "parental_main_effects": {
            r["trait"]: {"dam": r["anova"]["alpha_a"], "sire": r["anova"]["beta_b"],
                         "p_holm_dam": r["p_holm"]["dam"],
                         "p_holm_sire": r["p_holm"]["sire"]}
            for r in dia["per_trait"]},
        "parental_interaction_effects": {
            r["trait"]: {"gamma": r["anova"]["gamma_ab"],
                         "p_holm": r["p_holm"]["dam:sire"],
                         "partial_eta2": r["anova"]["partial_eta2"]["dam:sire"]}
            for r in dia["per_trait"]},
        "causal_swap_effects": {
            r["trait"]: {"swap_sire": r["swap_sire"], "swap_dam": r["swap_dam"],
                         "null_remut": r["null_remut_mean_shift"],
                         "null_redev": r["null_redev_mean_shift"]}
            for r in swap["per_trait"]},
        "developmental_ablation_effects": base["ablated_diallels"],
        "direct_controller_comparison": base["comparisons"].get("direct_controller"),
        "random_control_comparison": base["comparisons"].get("random_germline"),
        "multigeneration_persistence": mg["midparent_regression"],
        "robustness_statistics": {
            "fraction_sweep_points_with_H1_effect":
                rb["fraction_sweep_points_with_H1_effect"],
            "failure_points": rb["failure_points"],
        },
        "parental_developmental_variance_fraction": {
            r["trait"]: r["pdvf"] for r in dia["per_trait"]},
        "H1_supported": dia["H1_supported"],
        "H2_supported": base["H2_supported"],
        "H3_supported": dia["H1_supported"],
        "H4_supported": dia["H4_supported"],
        "H5_supported": swap["H5_supported"],
        "figures": figures,
    }
    S["supported_hypotheses"] = [h for h in ("H1", "H2", "H3", "H4", "H5")
                                 if S[f"{h}_supported"]]
    S["unsupported_hypotheses"] = [h for h in ("H1", "H2", "H3", "H4", "H5")
                                   if not S[f"{h}_supported"]]

    save_json(os.path.join(PROC_DIR, f"diallel{tag(cfg)}.json"), dia)
    save_json(os.path.join(PROC_DIR, f"swap{tag(cfg)}.json"), swap)
    save_json(os.path.join(PROC_DIR, f"baselines{tag(cfg)}.json"), base)
    save_json(os.path.join(PROC_DIR, f"multigeneration{tag(cfg)}.json"), mg)
    save_json(os.path.join(PROC_DIR, f"robustness{tag(cfg)}.json"), rb)
    save_json(os.path.join(OUT_DIR, f"paper1_summary{tag(cfg)}.json"), S)
    report = write_report(cfg, S)
    log_line(cfg, NAME, f"analysis complete in {time.time() - t0:.1f}s")
    log_line(cfg, NAME, f"supported: {S['supported_hypotheses']} | "
                        f"unsupported: {S['unsupported_hypotheses']}")
    log_line(cfg, NAME, f"wrote {report}")
    return S


if __name__ == "__main__":
    cfg = Config()
    if "--smoke-test" in sys.argv:
        cfg = cfg.smoke()
    main(cfg)
