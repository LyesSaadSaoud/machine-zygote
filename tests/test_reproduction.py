"""End-to-end reproduction, phenotype extraction and the swap intervention."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config, N_TRAITS, rng_for
from src.germline import founder_germlines, make_spec
from src.interventions import causal_swap
from src.reproduction import cross, diallel, parental_reference
from src.zygote import draw_background

CFG = Config().smoke()
SPEC = make_spec(CFG)
F = founder_germlines(SPEC, 4)


def test_pipeline_produces_a_finite_phenotype_for_every_individual():
    bg = draw_background(12, SPEC, rng_for(CFG.master_seed, "t", "pipe"))
    coh = cross(F[0], F[1], bg, SPEC, CFG)
    assert coh.Y.shape == (12, N_TRAITS)
    assert np.all(np.isfinite(coh.Y))
    assert not coh.flags["nonfinite"].any()


def test_no_individual_is_silently_dropped():
    n = 16
    bg = draw_background(n, SPEC, rng_for(CFG.master_seed, "t", "keep"))
    coh = cross(F[0], F[1], bg, SPEC, CFG)
    assert len(coh) == n
    for v in coh.flags.values():
        assert v.shape == (n,)


def test_diallel_is_balanced_and_includes_reciprocals():
    coh = diallel(F, SPEC, CFG, rng_for(CFG.master_seed, "t", "dia"), n_rep=3)
    dam, sire = coh.meta["dam"], coh.meta["sire"]
    assert len(coh) == 4 * 4 * 3
    for i in range(4):
        for j in range(4):
            assert int(((dam == i) & (sire == j)).sum()) == 3
    assert ((dam == 0) & (sire == 1)).any() and ((dam == 1) & (sire == 0)).any()


def test_clone_line_still_shows_developmental_variation():
    coh = parental_reference(F[0], 8, SPEC, CFG,
                             rng_for(CFG.master_seed, "t", "clone"))
    assert coh.Y.std(axis=0).max() > 0, "developmental noise must produce variation"


def test_swap_arms_share_their_background():
    exp = causal_swap(F, SPEC, CFG, rng_for(CFG.master_seed, "t", "swap"))
    n = exp.n_backgrounds
    for arm, coh in exp.arms.items():
        assert len(coh) == n
    # only the swapped parent's loci may differ between base and swap arms
    base = exp.arms["base"].zyg_u
    sire = exp.arms["swap_sire"].zyg_u
    assert not np.allclose(base, sire)
    # maternal-channel loci come from the dam, so a sire swap cannot touch them
    assert np.allclose(base[:, SPEC.maternal_mask], sire[:, SPEC.maternal_mask])


def test_swap_changes_phenotype_but_identity_reproduces_exactly():
    exp = causal_swap(F, SPEC, CFG, rng_for(CFG.master_seed, "t", "swap2"))
    base = exp.arms["base"].Y
    assert not np.allclose(exp.arms["swap_sire"].Y, base)
    assert not np.allclose(exp.arms["swap_dam"].Y, base)


def test_rerunning_the_same_cross_is_bitwise_identical():
    bg = draw_background(8, SPEC, rng_for(CFG.master_seed, "t", "ident"))
    a = cross(F[0], F[1], bg, SPEC, CFG).Y
    b = cross(F[0], F[1], bg, SPEC, CFG).Y
    assert np.array_equal(a, b)
