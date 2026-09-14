"""Germline structure, founder construction and recombination bookkeeping."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config, rng_for
from src.germline import (FOUNDER_NAMES, founder_germlines,
                          germline_distance_matrix, make_spec, random_germlines)
from src.zygote import dam_fraction, draw_background, recombine

CFG = Config().smoke()
SPEC = make_spec(CFG)


def test_locus_table_covers_every_coordinate():
    covered = np.zeros(SPEC.d, dtype=int)
    for l in SPEC.loci:
        covered[l.slice] += 1
    assert np.all(covered == 1)
    assert SPEC.maternal_mask.sum() + SPEC.autosomal_mask.sum() == SPEC.d


def test_denormalisation_respects_declared_bounds():
    u = np.linspace(-1, 1, SPEC.d)
    p = SPEC.denormalise(u)
    assert np.all(p >= SPEC.lo - 1e-12)
    assert np.all(p <= SPEC.hi + 1e-12)
    assert np.allclose(SPEC.denormalise(-np.ones(SPEC.d)), SPEC.lo)
    assert np.allclose(SPEC.denormalise(np.ones(SPEC.d)), SPEC.hi)


def test_founders_are_magnitude_matched_and_distinct():
    F = founder_germlines(SPEC, len(FOUNDER_NAMES))
    amp = CFG.germline.founder_amplitude
    assert np.allclose(np.abs(F), amp)
    D = germline_distance_matrix(F)
    assert np.allclose(np.diag(D), 0.0)
    off = D[~np.eye(len(F), dtype=bool)]
    assert off.min() > 0.5 * amp, "founders must be well separated"


def test_founder_construction_is_deterministic():
    a = founder_germlines(SPEC, 4)
    b = founder_germlines(SPEC, 4)
    assert np.array_equal(a, b)


def test_random_control_is_magnitude_matched():
    R = random_germlines(SPEC, 32, rng_for(CFG.master_seed, "t", "rand"))
    assert R.shape == (32, SPEC.d)
    assert np.allclose(np.abs(R), CFG.germline.founder_amplitude)


def test_maternal_channel_always_comes_from_the_dam():
    F = founder_germlines(SPEC, 4)
    bg = draw_background(64, SPEC, rng_for(CFG.master_seed, "t", "mat"))
    z = recombine(F[0], F[1], bg, SPEC)
    dam_gamete = np.clip(F[0] + bg.xi_dam, -1, 1)
    assert np.allclose(z[:, SPEC.maternal_mask], dam_gamete[:, SPEC.maternal_mask])


def test_recombination_takes_roughly_half_from_each_parent():
    F = founder_germlines(SPEC, 4)
    bg = draw_background(400, SPEC, rng_for(CFG.master_seed, "t", "half"))
    frac = dam_fraction(bg, SPEC).mean()
    assert 0.45 < frac < 0.58


def test_offspring_genome_differs_from_both_parents():
    F = founder_germlines(SPEC, 4)
    bg = draw_background(16, SPEC, rng_for(CFG.master_seed, "t", "mix"))
    z = recombine(F[0], F[1], bg, SPEC)
    for row in z:
        assert not np.allclose(row, F[0])
        assert not np.allclose(row, F[1])
