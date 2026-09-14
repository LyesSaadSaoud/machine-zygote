"""Development, differentiation and the ablation modes."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config, rng_for
from src.development import develop, generic_soma_state, module_positions
from src.germline import founder_germlines, make_spec
from src.soma import differentiate, role_diversity
from src.zygote import draw_background, recombine

CFG = Config().smoke()
SPEC = make_spec(CFG)
F = founder_germlines(SPEC, 4)


def _zygotes(n=24, label="dev"):
    bg = draw_background(n, SPEC, rng_for(CFG.master_seed, "t", label))
    return recombine(F[0], F[1], bg, SPEC), bg


def test_generic_soma_is_identical_and_undifferentiated():
    U = generic_soma_state(5, SPEC.gcfg.n_modules, SPEC.gcfg.n_genes)
    assert np.all(U == 0.0)
    assert np.allclose(module_positions(SPEC.gcfg.n_modules)[0], -1.0)


def test_development_differentiates_initially_identical_modules():
    z, bg = _zygotes()
    res = develop(z, SPEC, CFG, bg.dev_seed)
    assert res.frozen.shape == (z.shape[0], SPEC.gcfg.n_modules, SPEC.gcfg.n_genes)
    assert np.all(np.isfinite(res.frozen))
    assert res.module_spread.mean() > 1e-3, "modules never differentiated"


def test_development_is_reproducible_from_the_seed_alone():
    z, bg = _zygotes(n=16, label="repro")
    a = develop(z, SPEC, CFG, bg.dev_seed).frozen
    b = develop(z, SPEC, CFG, bg.dev_seed).frozen
    assert np.array_equal(a, b)


def test_developmental_noise_seed_actually_matters():
    z, bg = _zygotes(n=16, label="noise")
    a = develop(z, SPEC, CFG, bg.dev_seed).frozen
    b = develop(z, SPEC, CFG, bg.dev_seed + 1).frozen
    assert not np.allclose(a, b)


def test_chunking_does_not_change_results():
    z, bg = _zygotes(n=20, label="chunk")
    full = develop(z, SPEC, CFG, bg.dev_seed).frozen
    halves = np.concatenate([
        develop(z[:7], SPEC, CFG, bg.dev_seed[:7]).frozen,
        develop(z[7:], SPEC, CFG, bg.dev_seed[7:]).frozen])
    assert np.allclose(full, halves, atol=1e-12)


def test_nodev_static_leaves_every_module_identical():
    z, bg = _zygotes(n=8, label="static")
    res = develop(z, SPEC, CFG, bg.dev_seed, mode="nodev_static")
    assert np.allclose(res.module_spread, 0.0)
    par = SPEC.unpack(z)
    soma = differentiate(res.frozen, par["fate_temp"], par["kappa_J"], CFG)
    assert np.allclose(role_diversity(soma), 1.0), \
        "with identical modules no distinct fates can exist"


def test_nodev_quasistatic_keeps_positional_information():
    z, bg = _zygotes(n=8, label="quasi")
    res = develop(z, SPEC, CFG, bg.dev_seed, mode="nodev_quasistatic")
    assert res.module_spread.mean() > 0.0


def test_fate_weights_are_a_probability_simplex():
    z, bg = _zygotes(n=8, label="fate")
    res = develop(z, SPEC, CFG, bg.dev_seed)
    par = SPEC.unpack(z)
    soma = differentiate(res.frozen, par["fate_temp"], par["kappa_J"], CFG)
    assert np.allclose(soma.pi.sum(axis=-1), 1.0)
    assert np.all(soma.pi >= 0)
    assert np.allclose(np.diagonal(soma.J, axis1=1, axis2=2), 0.0)
