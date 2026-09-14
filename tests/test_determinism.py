"""Determinism, seed derivation, and the absence of learning."""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config, derive_seed, rng_for
from src.embodiment import simulate_twins
from src.development import develop
from src.germline import founder_germlines, make_spec
from src.reproduction import cross, evaluate_zygotes
from src.soma import differentiate, to_vector
from src.zygote import draw_background, recombine

CFG = Config().smoke()
SPEC = make_spec(CFG)
F = founder_germlines(SPEC, 4)


def test_seed_derivation_is_stable_and_label_sensitive():
    assert derive_seed(7, "a", 1) == derive_seed(7, "a", 1)
    assert derive_seed(7, "a", 1) != derive_seed(7, "a", 2)
    assert derive_seed(7, "a") != derive_seed(8, "a")
    # stable across processes (not Python's randomised hash)
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r);"
         "from src.config import derive_seed; print(derive_seed(7,'a',1))"
         % os.path.dirname(os.path.dirname(os.path.abspath(__file__)))],
        capture_output=True, text=True)
    assert int(out.stdout.strip()) == derive_seed(7, "a", 1)


def test_full_pipeline_is_bitwise_reproducible():
    bg = draw_background(10, SPEC, rng_for(CFG.master_seed, "t", "det"))
    a = cross(F[0], F[2], bg, SPEC, CFG)
    b = cross(F[0], F[2], bg, SPEC, CFG)
    assert np.array_equal(a.Y, b.Y)
    assert np.array_equal(a.frozen, b.frozen)


def test_chunk_size_does_not_change_any_phenotype():
    from dataclasses import replace
    bg = draw_background(20, SPEC, rng_for(CFG.master_seed, "t", "chunk2"))
    z = recombine(F[0], F[1], bg, SPEC)
    big = evaluate_zygotes(z, bg.dev_seed, SPEC, CFG).Y
    small = evaluate_zygotes(
        z, bg.dev_seed, SPEC,
        replace(CFG, run=replace(CFG.run, chunk_size=3))).Y
    assert np.allclose(big, small, atol=1e-12)


def test_no_learning_soma_is_unchanged_by_evaluation():
    """The controller must be numerically identical before and after the
    newborn is evaluated: no plasticity, no adaptation of parameters."""
    bg = draw_background(6, SPEC, rng_for(CFG.master_seed, "t", "nolearn"))
    z = recombine(F[0], F[1], bg, SPEC)
    par = SPEC.unpack(z)
    dev = develop(z, SPEC, CFG, bg.dev_seed)
    soma = differentiate(dev.frozen, par["fate_temp"], par["kappa_J"], CFG)
    before = to_vector(soma).copy()
    simulate_twins(soma, CFG)
    after = to_vector(soma)
    assert np.array_equal(before, after), "the newborn controller was modified"


def test_evaluation_initial_condition_is_identical_for_everyone():
    from src.embodiment import initial_state
    u1, a1, p1 = initial_state(4, SPEC.gcfg.n_modules, CFG)
    assert np.allclose(u1 - u1[0][None, :], 0.0)
    assert np.allclose(a1, 0.0) and np.allclose(p1, 0.0)


def test_no_learning_identifiers_in_the_evaluation_path():
    """Static audit of the AST: the evaluation path defines no update rule.

    Identifiers are inspected rather than raw text, so prose in the docstrings
    (which explicitly says there is no reward and no learning) cannot mask a
    real one.
    """
    import ast
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    banned = {"learning_rate", "lr", "optimizer", "optim", "gradient", "grad",
              "reward", "backward", "train", "fit", "update_weights", "loss"}
    for module in ("embodiment.py", "phenotype.py", "soma.py"):
        tree = ast.parse(open(os.path.join(root, "src", module)).read())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.FunctionDef):
                names.add(node.name)
                names.update(a.arg for a in node.args.args)
        clash = names & banned
        assert not clash, f"learning-like identifier(s) {clash} in {module}"
