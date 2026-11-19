"""
test_determinism.py — T14: determinism checks for the variant runner.

  1. Same seed twice -> bit-identical output (for every graded entry point
     that is implemented).
  2. Different seeds -> different output (the seed is actually used).
  3. No hidden global randomness: perturbing the global `random` and
     `np.random` state between runs does not change the output, and the
     student files contain no calls to global RNG functions.

Run:  python test_determinism.py      (or: python -m unittest test_determinism)
"""

import ast
import json
import random
import unittest

import numpy as np

import simulator
from policy import make_policy

with open("variant_params.json") as _f:
    PARAMS = json.load(_f)
SEED = PARAMS["grading"]["seed"]
SIM_MS = PARAMS["grading"]["sim_time_ms"]
STUDENT_FILES = ["simulator.py", "policy.py", "experiments.py"]

# np.random attributes that do NOT touch the global legacy state
NP_RANDOM_OK = {"default_rng", "Generator", "SeedSequence", "PCG64",
                "Philox", "SFC64", "MT19937", "BitGenerator"}


def run_variant(seed):
    return simulator.run_variant_scenario(PARAMS, SIM_MS, seed,
                                          make_policy(PARAMS))


def entry_points():
    """(name, callable(seed)) for every implemented graded entry point."""
    eps = [("run_variant_scenario", run_variant)]
    for name in ("run_one_hop", "run_two_hop"):
        fn = getattr(simulator, name)
        try:
            fn(PARAMS, 100, SEED)
        except NotImplementedError:
            continue
        eps.append((name, lambda s, fn=fn: fn(PARAMS, SIM_MS, s)))
    return eps


class TestDeterminism(unittest.TestCase):

    def test_same_seed_identical(self):
        for name, fn in entry_points():
            with self.subTest(entry=name):
                self.assertEqual(fn(SEED), fn(SEED))

    def test_different_seed_differs(self):
        self.assertNotEqual(run_variant(SEED), run_variant(SEED + 1))

    def test_global_rng_state_irrelevant(self):
        random.seed(1); np.random.seed(1)
        a = run_variant(SEED)
        random.seed(999); np.random.seed(999)
        random.random(); np.random.random(100)
        b = run_variant(SEED)
        self.assertEqual(a, b)

    def test_no_global_random_calls_in_source(self):
        for path in STUDENT_FILES:
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read(), path)
            for node in ast.walk(tree):
                with self.subTest(file=path, line=getattr(node, "lineno", 0)):
                    # `import random` / `from random import ...`
                    if isinstance(node, ast.Import):
                        self.assertNotIn("random",
                                         [a.name for a in node.names])
                    if isinstance(node, ast.ImportFrom):
                        self.assertNotEqual(node.module, "random")
                        if node.module == "numpy.random":
                            for a in node.names:
                                self.assertIn(a.name, NP_RANDOM_OK)
                    # np.random.<fn>(...) / numpy.random.<fn>(...)
                    if (isinstance(node, ast.Attribute)
                            and isinstance(node.value, ast.Attribute)
                            and node.value.attr == "random"
                            and isinstance(node.value.value, ast.Name)
                            and node.value.value.id in ("np", "numpy")):
                        self.assertIn(node.attr, NP_RANDOM_OK)


if __name__ == "__main__":
    unittest.main(verbosity=2)
