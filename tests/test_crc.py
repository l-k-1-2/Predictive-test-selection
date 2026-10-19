import os
import sys
from fractions import Fraction

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from crc import crc_grid, crc_threshold, unit_loss

ALPHAS = [0.01, 0.05, 0.2]


def random_cycles(n, rng, max_failed=10):
    return [rng.random(rng.integers(1, max_failed + 1)) for _ in range(n)]


def brute_threshold(cycles, alpha, kind):
    # exact arithmetic: loss counts are multiples of 1/2520, which is divisible by every cycle size up to 10
    a, n = Fraction(str(alpha)), len(cycles)
    candidates = sorted(set(np.concatenate(cycles))) + [np.inf]
    best = -np.inf
    for tau in candidates:
        if kind == "frac":
            total = sum(int((f < tau).sum()) * (2520 // f.size) for f in cycles)
        else:
            total = 2520 * sum(bool((f < tau).all()) for f in cycles)
        if Fraction(total + 2520, 2520 * (n + 1)) <= a:
            best = tau
    return best


@pytest.mark.parametrize("kind", ["frac", "any"])
@pytest.mark.parametrize("alpha", ALPHAS)
def test_brute_force(alpha, kind):
    cycles = random_cycles(200, np.random.default_rng(0))
    assert crc_threshold(cycles, alpha, kind) == brute_threshold(cycles, alpha, kind)


@pytest.mark.parametrize("kind", ["frac", "any"])
@pytest.mark.parametrize("alpha", ALPHAS)
def test_brute_force_several_seeds(alpha, kind):
    for seed in range(1, 6):
        cycles = random_cycles(150, np.random.default_rng(seed))
        assert crc_threshold(cycles, alpha, kind) == brute_threshold(cycles, alpha, kind)


@pytest.mark.parametrize("kind", ["frac", "any"])
@pytest.mark.parametrize("alpha", ALPHAS)
def test_minus_inf_below_one_over_alpha_minus_one(alpha, kind):
    n = int(np.ceil(1 / alpha - 1)) - 1
    assert n < 1 / alpha - 1
    assert crc_threshold(random_cycles(n, np.random.default_rng(0)), alpha, kind) == -np.inf


@pytest.mark.parametrize("kind", ["frac", "any"])
@pytest.mark.parametrize("alpha", ALPHAS)
def test_finite_at_exactly_one_over_alpha_minus_one(alpha, kind):
    n = int(round(1 / alpha - 1))
    assert crc_threshold(random_cycles(n, np.random.default_rng(0)), alpha, kind) > -np.inf


@pytest.mark.parametrize("kind", ["frac", "any"])
def test_plus_inf_when_alpha_at_least_one(kind):
    cycles = random_cycles(30, np.random.default_rng(0))
    assert crc_threshold(cycles, 1.0, kind) == np.inf
    assert crc_threshold(cycles, 1.5, kind) == np.inf


@pytest.mark.parametrize("kind", ["frac", "any"])
@pytest.mark.parametrize("alpha", [0.07, 0.13])
def test_grid_agrees_with_threshold(alpha, kind):
    # alpha values with a factor 25 in the denominator keep sums of 1/k from hitting the boundary exactly
    for seed in range(5):
        cycles = random_cycles(40, np.random.default_rng(seed))
        taus = np.concatenate([[np.inf], np.sort(np.concatenate(cycles))[::-1], [-np.inf]])
        lambdas = -taus
        L = np.array([[unit_loss(f, t, kind) for t in taus] for f in cycles])
        assert (np.diff(L, axis=1) <= 0).all()
        assert -crc_grid(L, lambdas, alpha) == crc_threshold(cycles, alpha, kind)
