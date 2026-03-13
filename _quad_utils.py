#!/usr/bin/env python3
# Fundamental quadrature functions
# Each polynomial basis has it's own set of points x and weights w

# Without using any transformation, a quadrauture can be written as
# ∫_{-1}^1 f(x) dx =  Σ w[i] * f(x[i])

# However, for bounce integrals, the integrand becomes singular near
# the bounce points -1 and 1. Therefore, we use the automorphism idea
# to cluster the points x near the singularities to resolve it while
# actually avoiding the singular point and at the same time,
# scaling the weights w appropriately.

# The automorphism is like a scaling function
# These functions are adapted from the DESC repo and are based on
# the work in Unalmis et al. 2024 (submitted to JPP)

import numpy as np
from numpy.polynomial.legendre import leggauss as _leggauss

def gauss_legendre(n):
    """
    Standard Gauss-Legendre on [-1, 1].

    Approximates ∫_{-1}^1 f(x) dx ≈ Σ w[i] * f(x[i]).
    """
    x, w = _leggauss(n)
    return x, w


def gauss_chebyshev_1(n):
    """
    Gauss-Chebyshev of the first kind with *implicit* weight.

    This is set up so that for integrands of the form
        f(x) = g(x) / sqrt(1 - x**2),
    we have
        ∫_{-1}^1 f(x) dx ≈ Σ w[i] * f(x[i])
    with
        x[i] = cos((2i-1)π / (2n))
        w[i] = (π/n) * sqrt(1 - x[i]**2).

    i.e. we absorbed the Chebyshev weight into the effective weights,
    matching the behavior of chebgauss1 in the DESC code.
    """
    k = np.arange(1, n + 1)
    x = np.cos((2 * k - 1) * np.pi / (2 * n))
    w = (np.pi / n) * np.sqrt(1.0 - x**2)
    return x, w


def gauss_chebyshev_2(n):
    """
    Gauss-Chebyshev of the second kind with *implicit* weight.

    This is set up so that for integrands of the form
        f(x) = g(x) * sqrt(1 - x**2),
    we have
        ∫_{-1}^1 f(x) dx ≈ Σ w[i] * f(x[i])
    with
        t[i] = iπ / (n+1),  i = 1..n
        x[i] = cos(t[i])
        w[i] = (π/(n+1)) * sin(t[i])

    which is the DESC-style chebgauss2 effective rule.
    """
    t = np.arange(n, 0, -1) * np.pi / (n + 1)
    x = np.cos(t)
    w = (np.pi / (n + 1)) * np.sin(t)
    return x, w


# These automorphisms assume that x ∈ [-1, 1]

def automorphism_sin(x, m=10):
    """
    Sin automorphism: [-1, 1] -> [-1, 1]

        y = sin(π x / 2)

    Used to cluster nodes near ±1, ideal for bounce-type 1/sqrt singularities
    at the endpoints (this is the 'GL & sin' in the paper).

    m controls a small epsilon buffer to avoid hitting exactly ±1 due to fp error.
    """
    y = np.sin(0.5 * np.pi * x)
    eps = m * np.finfo(float).eps
    return np.clip(y, -1.0 + eps, 1.0 - eps)


def grad_automorphism_sin(x):
    """
    dy/dx for automorphism_sin.
    """
    return 0.5 * np.pi * np.cos(0.5 * np.pi * x)


def automorphism_arcsin(x, gamma=np.cos(0.5)):
    """
    Arcsin (Kosloff–Tal-Ezer) automorphism: [-1,1] -> [-1,1]

        y = arcsin(gamma x) / arcsin(gamma)

    Makes nodes more uniform (almost-equispaced) while preserving spectral convergence.
    Included for completeness; not strictly required if you just want GL & sin.
    """
    return np.arcsin(gamma * x) / np.arcsin(gamma)


def grad_automorphism_arcsin(x, gamma=np.cos(0.5)):
    """
    dy/dx for automorphism_arcsin.
    """
    return gamma / np.arcsin(gamma) / np.sqrt(1.0 - (gamma * x) ** 2)



def apply_automorphism(x, w, auto=None, dauto=None):
    """
    Given a base quadrature (x, w) on [-1,1] for ∫_{-1}^1 g(z) dz,
    return the transformed rule for z = auto(ξ):

        ∫_{-1}^1 g(z) dz = ∫_{-1}^1 g(auto(ξ)) auto'(ξ) dξ
                         ≈ Σ w'[i] * g(z'[i])

    where
        z'[i] = auto(x[i]),
        w'[i] = w[i] * auto'(x[i]).

    If auto is None, (x, w) are returned unchanged.
    """
    if auto is None:
        return x, w
    if dauto is None:
        raise ValueError("Must supply derivative dauto when using an automorphism.")
    x_new = auto(x)
    w_new = w * dauto(x)
    return x_new, w_new


# Final functions to call

def gl_sin(n):
    """
    Gauss-Legendre composed with sin automorphism ('GL & sin').

    This is the main high-order rule used in the paper for weakly singular
    bounce integrals: start with GL on [-1,1], then apply z = sin(π ξ / 2).
    """
    x, w = gauss_legendre(n)
    return apply_automorphism(x, w, automorphism_sin, grad_automorphism_sin)


def gc1(n):
    """Alias for implicit Gauss-Chebyshev type-1 rule."""
    return gauss_chebyshev_1(n)


def gc2(n):
    """Alias for implicit Gauss-Chebyshev type-2 rule."""
    return gauss_chebyshev_2(n)


##############################################################################
##########-----------ELLIPTIC INTEGRAL TEST FROM THE PAPER----------##########
##############################################################################


from scipy import special
import pytest


def _map_interval(x, a, b):
    """Affine map from [-1, 1] to [a, b]."""
    return 0.5 * (b - a) * x + 0.5 * (b + a)


def strong_integrand(z, k):
    """Strongly singular bounce integrand: 1 / sqrt(k^2 - sin^2 z)."""
    return 1.0 / np.sqrt(k**2 - np.sin(z) ** 2)


def weak_integrand(z, k):
    """Weak (regular) bounce integrand: sqrt(k^2 - sin^2 z)."""
    return np.sqrt(k**2 - np.sin(z) ** 2)


def strong_exact(k):
    """I_s(k) = ∫_{-asin k}^{asin k} dz / sqrt(k^2 - sin^2 z) = 2 K(k)."""
    m = k**2
    return 2.0 * special.ellipk(m)


def weak_exact(k):
    """I_w(k) = ∫_{-asin k}^{asin k} sqrt(k^2 - sin^2 z) dz
               = 2 [E(k) - (1 - k^2) K(k)]."""
    m = k**2
    K = special.ellipk(m)
    E = special.ellipe(m)
    return 2.0 * (E - (1.0 - m) * K)


def quad_strong_gl_sin(k, N):
    """GL & sin quadrature for the strong integrand on [-asin k, asin k]."""
    a, b = -np.arcsin(k), np.arcsin(k)
    x, w = gl_sin(N)              # nodes/weights on [-1, 1]
    z = _map_interval(x, a, b)    # map to [a, b]
    w_scaled = w * (b - a) / 2.0
    return np.sum(w_scaled * strong_integrand(z, k))


def quad_weak_gl_sin(k, N):
    """GL & sin quadrature for the weak integrand on [-asin k, asin k]."""
    a, b = -np.arcsin(k), np.arcsin(k)
    x, w = gl_sin(N)
    z = _map_interval(x, a, b)
    w_scaled = w * (b - a) / 2.0
    return np.sum(w_scaled * weak_integrand(z, k))


@pytest.mark.parametrize("k", [0.25, 0.5, 0.8, 0.99])
@pytest.mark.parametrize("N", [16, 32, 64])
def test_gl_sin_strong_elliptic(k, N):
    """GL & sin should resolve the strong (1/sqrt) elliptic bounce integral."""
    numerical = quad_strong_gl_sin(k, N)
    exact = strong_exact(k)
    rel_err = abs(numerical - exact) / exact
    # Loose enough to be robust, tight enough to flag real regressions:
    assert rel_err < 1e-8


@pytest.mark.parametrize("k", [0.25, 0.5, 0.8, 0.99])
@pytest.mark.parametrize("N", [16, 32, 64])
def test_gl_sin_weak_elliptic(k, N):
    """GL & sin should resolve the weak (sqrt) elliptic bounce integral."""
    numerical = quad_weak_gl_sin(k, N)
    exact = weak_exact(k)
    rel_err = abs(numerical - exact) / exact
    assert rel_err < 1e-10




