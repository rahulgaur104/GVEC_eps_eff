# GVEC_eps_eff

Effective ripple calculation for a GVEC equilibrium.

This repository contains a small set of helper routines plus a main driver script for computing the effective ripple
\(\epsilon_\mathrm{eff}\) from a GVEC equilibrium using bounce-point detection, quadrature over trapped wells, and
pitch-angle integration.

## Repository structure

- `test_ripple_calculation.py`  
  Main end-to-end script. This is the entry point for the full effective ripple calculation.

- `ripple_calculation.py`  
  Final numerical reductions for the ripple calculation:
  - per-well \(H\) and \(I\)
  - field-line length \(L_\rho\)
  - pitch-angle integral for \(\epsilon_\mathrm{eff}^{3/2}\)

- `_find_bounce_pairs.py`  
  Bounce-point finder based on a DESC-style Hermite spline approach.

- `_bounce_points.py`  
  Utilities for mapping quadrature nodes from the reference interval to bounce intervals.

- `_quad_utils.py`  
  Quadrature rules used in the bounce integrals.

- `test_bounce_points.py`  
  Analytic/debug script for validating bounce-point detection on a synthetic magnetic field.

## What this code does

The main workflow is:

1. Load a GVEC equilibrium state.
2. Evaluate magnetic-field quantities along Boozer field lines.
3. Find bounce-point pairs for a set of pitch values.
4. Build quadrature nodes and weights on each trapped well.
5. Compute the per-well quantities \(H\) and \(I\).
6. Sum over wells and integrate over pitch.
7. Apply the geometric prefactor to obtain \(\epsilon_\mathrm{eff}\).

## Requirements

This code depends on:

- Python 3.10+
- NumPy
- SciPy
- xarray
- matplotlib
- `gvec` (must already be installed and available in your environment)

