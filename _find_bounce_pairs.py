#!/usr/bin/env python3
"""
NumPy/SciPy port of DESC-style Hermite-spline bounce point finding.

Inputs:
  - zeta (Nz,), strictly increasing
  - B(rho, zeta) and dB/dzeta(rho, zeta)
  - pitch_inv(rho, pitch) = 1/lambda

Outputs:
  - z1, z2: (nrho, Np, num_well) with 0.0 padding for invalid/missing pairs.

Key DESC behavior preserved:
  - z1: intersections with dB/dzeta <= 0
  - z2: intersections with dB/dzeta >= 0, but only those that CLOSE a well that
        started *inside the snapshot* (ignore wells that started before snapshot).
"""

import numpy as np
from scipy.interpolate import CubicHermiteSpline

# Sentinel is just here to impose a lower limit beyond which the root finder won't
# look for intersections
_BOUNCE_SENTINEL = -1e5

def polyder_vec_desc(c: np.ndarray) -> np.ndarray:
    """Differentiate DESC-ordered polynomial coefficients [a_n,...,a_0]."""
    c = np.asarray(c)
    deg = c.shape[-1] - 1
    if deg <= 0:
        return np.zeros(c.shape[:-1] + (0,), dtype=c.dtype)
    powers = np.arange(deg, 0, -1, dtype=c.dtype)  # [deg,...,1]
    return c[..., :-1] * powers


def polyval_vec_desc(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Evaluate DESC-ordered polynomial coefficients [a_n,...,a_0] at x."""
    x = np.asarray(x)
    c = np.asarray(c)
    deg = c.shape[-1] - 1
    pows = np.arange(deg, -1, -1, dtype=x.dtype)
    return np.sum(c * (x[..., None] ** pows), axis=-1)


def _filter_distinct_sorted(r: np.ndarray, sentinel: float, eps: float) -> np.ndarray:
    """If adjacent sorted roots are ~equal, mark duplicates as sentinel."""
    dr = np.diff(r, axis=-1, prepend=sentinel)
    dup = np.isclose(dr, 0.0, atol=eps)
    return np.where(dup, sentinel, r)


def polyroot_vec_desc(
    c: np.ndarray,
    k: np.ndarray,
    a_min,
    a_max,
    *,
    sort: bool = True,
    sentinel: float = -1.0,
    eps: float = 2.5e-12,
    distinct: bool = True,
) -> np.ndarray:
    """
    Roots of p(x)=k for each polynomial in c (DESC ordering), filtered to [a_min,a_max].

    c shape: (..., n_poly, deg+1)
    k shape: (..., n_poly) or broadcastable
    returns: (..., n_poly, deg) padded with sentinel
    """
    c = np.asarray(c, float)
    deg = c.shape[-1] - 1
    if deg < 1:
        return np.full(c.shape[:-1] + (0,), sentinel, dtype=float)

    k = np.asarray(k, float)
    k_b = np.broadcast_to(k, c.shape[:-1])

    a_min = np.asarray(a_min, float)
    a_max = np.asarray(a_max, float)
    if a_min.size == 1:
        a_min = np.full((c.shape[-2],), float(a_min))
    if a_max.size == 1:
        a_max = np.full((c.shape[-2],), float(a_max))

    out = np.full(c.shape[:-1] + (deg,), sentinel, dtype=float)

    # Flatten everything except polynomial coeff axis
    c2 = c.reshape((-1, c.shape[-2], deg + 1))
    k2 = k_b.reshape((-1, c.shape[-2]))
    out2 = out.reshape((-1, c.shape[-2], deg))

    for bi in range(c2.shape[0]):
        for j in range(c2.shape[1]):
            coeff = c2[bi, j].copy()
            coeff[-1] -= k2[bi, j]  # p(x)-k

            r = np.roots(coeff)
            rr = r.real[np.abs(r.imag) <= eps]
            lo, hi = a_min[j], a_max[j]
            rr = rr[(rr >= lo - eps) & (rr <= hi + eps)]

            if rr.size == 0:
                continue

            if sort or distinct:
                rr = np.sort(rr)
            if distinct and rr.size > 1:
                rr = _filter_distinct_sorted(rr, sentinel=sentinel, eps=eps)

            tmp = np.full((deg,), sentinel, dtype=float)
            m = min(deg, rr.size)
            tmp[:m] = rr[:m]
            out2[bi, j, :] = tmp

    return out


def flatten_last2(a: np.ndarray) -> np.ndarray:
    """Flatten last 2 axes into 1."""
    a = np.asarray(a)
    return a.reshape(a.shape[:-2] + (a.shape[-2] * a.shape[-1],))


def take_mask_first_k(arr: np.ndarray, mask: np.ndarray, *, size: int, fill_value=_BOUNCE_SENTINEL) -> np.ndarray:
    """Take first K elements where mask True along last axis; pad with fill_value."""
    arr = np.asarray(arr)
    mask = np.asarray(mask, bool)
    if arr.shape != mask.shape:
        raise ValueError("arr and mask must have same shape")

    out = np.full(arr.shape[:-1] + (size,), fill_value, dtype=arr.dtype)

    # ranks: 0,1,2,... for True entries
    rank = np.cumsum(mask, axis=-1) - 1
    valid = mask & (rank >= 0) & (rank < size)

    out2 = out.reshape((-1, size))
    arr2 = arr.reshape((-1, arr.shape[-1]))
    rank2 = rank.reshape((-1, rank.shape[-1]))
    valid2 = valid.reshape((-1, valid.shape[-1]))

    rr, cc = np.nonzero(valid2)
    out2[rr, rank2[rr, cc]] = arr2[rr, cc]
    return out


def in_epigraph_and_desc(is_intersect: np.ndarray, df_dy_sign: np.ndarray) -> np.ndarray:
    """
    Filter intersections so z2 only includes exits from wells that START inside snapshot.

    State machine scan left->right:
      - dB<0 intersection: "enter well" (B crosses down through pitch_inv)
      - dB>0 intersection: "exit well" (B crosses up through pitch_inv)
          keep this as z2 ONLY if we've previously entered inside snapshot.
      - dB==0: ignore (tangency/degenerate)

    This is the *mechanism* behind DESC's comment:
      "ignore bounce points trapped outside this snapshot of the field line"
    """
    inter = np.asarray(is_intersect, bool)
    sgn = np.asarray(df_dy_sign)
    keep = np.zeros_like(inter, dtype=bool)

    inter2 = inter.reshape((-1, inter.shape[-1]))
    sgn2 = sgn.reshape((-1, sgn.shape[-1]))
    keep2 = keep.reshape((-1, keep.shape[-1]))

    for r in range(inter2.shape[0]):
        inside = False
        for i in range(inter2.shape[1]):
            if not inter2[r, i]:
                continue
            si = sgn2[r, i]
            if si < 0:
                inside = True
            elif si > 0:
                if inside:
                    keep2[r, i] = True
                    inside = False
            else:
                # tangency -> ignore
                pass
    return keep


def build_spline_and_bounce_points_rho(
    *,
    zeta: np.ndarray,      # (Nz,)
    B: np.ndarray,         # (nrho, Nz)
    B_z: np.ndarray,       # (nrho, Nz)
    pitch_inv: np.ndarray, # (nrho, Np)
    num_well: int,
    eps_root: float = 2.5e-12,
):
    """
    Build CubicHermiteSpline and compute bounce points (z1,z2) in global zeta.

    Returns
    -------
    z1, z2 : (nrho, Np, num_well)
        Bounce points. Missing/invalid pairs are set to 0.0 (DESC behavior).
    """
    zeta = np.asarray(zeta, float)
    B = np.asarray(B, float)
    B_z = np.asarray(B_z, float)
    pitch_inv = np.asarray(pitch_inv, float)

    if zeta.ndim != 1:
        raise ValueError("zeta must be 1D")
    if np.any(np.diff(zeta) <= 0):
        raise ValueError("zeta must be strictly increasing")
    if B.shape[-1] != zeta.size or B_z.shape != B.shape:
        raise ValueError("B and B_z must be (nrho, Nz) with Nz == len(zeta)")
    if pitch_inv.ndim != 2 or pitch_inv.shape[0] != B.shape[0]:
        raise ValueError("pitch_inv must be (nrho, Np) with same nrho as B")
    if np.min(zeta) <= _BOUNCE_SENTINEL:
        raise ValueError(f"min(zeta) must be > {_BOUNCE_SENTINEL}")

    nrho, Nz = B.shape
    Np = pitch_inv.shape[1]
    nseg = Nz - 1
    dz = np.diff(zeta)  # (nseg,)

    #Hermite spline in PPoly form, vectorized over rho
    spl = CubicHermiteSpline(zeta, B, B_z, axis=-1)

    # spl.c is (4, nseg, nrho); put into (nrho, nseg, 4) with DESC power order
    Bcoef = np.moveaxis(spl.c, (0, 1), (-1, -2))  # (nrho, nseg, 4)
    dBcoef = polyder_vec_desc(Bcoef)              # (nrho, nseg, 3)

    #Broadcast over pitch axis
    Bc = np.broadcast_to(Bcoef[:, None, :, :], (nrho, Np, nseg, 4))
    dBc = np.broadcast_to(dBcoef[:, None, :, :], (nrho, Np, nseg, 3))

    #Find intersections per interval in local coord t ∈ [0, dz_i]
    k = np.broadcast_to(pitch_inv[:, :, None], (nrho, Np, nseg))
    intersect = polyroot_vec_desc(
        Bc, k, a_min=0.0, a_max=dz, sort=True, sentinel=-1.0, eps=eps_root, distinct=True
    )  # (nrho, Np, nseg, 3)

    #Classify by sign(dB/dzeta) at each intersection
    dB_eval = polyval_vec_desc(intersect, dBc[..., None, :])  # (nrho, Np, nseg, 3)
    dB_sign = np.sign(dB_eval)

    mask = flatten_last2(intersect >= 0.0)   # valid roots
    dB_sign_f = flatten_last2(dB_sign)

    z1_mask = (dB_sign_f <= 0.0) & mask
    z2_mask = (dB_sign_f >= 0.0) & in_epigraph_and_desc(mask, dB_sign_f)

    left = zeta[:-1]  # (nseg,)
    intersect_global = intersect + left[None, None, :, None]
    intersect_global_f = flatten_last2(intersect_global)

    z1 = take_mask_first_k(intersect_global_f, z1_mask, size=num_well, fill_value=_BOUNCE_SENTINEL)
    z2 = take_mask_first_k(intersect_global_f, z2_mask, size=num_well, fill_value=_BOUNCE_SENTINEL)

    # remove invalid pairs -> 0.0 (DESC behavior)
    ok = (z1 > _BOUNCE_SENTINEL) & (z2 > _BOUNCE_SENTINEL)
    z1 = np.where(ok, z1, 0.0)
    z2 = np.where(ok, z2, 0.0)
    return z1, z2




