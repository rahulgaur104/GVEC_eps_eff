#!/usr/bin/env python3
import numpy as np

_BOUNCE_SENTINEL = -1e5  # only used elsewhere; here we use 0.0 padding like DESC


def bijection_from_disc_numpy(x, a, b):
    """
    Map x ∈ [-1, 1] to z ∈ [a, b] (affine map), broadcasting over a,b.

        z = (b-a)/2 * x + (a+b)/2
    """
    x = np.asarray(x)
    a = np.asarray(a)
    b = np.asarray(b)
    return 0.5 * (b - a)[..., None] * x + 0.5 * (a + b)[..., None]


def grad_bijection_from_disc_numpy(a, b):
    """
    Jacobian dz/dx for bijection_from_disc_numpy:
        dz/dx = (b-a)/2
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return 0.5 * (b - a)


def bounce_quad_to_zeta(
    z1, z2, x, w, *, invalid_fill=0.0, return_flat=False
):
    """
    Convert reference quadrature nodes/weights (x,w) on [-1,1] into ζ-nodes/weights
    on each bounce interval [z1,z2].

    If return_flat=False:
        returns full padded arrays (nrho, Np, num_well, Nq) for convenience.

    If return_flat=True (Option B fast path):
        returns only *valid* wells packed contiguously, avoiding building padded zeta_q.

    Parameters
    ----------
    z1, z2 : ndarray, shape (nrho, Np, num_well)
        Bounce point pairs in global ζ. Invalid pairs are padded as 0.0,0.0.
    x, w : ndarray, shape (Nq,)
        Quadrature nodes and weights for ∫_{-1}^1 f(x) dx.
    invalid_fill : float
        Fill value for ζ nodes in invalid wells (only used in full-output mode).
    return_flat : bool
        If True, return packed arrays for only-valid wells.

    Returns (return_flat=False)
    --------------------------
    zeta_q : (nrho, Np, num_well, Nq)
    w_zeta : (nrho, Np, num_well, Nq)
    pair_mask : (nrho, Np, num_well) bool

    Returns (return_flat=True)
    --------------------------
    pair_mask : (nrho, Np, num_well) bool
    pair_idx  : (n_valid, 3) int      rows are (irho, ipitch, iwell)
    zeta_pairs: (n_valid, Nq)
    w_pairs   : (n_valid, Nq)
    zeta_flat : (n_valid*Nq,)
    w_flat    : (n_valid*Nq,)
    """
    z1 = np.asarray(z1, float)
    z2 = np.asarray(z2, float)
    x = np.asarray(x, float)
    w = np.asarray(w, float)

    if x.ndim != 1 or w.ndim != 1 or x.shape != w.shape:
        raise ValueError("x and w must be 1D arrays with the same shape")
    if z1.shape != z2.shape or z1.ndim != 3:
        raise ValueError("z1 and z2 must both be shape (nrho, Np, num_well)")

    # Valid pairs: DESC pads invalid wells with z1=z2=0.0
    pair_mask = (z1 - z2) != 0.0

    # -------- Option B fast path: pack only valid wells (no padded allocation) --------
    if return_flat:
        pair_idx = np.argwhere(pair_mask)  # (n_valid, 3)
        if pair_idx.size == 0:
            # no wells anywhere
            n_valid = 0
            Nq = x.size
            empty_pairs = np.zeros((0, Nq), dtype=float)
            empty_flat = np.zeros((0,), dtype=float)
            return pair_mask, pair_idx, empty_pairs, empty_pairs, empty_flat, empty_flat

        a = z1[pair_mask]  # (n_valid,)
        b = z2[pair_mask]  # (n_valid,)

        half = 0.5 * (b - a)[:, None]   # (n_valid,1)
        mid  = 0.5 * (b + a)[:, None]   # (n_valid,1)

        zeta_pairs = half * x[None, :] + mid          # (n_valid,Nq)
        w_pairs    = half * w[None, :]                # (n_valid,Nq)

        zeta_flat = zeta_pairs.reshape(-1)
        w_flat = w_pairs.reshape(-1)

        return pair_mask, pair_idx, zeta_pairs, w_pairs, zeta_flat, w_flat

    # -------- Full padded output mode (convenient for downstream vectorized math) ----
    zeta_q = bijection_from_disc_numpy(x, z1, z2)  # (nrho, Np, num_well, Nq)
    cov = grad_bijection_from_disc_numpy(z1, z2)   # (nrho, Np, num_well)
    w_zeta = cov[..., None] * w                    # (nrho, Np, num_well, Nq)

    zeta_q = np.where(pair_mask[..., None], zeta_q, invalid_fill)
    w_zeta = np.where(pair_mask[..., None], w_zeta, 0.0)

    return zeta_q, w_zeta, pair_mask


def make_rho_flat(pair_idx, Nq, rho_nodes):
    """
    Build rho_flat aligned with zeta_flat, if your equilibrium call needs rho per point.

    rho_nodes: (nrho,)
    pair_idx:  (n_valid,3) with first col = irho
    returns:   (n_valid*Nq,)
    """
    rho_nodes = np.asarray(rho_nodes, float)
    rho_pair = rho_nodes[pair_idx[:, 0]]     # (n_valid,)
    return np.repeat(rho_pair, Nq)           # (n_valid*Nq,)


def scatter_pairs_to_padded(Q_pairs, pair_mask, *, fill_value=0.0):
    """
    Scatter a packed per-well quantity back to padded (nrho, Np, num_well, ...).

    Parameters
    ----------
    Q_pairs : (n_valid, Nq, ...)  OR (n_valid, ...)
        Packed quantity for valid wells.
    pair_mask : (nrho,Np,num_well) bool
    fill_value : float
        Value for invalid wells.

    Returns
    -------
    Q_full : (nrho, Np, num_well, Nq, ...) OR (nrho, Np, num_well, ...)
    """
    pair_mask = np.asarray(pair_mask, bool)
    n_valid = int(pair_mask.sum())

    Q_pairs = np.asarray(Q_pairs)
    if Q_pairs.shape[0] != n_valid:
        raise ValueError(f"Q_pairs has n_valid={Q_pairs.shape[0]}, expected {n_valid}")

    out_shape = pair_mask.shape + Q_pairs.shape[1:]
    Q_full = np.full(out_shape, fill_value, dtype=Q_pairs.dtype)
    Q_full[pair_mask] = Q_pairs
    return Q_full


