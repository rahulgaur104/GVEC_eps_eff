#!/usr/bin/env python3

import numpy as np
from scipy.integrate import simpson

def compute_HI_pairs(
    *,
    B_pairs, Bzeta_pairs, grad_rho_pairs, kappa_g_pairs,   # (n_valid, Nq)
    w_pairs,                                               # (n_valid, Nq)  weights in zeta
    pair_idx,                                              # (n_valid, 3)
    pitch_inv,                                             # (nrho, Np)
    eps=1e-300,
):
    """
    Compute H and I for each valid well-row (each row is one (rho,pitch,well)).
    Returns H_pairs, I_pairs with shape (n_valid,).
    """
    #B_pairs = np.asarray(B_pairs, float)
    #Bzeta_pairs = np.asarray(Bzeta_pairs, float)
    #grad_rho_pairs = np.asarray(grad_rho_pairs, float)
    #kappa_g_pairs = np.asarray(kappa_g_pairs, float)
    #w_pairs = np.asarray(w_pairs, float)
    #pair_idx = np.asarray(pair_idx, int)
    #pitch_inv = np.asarray(pitch_inv, float)

    n_valid, Nq = B_pairs.shape

    # pull pitch_inv for each row using (irho, ipitch)
    irho = pair_idx[:, 0]
    ip   = pair_idx[:, 1]
    u = pitch_inv[irho, ip]             # (n_valid,)  u = pitch_inv = 1/lambda
    lam = 1.0 / u                       # (n_valid,)

    # dl/dzeta = |B| / B^zeta
    dl_dzeta = np.abs(B_pairs / (Bzeta_pairs + eps))   # (n_valid, Nq)

    # S = sqrt(|1 - lambda*B|)
    S = np.sqrt(np.abs(1.0 - lam[:, None] * B_pairs))  # (n_valid, Nq)

    # I integrand:  (dl) * S/B
    dI = w_pairs * dl_zeta * (S / (B_pairs + eps))      # (n_valid, Nq)
    I_pairs = np.sum(dI, axis=1)                       # (n_valid,)

    # H integrand: (dl) * S*(4/(lambda B)-1) * (|grad rho| kappa_g)/B
    factor = (4.0 / (lam[:, None] * B_pairs + eps) - 1.0)
    dH = w_pairs * dl_zeta * S * factor * (grad_rho_pairs * kappa_g_pairs) / (B_pairs + eps)
    H_pairs = np.sum(dH, axis=1)

    return H_pairs, I_pairs



def compute_L_rho(zeta_line, Bzeta_line, eps=1e-300):
    """
    zeta_line: (Nz,)
    Bzeta_line: (nrho, Nz)   (contravariant B^zeta along the field line)
    returns L_rho: (nrho,)
    """
    zeta_line = np.asarray(zeta_line, float)
    Bzeta_line = np.asarray(Bzeta_line, float)
    integrand = 1.0 / (np.abs(Bzeta_line) + eps)   # (nrho, Nz)
    return simpson(integrand, x=zeta_line, axis=-1)


def pitch_integral_simpson(S_rp, pitch_inv, L_rho, eps=1e-300):
    """
    S_rp:      (nrho, Np)
    pitch_inv: (nrho, Np)  (must be increasing along axis=-1 for best behavior)
    L_rho:     (nrho,)
    returns eps32_base: (nrho,)
    """
    #S_rp = np.asarray(S_rp, float)
    #pitch_inv = np.asarray(pitch_inv, float)
    #L_rho = np.asarray(L_rho, float)

    y = S_rp / (pitch_inv**3 + eps)   # (nrho, Np)
    num = simpson(y, x=pitch_inv, axis=-1)  # (nrho,)
    return num / (L_rho + eps)

def pitch_integral_eps32_base(S_rp, pitch_inv, pitch_w, L_rho, eps=1e-300):
    """
    Compute eps32_base[rho] = (1/L) * sum_p S_rp * pitch_w / pitch_inv^3
    Shapes:
      S_rp      : (nrho, Np)
      pitch_inv : (nrho, Np)  (or broadcastable)
      pitch_w   : (nrho, Np)  (or (Np,))
      L_rho     : (nrho,)
    """
    #S_rp = np.asarray(S_rp, float)
    #pitch_inv = np.asarray(pitch_inv, float)
    #pitch_w = np.asarray(pitch_w, float)
    #L_rho = np.asarray(L_rho, float)

    integrand = S_rp * pitch_w / (pitch_inv**3 + eps)   # (nrho,Np)
    num = np.sum(integrand, axis=1)                     # (nrho,)
    return num / (L_rho + eps)

