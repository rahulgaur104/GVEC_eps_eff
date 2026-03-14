#!/usr/bin/env python3

import os

import pdb
import gvec
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from _bounce_points import *
from _find_bounce_pairs import *
from _quad_utils import *
from ripple_calculation import *

# load equilibrium
state = gvec.find_state(os.path.dirname(os.getcwd()))

rho_array = np.linspace(0.5, 1.0, 4)
ripple_array = np.zeros_like(rho_array)

for i, rho0 in enumerate(rho_array):
    #rho0 = 0.2
    rho_eval = np.array([rho0], dtype=float)  # only for interfaces that require a rho axis
    
    alpha = np.linspace(0.0, 2.0 * np.pi, 1, endpoint=False)  # field-line label
    Nz = 769
    zeta_B = np.linspace(-7.0 * np.pi, 7.0 * np.pi, Nz)
    
    r_major = float(state.evaluate("r_major").r_major.data)
    
    B0 = float(
        np.max(
            state.evaluate(
                "mod_B",
                rho=rho_eval,
                theta=np.linspace(0.0, 2.0 * np.pi, 100),
                zeta =np.linspace(0.0, 2.0 * np.pi, 100),
            ).mod_B.data
        )
    )
    
    # rotational transform on the desired surface
    iota = float(state.evaluate("iota", rho=rho_eval, theta=None, zeta=None).iota.data[0])
    
    # Boozer field-line coordinates
    theta_B = alpha[None, :, None] + iota * zeta_B[None, None, :]
    
    ev = gvec.EvaluationsBoozer(
        rho=rho_eval,
        theta_B=theta_B,
        zeta_B=zeta_B,
        state=state,
        # MNfactor=5,
    )
    
    ev["alpha"] = ("pol", alpha)
    ev["alpha"].attrs = dict(symbol=r"\alpha", long_name="fieldline label")
    ev = ev.set_coords("alpha").set_xindex("alpha")
    
    # quantities on the field-aligned grid
    state.compute(
        ev,
        "B",
        "dmod_B_dz_B",
        "mod_B",
        "kappa_G",
        "mod_grad_rho",
        "B_zeta_B",
        "B_contra_z_B",
    )
    
    # use integration points for flux-surface-averaged quantities
    ev_int = state.evaluate("mod_grad_rho", "Jac", "pos", rho=rho_eval, theta="int", zeta="int")
    
    grad_rho_avg = gvec.fluxsurface_integral(ev_int.mod_grad_rho * ev_int.Jac) / gvec.fluxsurface_integral(ev_int.Jac)
    grad_rho_avg = float(np.asarray(grad_rho_avg).squeeze())
 
    R0 = gvec.fluxsurface_integral(np.sqrt(ev_int.pos[0] ** 2 + ev_int.pos[1] ** 2) * ev_int.Jac) / gvec.fluxsurface_integral(ev_int.Jac)

    #print(r_major, R0.data)

    # Collapse to the rho of interest
    ev = ev.sel(rho=rho0)
    
    B_line = np.asarray(ev.mod_B.data).squeeze()
    B_z_line = np.asarray(ev.dmod_B_dz_B.data).squeeze()
    B_sup_zeta_line = np.asarray(ev.B_contra_z_B.data).squeeze()
    mod_grad_rho_line = np.asarray(ev.mod_grad_rho.data).squeeze()   # actually |grad rho|
    kappa_G_line = np.asarray(ev.kappa_G.data).squeeze()
    
    # helper routines still expect an explicit rho axis
    B = B_line[None, :]
    B_z = B_z_line[None, :]
    B_sup_zeta = B_sup_zeta_line[None, :]
    mod_grad_rho = mod_grad_rho_line[None, :]
    kappa_G = kappa_G_line[None, :]
    
    
    # pitch angle grid construction
    Np = 27
    mn = float(B_line.min())
    mx = float(B_line.max())
    margin = 0.03 * (mx - mn)
    
    # shape (1, Np) because helper expects an explicit rho axis
    pitch_inv = np.linspace(mn + margin, mx - margin, Np)[None, :]
    
    num_well = 25
    
    z1, z2 = build_spline_and_bounce_points_rho(
        zeta=zeta_B,
        B=B,
        B_z=B_z,
        pitch_inv=pitch_inv,
        num_well=num_well,
    )
    
    rho_nodes = np.array([rho0], dtype=float)
    
    x, w = gl_sin(11)
    
    # pair_idx rows are [rho_idx, pitch_idx, well_idx]
    pair_mask, pair_idx, zeta_pairs, w_pairs, zeta_flat, w_flat = bounce_quad_to_zeta(
        z1,
        z2,
        x,
        w,
        return_flat=True,
    )
    
    rho_flat = make_rho_flat(pair_idx, Nq=len(x), rho_nodes=rho_nodes)
    zeta_full = scatter_pairs_to_padded(zeta_pairs, pair_mask, fill_value=0.0)
    
    
    # Interpolate geometric quantities to quadrature points
    
    B_pairs = np.interp(zeta_pairs, zeta_B, B_line)
    Bzeta_pairs = np.interp(zeta_pairs, zeta_B, B_sup_zeta_line)
    grad_rho_pairs = np.interp(zeta_pairs, zeta_B, mod_grad_rho_line)
    kappa_g_pairs = np.interp(zeta_pairs, zeta_B, kappa_G_line)
    
    H_pairs, I_pairs = compute_HI_pairs(
        B_pairs=B_pairs,
        Bzeta_pairs=Bzeta_pairs,
        grad_rho_pairs=grad_rho_pairs,
        kappa_g_pairs=kappa_g_pairs,
        w_pairs=w_pairs,
        pair_idx=pair_idx,
        pitch_inv=pitch_inv,
    )
    
    # sum over all the wells for each pitch angle
    S_rp = np.zeros((1, Np), dtype=float)
    
    # each row of H_pairs/I_pairs is one valid well
    # pair_idx[:, 0:2] tells you which (rho, pitch) it belongs to
    contrib = np.where(I_pairs != 0.0, H_pairs**2 / I_pairs, 0.0)
    np.add.at(S_rp, (pair_idx[:, 0], pair_idx[:, 1]), contrib)
    
    L_rho = compute_L_rho(zeta_B, B_sup_zeta)
    
    eps32_base = pitch_integral_simpson(S_rp, pitch_inv, L_rho)
    
    eps32 = eps32_base * (B0 * R0 / grad_rho_avg) ** 2 * (np.pi / (8.0 * np.sqrt(2.0)))
    eps_eff = eps32 ** (2.0 / 3.0)
    ripple_array[i] = eps_eff.item()

    #print("S_rp =", S_rp)
    #print("L_rho =", L_rho)
    #print("eps32_base =", eps32_base)
    #print("effective ripple 3/2 =", eps32)
    print("effective ripple =", eps_eff.data)
    
    
    plot_bounce_pairs=False
    
    if plot_bounce_pairs:
        # ---- plotting ----
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.plot(zeta_B, B[0], lw=2)
        #ax.plot(phi, B_z[0]+50, '-k', lw=0.5)
        ax.set_xlabel(r"$\zeta$")
        ax.set_ylabel(r"$|B(\zeta)|$")
        ax.set_title(r"Bounce points on a finite snapshot: $|B(\zeta)|$ and intersections with pitch\_inv")
        
        pad = 0.05 * (mx - mn)
        ax.set_ylim(mn - pad, mx + pad)
        
        for p in range(Np):
            pv = pitch_inv[0, p]
            ax.axhline(pv, ls="--", lw=1, alpha=0.35)
            for w in range(num_well):
                a = z1[0, p, w]
                b = z2[0, p, w]
                if a == 0.0 and b == 0.0:
                    continue
                ax.plot([a, b], [pv, pv], lw=3, alpha=0.85)
                ax.scatter([a], [pv], s=45, marker="o")  # z1
                ax.scatter([b], [pv], s=55, marker="x")  # z2
        
        fig.tight_layout()
        fig.savefig("bounce_points_debug.png" , dpi=400)
 
plt.plot(rho_array, ripple_array, "-or")
plt.xlabel(r"$\rho$", fontsize=20)
plt.ylabel(r"$\epsilon_{\mathrm{eff}}$", fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig("ripple.png", dpi=400)

np.savez("ripple_data.npz", ripple=ripple_array, rho=rho_array)
