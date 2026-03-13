#!/usr/bin/env python3

import gvec
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from _bounce_points import *
from _find_bounce_pairs import *
from _quad_utils import *
from ripple_calculation import *
import os
import pdb

plot_bounce_pairs=False

state = gvec.find_state(os.path.dirname(os.getcwd()))

rho = [0.1]  # radial positions
nrho = len(rho)
alpha = np.linspace(
            0, 2 * np.pi, 1, endpoint=False
            )  # fieldline label

Nz = 769
zeta_B = np.linspace(-7.0 * np.pi, 7.0 * np.pi, Nz)

R0 = state.evaluate("r_major").r_major.data
B0 = np.max(state.evaluate("mod_B", rho=rho, theta=np.linspace(0,2*np.pi, 100) , zeta=zeta_B).mod_B.data)

# evaluate the rotational transform (fieldline angle) on the desired surfaces
iota = state.evaluate("iota", rho=rho, theta=None, zeta=None).iota

# 3D toroidal and poloidal arrays that correspond to fieldline coordinates for each surface
theta_B = (
            alpha[None, :, None]
                + iota.data[:, None, None] * zeta_B[None, None, :]
                )

# create the grid
ev = gvec.EvaluationsBoozer(
            rho=rho, theta_B=theta_B, zeta_B=zeta_B, state=state,# MNfactor=5
            )
# set the fiedline label as poloidal coordinate & index (not necessary, but good practice)
ev["alpha"] = ("pol", alpha)
ev["alpha"].attrs = dict(
            symbol=r"\alpha", long_name="fieldline label"
            )
ev = ev.set_coords("alpha").set_xindex("alpha")

# First we calculate mod_B, dB_dz (assuming derivative is in Boozer zeta)
state.compute(ev, "B", "dmod_B_dz", "mod_B", "kappa_G", "mod_grad_rho", "B_zeta_B", "B_contra_z_B")


# use integration points for computing integral quantities
ev_int= state.evaluate("grad_rho","Jac",rho=rho,theta="int",zeta="int")
# add a new quantity
ev_int["mod_grad_rho"]= xr.dot(ev_int.grad_rho, ev_int.grad_rho, dim="xyz")

# flux surface average example = int(|mod_grad_rho|*Jac)/int(Jac)
grad_rho_avg = gvec.fluxsurface_integral(ev_int.mod_grad_rho*ev_int.Jac) / gvec.fluxsurface_integral(ev_int.Jac)


# reduces all the quantities to 2D
ev = ev.sel(rho=rho[0])

B = ev.mod_B.data
B_z = ev.dmod_B_dz.data
B_sup_zeta = ev.B_contra_z_B.data
mod_grad_rho = ev.mod_grad_rho.data # actually |grad rho|
kappa_G = ev.kappa_G.data


# pitch_inv in (minB,maxB)
Np = 31
mn, mx = float(B.min()), float(B.max())
margin = 0.01 * (mx - mn)
pitch_inv = np.linspace(mn + margin, mx - margin, Np)[None, :]

num_well = 25
z1, z2 = build_spline_and_bounce_points_rho(
    zeta=zeta_B,
    B=B,
    B_z=B_z,
    pitch_inv=pitch_inv,
    num_well=num_well,
)

rho_nodes = np.array([1.0])
x, w = gl_sin(11)

# zeta_pairs shape (, Nq)
# pair_idx shape [rho_idx, pitch_idx, well_idx]
pair_mask, pair_idx, zeta_pairs, w_pairs, zeta_flat, w_flat = bounce_quad_to_zeta(z1, z2, x, w, return_flat=True)
rho_flat = make_rho_flat(pair_idx, Nq=len(x), rho_nodes=rho_nodes)
zeta_full  = scatter_pairs_to_padded(zeta_pairs, pair_mask, fill_value=0.0)


B_pairs = np.interp(zeta_pairs, zeta_B, B[0])
Bzeta_pairs = np.interp(zeta_pairs, zeta_B, B_sup_zeta[0])
grad_rho_pairs = np.interp(zeta_pairs, zeta_B, mod_grad_rho[0])
kappa_g_pairs = np.interp(zeta_pairs, zeta_B, kappa_G[0])

H_pairs, I_pairs = compute_HI_pairs(
        B_pairs=B_pairs, 
        Bzeta_pairs=Bzeta_pairs,
        grad_rho_pairs=grad_rho_pairs,
        kappa_g_pairs=kappa_g_pairs,
        w_pairs=w_pairs,
        pair_idx=pair_idx,
        pitch_inv=pitch_inv
        )

S_rp = np.zeros((nrho, Np), dtype=float)

# each row of H_pairs/I_pairs is one valid well; pair_idx[:,0:2] tells you which (rho,pitch)
contrib = np.where(I_pairs != 0.0, H_pairs**2 / I_pairs, 0.0)
np.add.at(S_rp, (pair_idx[:, 0], pair_idx[:, 1]), contrib)

# Fieldline length
L_rho = compute_L_rho(zeta_B, B_sup_zeta)

eps32_base = pitch_integral_simpson(S_rp, pitch_inv, L_rho)

eps32 = eps32_base * (B0 * R0 / grad_rho_avg)**2 * (np.pi / (8.0 * np.sqrt(2.0)))
eps_eff = eps32**(2.0 / 3.0)


print("S_rp =", S_rp)
print("L_rho =", L_rho)
print("eps32_base =", eps32_base)
print("effective ripple 3/2 =", eps32.data)
print("effective ripple =", eps_eff.data)


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


