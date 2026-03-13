#!/usr/bin/env python3
"""
Analytic test for build_spline_and_bounce_points_rho + plotting.

Domain: zeta in [-3*pi, 3*pi] (finite snapshot, NOT periodic)
Field:  sin(z), sin(2z), sin(3z), sin(z/2), sin(z/3)

Outputs:
  - prints diagnostics
  - saves plot: bounce_points_debug.png
"""
import pdb
import numpy as np

# safe on headless nodes
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _find_bounce_pairs import build_spline_and_bounce_points_rho

from _quad_utils import *
from _bounce_points import *

# creating a "fake" |B| for a test
def make_B_and_dB(
    z,
    B0=1.3,
    a1=0.12, a2=0.08, a3=0.06,
    a_half=0.05, a_third=0.04,
):
    z = np.asarray(z)
    B = B0 * (
        1.0
        + a1 * np.sin(z)
        + a2 * np.sin(2.0 * z)
        + a3 * np.sin(3.0 * z)
        + a_half * np.sin(0.5 * z)
        + a_third * np.sin((1.0 / 3.0) * z)
    )
    dB = B0 * (
        a1 * np.cos(z)
        + 2.0 * a2 * np.cos(2.0 * z)
        + 3.0 * a3 * np.cos(3.0 * z)
        + 0.5 * a_half * np.cos(0.5 * z)
        + (1.0 / 3.0) * a_third * np.cos((1.0 / 3.0) * z)
    )
    return B, dB


Nz = 1281
zeta = np.linspace(-5.0 * np.pi, 5.0 * np.pi, Nz)


# single rho
B, B_z = make_B_and_dB(zeta)
B = B[None, :]
B_z = B_z[None, :]

# pitch_inv in (minB,maxB)
Np = 27
mn, mx = float(B.min()), float(B.max())
margin = 0.03 * (mx - mn)
pitch_inv = np.linspace(mn + margin, mx - margin, Np)[None, :]

pdb.set_trace()

num_well = 15
z1, z2 = build_spline_and_bounce_points_rho(
    zeta=zeta,
    B=B,
    B_z=B_z,
    pitch_inv=pitch_inv,
    num_well=num_well,
)


rho_nodes = np.array([0.5])
x, w = gl_sin(11)

# zeta_pairs shape (, Nq)
# pair_idx shape [rho_idx, pitch_idx, well_idx]
pair_mask, pair_idx, zeta_pairs, w_pairs, zeta_flat, w_flat = bounce_quad_to_zeta(z1, z2, x, w, return_flat=True)
rho_flat = make_rho_flat(pair_idx, Nq=len(x), rho_nodes=rho_nodes)
zeta_full  = scatter_pairs_to_padded(zeta_pairs, pair_mask, fill_value=0.0)

#pdb.set_trace()

# theta_PEST at these zeta points should be np.mod(iota * zeta, 2*np.pi)
# by design since we always choose the alpha = 0 field line
##geom_flat = eq_geom(rho_flat, zeta_flat) 

# ---- DESC-style masking for structural checks ----
# mask identifies "real" pairs; padded invalid slots become NaN for comparisons
mask = (z1 - z2) != 0.0
z1c = np.where(mask, z1, np.nan)
z2c = np.where(mask, z2, np.nan)

cond_inv = (z1c > z2c)
cond_disc = (z1c[..., 1:] < z2c[..., :-1])

if np.any(cond_inv):
    r, p, w = np.argwhere(cond_inv)[0]
    print("INVERSION at (rho,pitch,well) =", (r, p, w))
    print("z1,z2 =", z1[r, p, w], z2[r, p, w], "pitch_inv =", pitch_inv[r, p])
    raise AssertionError("Intersects have an inversion (z1 > z2).")

if np.any(cond_disc):
    r, p, w = np.argwhere(cond_disc)[0]
    print("DISCONTINUITY at (rho,pitch,well) =", (r, p, w), "between wells", w, "and", w + 1)
    print("z2[w] =", z2[r, p, w], "z1[w+1] =", z1[r, p, w + 1], "pitch_inv =", pitch_inv[r, p])
    raise AssertionError("Detected discontinuity: z1[next] < z2[prev].")

# ---- numerical checks ----
def Bfun(x):  return make_B_and_dB(x)[0]
def dBfun(x): return make_B_and_dB(x)[1]

errs = []
viol_sign = 0
mid_ok = 0
mid_tot = 0

for p in range(Np):
    pv = pitch_inv[0, p]
    for w in range(num_well):
        a = z1[0, p, w]
        b = z2[0, p, w]
        if a == 0.0 and b == 0.0:
            continue

        errs.append(abs(Bfun(a) - pv))
        errs.append(abs(Bfun(b) - pv))

        if dBfun(a) > 1e-10:
            viol_sign += 1
        if dBfun(b) < -1e-10:
            viol_sign += 1

        mid = 0.5 * (a + b)
        mid_tot += 1
        if Bfun(mid) < pv:
            mid_ok += 1

print("max |B(z_bounce)-pitch_inv|:", (np.max(errs) if errs else 0.0))
print("median |B(z_bounce)-pitch_inv|:", (np.median(errs) if errs else 0.0))
print("derivative-sign violations:", viol_sign)
print("midpoint trapped-region check (B(mid)<pitch_inv):", mid_ok, "/", mid_tot)

# ---- plotting ----
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(zeta, B[0], lw=2)
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
#out = "bounce_points_debug.png"
#fig.savefig(out, dpi=200)
plt.show()





