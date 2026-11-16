#!/usr/bin/env python
"""Panel plots of the aligned CAMELS coordinates against the parameters.

Reads ``targets.npz`` written by ``camels_sr_native_mdl.py`` (the six aligned
coordinates of each split, their member spread, and the scaled parameters).

    python scripts/camels_aligned_coords_plot.py results/whittle_camels/sr_native
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1a1a19", "#6b6a63", "#e4e3dd"
SPLIT_COL = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 8, "axes.linewidth": 0.8,
    "font.family": "sans-serif",
})
PRETTY = {"Omega_m": r"$\Omega_m$", "Omega_b": r"$\Omega_b$", "H_0": r"$H_0$", "sigma_8": r"$\sigma_8$",
          "n_s": r"$n_s$", "A_SN1": r"$A_{\rm SN1}$", "A_SN2": r"$A_{\rm SN2}$", "IMFslope": "IMF slope",
          "SNII_MinMass": "SNII min mass", "WindFreeTravelDensFac": "wind free-travel\ndensity", "MinWindVel": "min wind vel.",
          "WindEnergyReductionFactor": "wind energy\nreduction", "WindDumpFactor": "wind dump"}

out = Path(sys.argv[1])
z = np.load(out / "targets.npz", allow_pickle=False)
names = [str(n) for n in z["names"]]
L = z["L"]
D = {t: dict(X=z[f"X_{t}"], c=z[f"c_{t}"], s=z[f"c_std_{t}"]) for t in range(3)}
order = np.argsort(-(L ** 2).sum(1))          # parameters by their share of the informative subspace
n_c, n_p = L.shape[1], len(names)

# ---- figure 1: every coordinate against every parameter
fig, axes = plt.subplots(n_c, n_p, figsize=(18, 9.6), sharex="col", sharey="row")
for j in range(n_c):
    for col, i in enumerate(order):
        ax = axes[j, col]
        strong = abs(L[i, j]) >= 0.15
        for t in range(3):
            ax.scatter(D[t]["X"][:, i], D[t]["c"][:, j], s=3.0, color=SPLIT_COL[t], alpha=0.55, linewidths=0,
                       rasterized=True)
        for sp in ax.spines.values():
            sp.set_edgecolor(INK if strong else GRID)
            sp.set_linewidth(1.1 if strong else 0.8)
        if strong:
            ax.text(0.05, 0.93, f"{L[i, j]:+.2f}", transform=ax.transAxes, va="top", ha="left", fontsize=8,
                    color=INK, fontweight="bold",
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0, alpha=0.85))
        ax.set_xlim(0.97, 2.03)
        ax.set_xticks([1, 2])
        ax.tick_params(length=2)
        if j == 0:
            ax.set_title(PRETTY.get(names[i], names[i]), fontsize=9, color=INK)
        if col == 0:
            ax.set_ylabel(rf"$\eta_{j + 1}$", fontsize=11, rotation=0, labelpad=10, va="center")
handles = [plt.Line2D([], [], marker="o", ls="", ms=6, color=SPLIT_COL[t], label=f"split {t}") for t in range(3)]
fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.995, 0.995))
fig.suptitle("Aligned coordinates of the neural map against each scaled parameter (205 test simulations per split)",
             x=0.01, y=0.99, ha="left", fontsize=12, fontweight="bold")
fig.text(0.01, 0.948, "Dark frames: parameters with a loading of 0.15 or more on that coordinate; the number is the loading. "
         "Parameters are ordered by their share of the informative subspace.", fontsize=9, color=MUTED)
fig.supxlabel("parameter, scaled to [1, 2]", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.4, w_pad=0.3)
fig.savefig(out / "aligned_coords_vs_theta.png", dpi=170)
plt.close(fig)

# ---- figure 2: linear part, and what is left
RES = {}
fig, axes = plt.subplots(2, n_c, figsize=(18, 6.6), gridspec_kw=dict(height_ratios=[1.25, 1]))
for j in range(n_c):
    a0, a1 = axes[0, j], axes[1, j]
    lims, shares = [], []
    for t in range(3):
        d = D[t]
        A = np.c_[d["X"], np.ones(len(d["X"]))]
        w = 1.0 / d["s"][:, j]
        b, *_ = np.linalg.lstsq(A * w[:, None], d["c"][:, j] * w, rcond=None)
        lin = A @ b
        res = d["c"][:, j] - lin
        share = res.var() / d["c"][:, j].var()
        a0.errorbar(lin, d["c"][:, j], yerr=d["s"][:, j], fmt="none", ecolor=SPLIT_COL[t], elinewidth=0.5, alpha=0.3)
        a0.scatter(lin, d["c"][:, j], s=5, color=SPLIT_COL[t], alpha=0.75, linewidths=0, label=f"split {t}")
        a1.errorbar(lin, res, yerr=d["s"][:, j], fmt="none", ecolor=SPLIT_COL[t], elinewidth=0.5, alpha=0.25)
        a1.scatter(lin, res, s=5, color=SPLIT_COL[t], alpha=0.6, linewidths=0)
        edges = np.quantile(lin, np.linspace(0, 1, 8))
        mid = 0.5 * (edges[1:] + edges[:-1])
        k = np.clip(np.digitize(lin, edges[1:-1]), 0, len(mid) - 1)
        mean = np.array([res[k == q].mean() for q in range(len(mid))])
        a1.plot(mid, mean, color=SPLIT_COL[t], lw=2, solid_capstyle="round")
        lims += [lin.min(), lin.max()]
        shares.append(share)
    lo, hi = min(lims) - 0.15, max(lims) + 0.15
    a0.plot([lo, hi], [lo, hi], color=INK, lw=0.8, zorder=0)
    a1.axhline(0, color=INK, lw=0.8, zorder=0)
    top = [names[i] for i in np.argsort(-np.abs(L[:, j])) if abs(L[i, j]) >= 0.3]
    a0.set_title(rf"$\eta_{j + 1}$: " + ", ".join(PRETTY.get(n, n).replace("\n", " ") for n in top), fontsize=9.5)
    a1.set_xlabel("linear part (weighted fit on the 13 parameters)", fontsize=8.5)
    a1.set_title("remainder: " + " / ".join(f"{100 * v:.1f}" for v in shares) + "% of variance (splits 0 / 1 / 2)",
                 fontsize=8.5, color=MUTED)
    RES[j] = shares
    a1.set_ylim(-0.75, 0.75)
    for ax in (a0, a1):
        ax.set_xlim(lo, hi)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=2)
axes[0, 0].set_ylabel("aligned coordinate", fontsize=9.5)
axes[1, 0].set_ylabel("coordinate minus linear part", fontsize=9.5)
axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper left", markerscale=1.6, handletextpad=0.2)
fig.suptitle("Each aligned coordinate against its own linear part (top), and the nonlinear remainder (bottom)",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.text(0.01, 0.925, "Bars: spread of the ten ensemble members. Thick lines: mean remainder in seven equal-count bins, per split.",
         fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(out / "linear_part_and_remainder.png", dpi=170)
plt.close(fig)

# ---- figure 3: the nonlinear remainder against every parameter
fig, axes = plt.subplots(n_c, n_p, figsize=(18, 9.6), sharex="col", sharey=True)
for j in range(n_c):
    res = {}
    for t in range(3):
        d = D[t]
        A = np.c_[d["X"], np.ones(len(d["X"]))]
        w = 1.0 / d["s"][:, j]
        b, *_ = np.linalg.lstsq(A * w[:, None], d["c"][:, j] * w, rcond=None)
        res[t] = d["c"][:, j] - A @ b
    for col, i in enumerate(order):
        ax = axes[j, col]
        strong = abs(L[i, j]) >= 0.15
        for t in range(3):
            x = D[t]["X"][:, i]
            ax.scatter(x, res[t], s=3.0, color=SPLIT_COL[t], alpha=0.35, linewidths=0, rasterized=True)
        for t in range(3):
            x = D[t]["X"][:, i]
            edges = np.quantile(x, np.linspace(0, 1, 7))
            mid = np.array([x[(x >= edges[q]) & (x <= edges[q + 1])].mean() for q in range(6)])
            mean = np.array([res[t][(x >= edges[q]) & (x <= edges[q + 1])].mean() for q in range(6)])
            ax.plot(mid, mean, color=SURFACE, lw=3.6, solid_capstyle="round")
            ax.plot(mid, mean, color=SPLIT_COL[t], lw=1.8, solid_capstyle="round")
        ax.axhline(0, color=INK, lw=0.6, zorder=0)
        for sp in ax.spines.values():
            sp.set_edgecolor(INK if strong else GRID)
            sp.set_linewidth(1.1 if strong else 0.8)
        ax.set_xlim(0.97, 2.03)
        ax.set_ylim(-0.45, 0.45)
        ax.set_xticks([1, 2])
        ax.tick_params(length=2)
        if j == 0:
            ax.set_title(PRETTY.get(names[i], names[i]), fontsize=9, color=INK)
        if col == 0:
            ax.set_ylabel(rf"$\eta_{j + 1}$", fontsize=11, rotation=0, labelpad=10, va="center")
fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.995, 0.995))
fig.suptitle("Nonlinear remainder of each coordinate (coordinate minus its linear part) against each scaled parameter",
             x=0.01, y=0.99, ha="left", fontsize=12, fontweight="bold")
fig.text(0.01, 0.948, "Lines: mean remainder in six equal-count bins, per split. Dark frames: parameters with a loading of 0.15 "
         "or more on that coordinate. Each coordinate has unit variance.", fontsize=9, color=MUTED)
fig.supxlabel("parameter, scaled to [1, 2]", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.4, w_pad=0.3)
fig.savefig(out / "remainder_vs_theta.png", dpi=170)
print("wrote", out / "aligned_coords_vs_theta.png", ",", out / "linear_part_and_remainder.png", "and", out / "remainder_vs_theta.png")
