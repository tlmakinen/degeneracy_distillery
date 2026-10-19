"""Side-by-side imshow of the learned Fisher F = J^T J over the prior.

Panel (a) is mean(F), which is signed, so it gets a diverging blue<->red ramp
with a neutral gray midpoint. Panel (b) is var(F), which is non-negative, so it
gets a single-hue sequential ramp. Both span orders of magnitude, hence symlog
and log norms; the colorbars say so.

Restricted to the top-k parameters by sqrt(diag(mean F)), ordered strongest
first, which is the ranking --sr-screen fisher uses.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, SymLogNorm

# Design-system values: blue sequential ramp, blue<->red diverging pair with a
# gray midpoint. Never a rainbow, never a hue at the diverging midpoint.
BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
RED = ["#fbd7d5", "#f4a89f", "#ea7a68", "#d9503a", "#ab2f26", "#7d1d17"]
GRAY_MID = "#f0efec"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"

SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE)
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", BLUE[::-1] + [GRAY_MID] + RED
)

PRETTY = {
    "Omega_m": r"$\Omega_m$",
    "Omega_b": r"$\Omega_b$",
    "sigma_8": r"$\sigma_8$",
    "n_s": r"$n_s$",
    "H_0": r"$H_0$",
    "A_SN1": r"$A_{\rm SN1}$",
    "A_SN2": r"$A_{\rm SN2}$",
    "A_AGN1": r"$A_{\rm AGN1}$",
    "A_AGN2": r"$A_{\rm AGN2}$",
    "IMFslope": "IMF slope",
    "SNII_MinMass": r"SNII $M_{\rm min}$",
    "WindFreeTravelDensFac": r"Wind travel $\rho$",
    "QuasarThreshold": "Quasar thresh.",
    "QuasarThresholdPower": "Quasar thr. pow.",
    "MinWindVel": r"Min wind $v$",
    "SofteningLength": "Softening",
    "ThermalWindFraction": "Therm. wind frac.",
    "VariableWindSpecMomentum": "Wind spec. mom.",
    "WindDumpFactor": "Wind dump",
    "BH_AccretionFactor": "BH accretion",
    "BH_EddingtonFactor": "BH Eddington",
    "SeedBlackHoleMass": r"Seed $M_{\rm BH}$",
    "UVB_H0_beta": r"UVB H$\,\beta$",
    "MaxSfrTimescale": "Max SFR timescale",
}


def label(name: str) -> str:
    return PRETTY.get(name, name.replace("_", " "))


def panel(ax, mat, names, cmap, norm, title, cbar_label):
    im = ax.imshow(mat, cmap=cmap, norm=norm, interpolation="nearest")
    n = len(names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([label(s) for s in names], rotation=55, ha="right",
                       fontsize=8.5, color=INK_2)
    ax.set_yticklabels([label(s) for s in names], fontsize=8.5, color=INK_2)
    # Thin surface separators between cells, so adjacent fills do not merge.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.1)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=11, color=INK, pad=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(cbar_label, fontsize=9, color=INK_2)
    cb.ax.tick_params(labelsize=8, color=INK_2, labelcolor=INK_2, length=2)
    cb.outline.set_visible(False)
    return im


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True,
                   help="Run directory holding runs/*/sr/split_data.npz")
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from degeneracy_distillery.problems.camels import PARAM_NAMES

    npz = sorted(a.run.glob("runs/*/sr/split_data.npz"))[0]
    F = np.asarray(np.load(npz, allow_pickle=True)["Fs_test"], dtype=np.float64)
    M, V = F.mean(0), F.var(0)
    keep = [int(v) for v in np.argsort(-np.sqrt(np.clip(np.diag(M), 0, None)))[: a.top_k]]
    names = [PARAM_NAMES[i] for i in keep]
    Mb, Vb = M[np.ix_(keep, keep)], V[np.ix_(keep, keep)]

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.9), facecolor=SURFACE)
    lim = float(np.abs(Mb).max())
    panel(axes[0], Mb, names, DIV,
          SymLogNorm(linthresh=1.0, vmin=-lim, vmax=lim, base=10),
          r"(a)  mean $F = J^{\mathsf{T}}J$   over the prior",
          "mean F  (symlog, linear below |1|)")
    panel(axes[1], Vb, names, SEQ,
          LogNorm(vmin=float(Vb.min()), vmax=float(Vb.max())),
          r"(b)  var $F = J^{\mathsf{T}}J$   over the prior",
          "var F  (log)")
    for ax in axes:
        ax.set_facecolor(SURFACE)

    fig.suptitle(
        f"CAMELS learned Fisher, top-{a.top_k} parameters by "
        r"$\sqrt{\mathrm{diag}\,\langle F\rangle}$ (strongest first);  "
        r"$\theta$ min-max scaled to $[1,2]$",
        fontsize=11.5, color=INK, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = a.out or (a.run / "fisher_top12.png")
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
