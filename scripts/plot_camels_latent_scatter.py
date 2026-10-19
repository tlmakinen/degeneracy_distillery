"""Scatter of each learned latent against its strongest drivers.

For each component ``eta_i`` the columns are the top-k parameters by
``Var[d eta_i / d theta_j]`` over the prior. A large derivative variance means
the response bends, so this ranks parameters by how nonlinearly they drive
that component.

Each panel adds a binned median. eta_i depends on all 35 parameters, so a
single-parameter scatter is a cloud; the median line shows the trend through
it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PT = "#2a78d6"      # categorical slot 1
TREND = "#eb6834"   # categorical slot 2
SURFACE = "#fcfcfb"
INK, INK2 = "#0b0b0b", "#52514e"

PRETTY = {
    "Omega_m": r"$\Omega_m$", "Omega_b": r"$\Omega_b$", "sigma_8": r"$\sigma_8$",
    "n_s": r"$n_s$", "H_0": r"$H_0$", "A_SN1": r"$A_{\rm SN1}$",
    "A_SN2": r"$A_{\rm SN2}$", "A_AGN1": r"$A_{\rm AGN1}$", "A_AGN2": r"$A_{\rm AGN2}$",
    "IMFslope": "IMF slope", "SNII_MinMass": r"SNII $M_{\rm min}$",
    "WindFreeTravelDensFac": r"Wind travel $\rho$", "QuasarThreshold": "Quasar thresh.",
    "MinWindVel": r"Min wind $v$", "SofteningLength": "Softening",
    "ThermalWindFraction": "Therm. wind frac.", "WindDumpFactor": "Wind dump",
    "VariableWindSpecMomentum": "Wind spec. mom.", "BH_AccretionFactor": "BH accretion",
    "BH_EddingtonFactor": "BH Eddington", "UVB_H0_beta": r"UVB H$\,\beta$",
}


def label(n: str) -> str:
    return PRETTY.get(n, n.replace("_", " "))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from degeneracy_distillery.problems.camels import PARAM_NAMES

    npz = sorted(a.run.glob("runs/*/sr/split_data.npz"))[0]
    d = np.load(npz, allow_pickle=True)
    X = np.asarray(d["X_test"], dtype=np.float64)
    Y = np.asarray(d["y_test"], dtype=np.float64)
    dy = np.asarray(d["dy_sr_test"], dtype=np.float64)

    # Augmentation may repeat theta rows; one point per distinct theta.
    _, uniq = np.unique(np.round(X, 10), axis=0, return_index=True)
    uniq = np.sort(uniq)
    Xu, Yu = X[uniq], Y[uniq]
    V = dy.var(0)                       # (m, d)
    m, k = Y.shape[1], a.top_k

    fig, axes = plt.subplots(m, k, figsize=(3.05 * k, 2.35 * m),
                             facecolor=SURFACE, squeeze=False)
    for i in range(m):
        order = np.argsort(-V[i])[:k]
        for c, j in enumerate(order):
            ax = axes[i][c]
            x, y = Xu[:, j], Yu[:, i]
            ax.scatter(x, y, s=4, c=PT, alpha=0.22, linewidths=0, rasterized=True)
            edges = np.quantile(x, np.linspace(0, 1, 13))
            mid, med = [], []
            for b in range(len(edges) - 1):
                sel = (x >= edges[b]) & (x <= edges[b + 1])
                if sel.sum() > 5:
                    mid.append(0.5 * (edges[b] + edges[b + 1]))
                    med.append(np.median(y[sel]))
            ax.plot(mid, med, color=TREND, lw=2.0, solid_capstyle="round")
            r = np.corrcoef(x, y)[0, 1]
            ax.set_title(f"{label(PARAM_NAMES[j])}   Var$[\\partial]$ {V[i, j]:.3f}   "
                         f"r {r:+.2f}", fontsize=8.5, color=INK, pad=4)
            ax.tick_params(labelsize=7, colors=INK2, length=2)
            ax.grid(alpha=0.15, lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color("#d8d7d2")
            ax.set_facecolor(SURFACE)
            if c == 0:
                ax.set_ylabel(rf"$\eta_{i}$", fontsize=11, color=INK)
            if i == m - 1:
                ax.set_xlabel(r"$\theta$ (scaled)", fontsize=8, color=INK2)

    fig.suptitle(
        "Learned latents against their strongest drivers\n"
        f"columns: top-{k} parameters by "
        r"Var$[\partial\eta_i/\partial\theta_j]$"
        f" over the prior   .   {len(Xu)} distinct "
        r"$\theta$" "   .   orange = binned median",
        fontsize=10.5, color=INK, y=0.995, linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.967))
    out = a.out or (a.run / "latent_scatter.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}   ({len(Xu)} distinct theta of {len(X)} rows)")


if __name__ == "__main__":
    main()
