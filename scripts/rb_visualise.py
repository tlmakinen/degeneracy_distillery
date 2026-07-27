"""
Visualisations for the Rayleigh-Bénard degeneracy-distillery run.
Saves figures to the run's output directory.
"""
import argparse, pickle, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy import stats

from degeneracy_distillery.align_coords import load_and_process_data_v2
from degeneracy_distillery.sr_utils import get_y_sr

# ── physical-space labels ────────────────────────────────────────────────────
THETA_LABELS   = [r"$\log_{10}\,\mathrm{Ra}$", r"$\log_{10}\,\mathrm{Pr}$", r"$\log_{10}\,\Gamma$"]
THETA_NAMES    = ["log10_Ra", "log10_Pr", "log10_Gamma"]
ETA_LABELS     = [r"$\eta_1$", r"$\eta_2$", r"$\eta_3$"]
CMAPS          = ["plasma", "viridis", "cividis"]


def load_data(out_dir):
    """Return aligned dict, physical theta subset, and pkl."""
    pkl_path = os.path.join(out_dir, "sr_results_rb", "sr_expressions.pkl")
    with open(pkl_path, "rb") as f:
        pkl = pickle.load(f)

    aligned = load_and_process_data_v2(
        out_dir + "/", "rb_grossmann_lohse.npz",
        num_samps=4000, seed=0,
        align_mode="kabsch", separate_nonlinearity=False
    )

    # recover physical theta for the aligned subset
    # scaled theta in [1,2]; inverse: phys = data_min + (scaled-1)*(data_max-data_min)
    npz_flat = np.load(os.path.join(out_dir, "rb_grossmann_lohse.npz"))
    theta_scaled_all = np.array(npz_flat["theta"])           # (N, 3)
    theta_sub = theta_scaled_all[aligned["randidx"]]          # (4000, 3)
    data_min = pkl["scaler_data_min"].astype(float)
    data_max = pkl["scaler_data_max"].astype(float)
    theta_phys = data_min + (theta_sub - 1.0) * (data_max - data_min)

    return aligned, theta_phys, pkl


# ── Figure 1: aligned-coord scatter coloured by physical params ──────────────
def fig_X_vs_theta(aligned, theta_phys, out_dir):
    X = aligned["X"]   # (N, 3)  — Fisher eigenvector coords
    y = aligned["y"]   # (N, 3)  — flattened eta coords

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle("Aligned coordinates $X$ coloured by physical parameters", fontsize=13)

    for col, (lbl, cmap) in enumerate(zip(THETA_LABELS, CMAPS)):
        c = theta_phys[:, col]
        norm = Normalize(c.min(), c.max())
        for row in range(3):
            ax = axes[row, col]
            ax.scatter(X[:, col], X[:, row], c=c, cmap=cmap, s=3, alpha=0.5, norm=norm, rasterized=True)
            ax.set_xlabel(f"$X_{col+1}$", fontsize=9)
            ax.set_ylabel(f"$X_{row+1}$", fontsize=9)
            ax.tick_params(labelsize=7)
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes[:, col], shrink=0.6, pad=0.02)
        cb.set_label(lbl, fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "fig1_X_vs_theta.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ── Figure 2: eta scatter coloured by physical params ───────────────────────
def fig_eta_vs_theta(aligned, theta_phys, out_dir):
    y = aligned["y"]   # (N, 3)

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle(r"Flattened coordinates $\eta$ coloured by physical parameters", fontsize=13)

    for col, (lbl, cmap) in enumerate(zip(THETA_LABELS, CMAPS)):
        c = theta_phys[:, col]
        norm = Normalize(c.min(), c.max())
        for row in range(3):
            ax = axes[row, col]
            ax.scatter(y[:, col], y[:, row], c=c, cmap=cmap, s=3, alpha=0.5, norm=norm, rasterized=True)
            ax.set_xlabel(ETA_LABELS[col], fontsize=9)
            ax.set_ylabel(ETA_LABELS[row], fontsize=9)
            ax.tick_params(labelsize=7)
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes[:, col], shrink=0.6, pad=0.02)
        cb.set_label(lbl, fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "fig2_eta_vs_theta.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ── Figure 3: each eta component vs log10(Gamma) ────────────────────────────
def fig_eta_vs_gamma(aligned, theta_phys, pkl, out_dir):
    y = aligned["y"]
    log_gamma = theta_phys[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(r"Flattened coordinates $\eta_i$ vs $\log_{10}\,\Gamma$  (degeneracy check)", fontsize=12)

    rows = pkl["gamma_validation"]["rows"]
    best_r = pkl["gamma_validation"]["best_gamma_abs_corr"]

    for i, ax in enumerate(axes):
        r = rows[i]["r_with_log_gamma"]
        ax.scatter(log_gamma, y[:, i], s=3, alpha=0.4, color=f"C{i}", rasterized=True)
        ax.set_xlabel(r"$\log_{10}\,\Gamma$", fontsize=11)
        ax.set_ylabel(ETA_LABELS[i], fontsize=11)
        ax.set_title(f"Pearson $r={r:+.3f}$", fontsize=10)
        ax.tick_params(labelsize=8)
        if abs(r) == best_r:
            ax.set_facecolor("#fff8e1")  # highlight the best

    fig.tight_layout()
    path = os.path.join(out_dir, "fig3_eta_vs_gamma.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ── Figure 4: SR expression vs true eta ─────────────────────────────────────
def fig_sr_vs_eta(aligned, pkl, out_dir):
    X = aligned["X"]
    y = aligned["y"]
    pruned = pkl["pruned_exprs"]
    y_sr = np.array(get_y_sr(pruned, X))  # (N, 3)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(r"SR expression $\hat{\eta}_i(X)$ vs true $\eta_i$", fontsize=12)

    for i, ax in enumerate(axes):
        r, _ = stats.pearsonr(y_sr[:, i], y[:, i])
        lims = [min(y[:, i].min(), y_sr[:, i].min()),
                max(y[:, i].max(), y_sr[:, i].max())]
        ax.scatter(y[:, i], y_sr[:, i], s=3, alpha=0.4, color=f"C{i}", rasterized=True)
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.6)
        ax.set_xlabel(fr"True $\eta_{i+1}$", fontsize=11)
        ax.set_ylabel(fr"SR $\hat{{\eta}}_{i+1}$", fontsize=11)
        ax.set_title(f"$r={r:.4f}$", fontsize=10)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_sr_vs_eta.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ── Figure 5: Fisher eigenvalue spectra + flatness ──────────────────────────
def fig_fisher_spectra(aligned, pkl, out_dir):
    Fs = aligned["Fs"]          # (N, 3, 3)  — Fisher in aligned coords
    Favg = aligned["Favg"]      # (N, 3, 3)  — same

    # per-point eigenvalues
    eigs = np.linalg.eigvalsh(Fs)         # (N, 3), ascending
    eigs_sorted = np.sort(eigs, axis=1)[:, ::-1]  # descending

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Fisher information spectrum after flattening", fontsize=12)

    # violin per eigenvalue
    ax = axes[0]
    vp = ax.violinplot([eigs_sorted[:, i] for i in range(3)],
                       positions=[1, 2, 3], showmedians=True)
    for body in vp['bodies']:
        body.set_alpha(0.5)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([r"$\lambda_1$ (largest)", r"$\lambda_2$", r"$\lambda_3$ (degenerate)"])
    ax.set_ylabel("Eigenvalue", fontsize=10)
    ax.set_title("Per-point eigenvalue distribution", fontsize=10)
    ax.set_yscale("log")

    # condition number distribution
    ax2 = axes[1]
    cond = eigs_sorted[:, 0] / (eigs_sorted[:, 2] + 1e-30)
    ax2.hist(np.log10(cond), bins=50, color="steelblue", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax2.set_xlabel(r"$\log_{10}\,\kappa(F)$  (condition number)", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_title("Fisher condition number distribution", fontsize=10)
    ax2.axvline(np.log10(np.median(cond)), color="C1", ls="--", label=f"median $\\kappa={np.median(cond):.0f}$")
    ax2.legend(fontsize=9)

    flat = pkl["flatness"]
    textstr = (f"Flatness (MDL): {flat['mdl']:.3f}\n"
               f"Flatness (NN): {flat['nn']:.3f}\n"
               f"Raw $\\theta$ flatness: {flat['raw_theta']:.3f}")
    fig.text(0.5, -0.04, textstr, ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

    fig.tight_layout()
    path = os.path.join(out_dir, "fig5_fisher_spectra.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ── Figure 6: physical-space expression heatmaps ────────────────────────────
def fig_physical_expressions(pkl, out_dir):
    """Evaluate the physical expressions on a Ra-Pr-Gamma grid."""
    phys_exprs = pkl["physical_exprs"]
    data_min = pkl["scaler_data_min"].astype(float)
    data_max = pkl["scaler_data_max"].astype(float)

    N = 80
    log_Ra  = np.linspace(data_min[0], data_max[0], N)
    log_Pr  = np.linspace(data_min[1], data_max[1], N)
    log_G   = np.linspace(data_min[2], data_max[2], N)

    def eval_expr(expr_str, t1, t2, t3):
        import re
        safe = {"sqrt": np.sqrt, "__builtins__": {}}
        code = re.sub(r'\btheta1\b', 't1', expr_str)
        code = re.sub(r'\btheta2\b', 't2', code)
        code = re.sub(r'\btheta3\b', 't3', code)
        return eval(compile(code, "<expr>", "eval"), {"sqrt": np.sqrt, "t1": t1, "t2": t2, "t3": t3})

    # Heatmaps: fix Gamma at median, vary Ra vs Pr
    log_G_mid = 0.5 * (data_min[2] + data_max[2])
    Ra_grid, Pr_grid = np.meshgrid(log_Ra, log_Pr)  # each (N, N)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        fr"Physical expressions evaluated on (log Ra, log Pr) grid  [$\log_{{10}}\Gamma={log_G_mid:.2f}$]",
        fontsize=12
    )

    for i, (expr, lbl) in enumerate(zip(phys_exprs, ETA_LABELS)):
        try:
            Z = eval_expr(expr, Ra_grid, Pr_grid, log_G_mid)
        except Exception as e:
            print(f"  expr {i} eval error: {e}")
            continue
        ax = axes[0, i]
        im = ax.pcolormesh(Ra_grid, Pr_grid, Z, cmap="RdBu_r", shading="auto")
        ax.set_xlabel(r"$\log_{10}\,\mathrm{Ra}$", fontsize=9)
        ax.set_ylabel(r"$\log_{10}\,\mathrm{Pr}$", fontsize=9)
        ax.set_title(lbl, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)

    # Second row: vary Ra vs Gamma, fix Pr at median
    log_Pr_mid = 0.5 * (data_min[1] + data_max[1])
    Ra_grid2, G_grid2 = np.meshgrid(log_Ra, log_G)

    for i, (expr, lbl) in enumerate(zip(phys_exprs, ETA_LABELS)):
        try:
            Z2 = eval_expr(expr, Ra_grid2, log_Pr_mid, G_grid2)
        except Exception as e:
            print(f"  expr {i} eval error: {e}")
            continue
        ax = axes[1, i]
        im = ax.pcolormesh(Ra_grid2, G_grid2, Z2, cmap="RdBu_r", shading="auto")
        ax.set_xlabel(r"$\log_{10}\,\mathrm{Ra}$", fontsize=9)
        ax.set_ylabel(r"$\log_{10}\,\Gamma$", fontsize=9)
        ax.set_title(lbl + fr"  [$\log_{{10}}\,\mathrm{{Pr}}={log_Pr_mid:.1f}$]", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "fig6_physical_expressions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ── Figure 7: eta3 (degenerate direction) sensitivity to Gamma ───────────────
def fig_gamma_sensitivity(aligned, theta_phys, pkl, out_dir):
    """Show how eta_3 varies with Gamma at fixed Ra/Pr slices."""
    y = aligned["y"]
    log_gamma = theta_phys[:, 2]
    log_Ra    = theta_phys[:, 0]
    log_Pr    = theta_phys[:, 1]

    # Bin by Ra and Pr quintiles, show eta_3 vs Gamma in each bin
    Ra_q  = np.quantile(log_Ra, [0.2, 0.8])
    Pr_q  = np.quantile(log_Pr, [0.2, 0.8])

    masks = {
        "low Ra, low Pr":  (log_Ra < Ra_q[0]) & (log_Pr < Pr_q[0]),
        "high Ra, low Pr": (log_Ra > Ra_q[1]) & (log_Pr < Pr_q[0]),
        "low Ra, high Pr": (log_Ra < Ra_q[0]) & (log_Pr > Pr_q[1]),
        "high Ra, high Pr":(log_Ra > Ra_q[1]) & (log_Pr > Pr_q[1]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["C0", "C1", "C2", "C3"]
    ax = axes[0]
    for (label, mask), col in zip(masks.items(), colors):
        if mask.sum() < 5:
            continue
        ax.scatter(log_gamma[mask], y[mask, 2], s=8, alpha=0.6, color=col, label=label, rasterized=True)
    ax.set_xlabel(r"$\log_{10}\,\Gamma$", fontsize=11)
    ax.set_ylabel(r"$\eta_3$  (degenerate direction)", fontsize=11)
    ax.set_title(r"$\eta_3$ vs $\Gamma$ in Ra-Pr slices", fontsize=11)
    ax.legend(fontsize=8, markerscale=2)

    # eta3 marginal histogram coloured by Gamma quartile
    ax2 = axes[1]
    q_gamma = np.quantile(log_gamma, [0, 0.25, 0.5, 0.75, 1.0])
    q_labels = ["Q1 (low Γ)", "Q2", "Q3", "Q4 (high Γ)"]
    for qi in range(4):
        m = (log_gamma >= q_gamma[qi]) & (log_gamma < q_gamma[qi+1])
        ax2.hist(y[m, 2], bins=30, alpha=0.5, label=q_labels[qi], density=True)
    ax2.set_xlabel(r"$\eta_3$", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title(r"$\eta_3$ distribution by $\Gamma$ quartile", fontsize=11)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig7_gamma_sensitivity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=
        "results/rb_notebook/full_3618408_20260622_164431",
        help="Run output directory")
    args = parser.parse_args()

    print(f"Loading data from {args.out_dir} ...")
    aligned, theta_phys, pkl = load_data(args.out_dir)
    print(f"  X: {aligned['X'].shape}  y: {aligned['y'].shape}  theta_phys: {theta_phys.shape}")

    print("\nGenerating figures ...")
    fig_X_vs_theta(aligned, theta_phys, args.out_dir)
    fig_eta_vs_theta(aligned, theta_phys, args.out_dir)
    fig_eta_vs_gamma(aligned, theta_phys, pkl, args.out_dir)
    fig_sr_vs_eta(aligned, pkl, args.out_dir)
    fig_fisher_spectra(aligned, pkl, args.out_dir)
    fig_physical_expressions(pkl, args.out_dir)
    fig_gamma_sensitivity(aligned, theta_phys, pkl, args.out_dir)

    print("\nDone. Figures saved to", args.out_dir)


if __name__ == "__main__":
    main()
