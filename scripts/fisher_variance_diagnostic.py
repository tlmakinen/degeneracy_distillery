#!/usr/bin/env python
"""How much does the learned Fisher vary over the prior?

If Var_theta(F) is small there is little nonlinear parameter dependence for the
flattener and symbolic regression to extract, and a low recovery rate is a
property of the problem rather than a failure of the pipeline.

Reads F_ensemble / ensemble_weights / theta straight out of each run's saved
*_flatten.npz, so it needs no retraining and no GPU.

    python scripts/fisher_variance_diagnostic.py
    python scripts/fisher_variance_diagnostic.py --npz path/to/x_flatten.npz
"""
import argparse, glob, os
import numpy as np

SCRATCH = os.environ.get("SCRATCH", "/data103/makinen/degeneracy_experiments")
FU = os.path.join(SCRATCH, "follow_up_results")

DEFAULT_RUNS = [
    ("Rosenbrock  (10/10)", f"{FU}/rosenbrock/rebuttal/seed_*/rosen_flatten.npz"),
    ("SIR         (10/10)", f"{FU}/sir/rebuttal/seed_*/sir_flatten.npz"),
    ("TaylorF2    (10/10)", f"{FU}/gw_taylorf2/rebuttal/seed_*/gw_flatten.npz"),
    ("IMRPhenomD  ( 7/10)", f"{FU}/gw_imrphenomd/rebuttal/seed_*/imr_flatten.npz"),
    ("Kolmogorov  ( 1/10)", f"{FU}/kolmogorov/rebuttal/seed_*/kolmogorov_flatten.npz"),
    ("Kuramoto    ( 0/9 )", f"{FU}/kuramoto/rebuttal/seed_*/kuramoto_flatten.npz"),
    ("Rayleigh-Ben( 0/9 )", f"{SCRATCH}/rebuttal_discovery/rayleigh_benard/seed_*/rayleigh_benard_flatten.npz"),
    ("QM7b        ( 0/10)", f"{FU}/qm7b/rebuttal_scalefix/seed_*/qm7b_flattening.npz"),
]


def analyse(npz):
    d = np.load(npz)
    w = d["ensemble_weights"].astype(np.float64)
    w = w / w.sum()
    F = np.einsum("m,mnij->nij", w, d["F_ensemble"].astype(np.float64))
    th = d["theta"].astype(np.float64)
    N, p, _ = F.shape

    Fbar = F.mean(0)
    # Overall relative variation of F across the prior.
    dev = np.linalg.norm(F - Fbar, axis=(1, 2)) / (np.linalg.norm(Fbar) + 1e-30)

    # Shape-only: normalise each F to unit trace so overall scale drops out and
    # only the change in anisotropy / orientation survives.
    Fs = F / np.trace(F, axis1=1, axis2=2)[:, None, None]
    Fsbar = Fs.mean(0)
    devs = np.linalg.norm(Fs - Fsbar, axis=(1, 2)) / (np.linalg.norm(Fsbar) + 1e-30)

    # Fraction of Var_theta(F) explained by a LINEAR model in theta. 1 - R^2 is
    # the genuinely nonlinear content, which is what SR has to find.
    X = np.concatenate([np.ones((N, 1)), th], 1)
    Y = F.reshape(N, -1)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ beta
    ss_tot = ((Y - Y.mean(0)) ** 2).sum()
    r2_lin = 1 - (res ** 2).sum() / (ss_tot + 1e-30)

    return dict(p=p, dev=float(np.median(dev)), shape=float(np.median(devs)),
                r2_lin=float(r2_lin),
                nonlin=float(np.median(dev) * (1 - r2_lin)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", help="analyse a single npz instead of the campaign")
    a = ap.parse_args()
    runs = [("(single)", a.npz)] if a.npz else DEFAULT_RUNS

    hdr = f"{'experiment':<21}{'p':>2}{'|dF|/|F|':>10}{'shape':>8}{'R2lin':>8}{'nonlin':>9}{'seeds':>7}"
    print(hdr)
    print("-" * len(hdr))
    for name, pat in runs:
        files = sorted(glob.glob(pat)) if "*" in pat else [pat]
        rs = []
        for f in files:
            try:
                rs.append(analyse(f))
            except Exception:
                pass
        if not rs:
            print(f"{name:<21} no data")
            continue
        g = lambda k: np.array([r[k] for r in rs])
        print(f"{name:<21}{rs[0]['p']:>2}{g('dev').mean():>10.3f}{g('shape').mean():>8.3f}"
              f"{g('r2_lin').mean():>8.3f}{g('nonlin').mean():>9.3f}{len(rs):>7}")


if __name__ == "__main__":
    main()
