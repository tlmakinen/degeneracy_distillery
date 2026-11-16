#!/usr/bin/env python
"""Compare two Colab ensemble runs of the CAMELS top-13 problem, run by run.

    python scripts/camels_deep_compare.py results/whittle_camels/colab_top13 results/whittle_camels/colab_top13_deep

Per setting (13 latents, informative rank) and split: rank, held-out
information from the run record, member spread on the six leading axes, how
linear those axes are in the 13 scaled parameters, and the principal angles
between the two runs' six-dimensional parameter subspaces. For runs that saved
the member data summaries (``eta_raw``, ``eta_hat_raw``) it also gives the
information per axis measured on the test simulations, per member. numpy and
scipy only.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles

np.set_printoptions(precision=2, suppress=True, linewidth=200)
N_LEAD = 6


def cv_r2(A, c, folds=5):
    n = len(c)
    idx = np.random.default_rng(0).permutation(n)
    pred = np.zeros_like(c)
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        b, *_ = np.linalg.lstsq(A[tr], c[tr], rcond=None)
        pred[te] = A[te] @ b
    return 1 - ((c - pred) ** 2).mean(0) / c.var(0)


def test_information(eta, eta_hat):
    """Per member: 0.5 log(var / residual) along the principal axes of eta on the test set."""
    out = []
    for e, h in zip(eta, eta_hat):
        e = np.asarray(e, dtype=np.float64)
        h = np.asarray(h, dtype=np.float64)
        ec = e - e.mean(0)
        _, _, Vt = np.linalg.svd(ec, full_matrices=False)
        var = (ec @ Vt.T).var(0)
        res = (((e - h) @ Vt.T) ** 2).mean(0)
        out.append(0.5 * np.log(var / res))
    return np.array(out)


def one(path: Path):
    z = np.load(path / "ensemble.npz", allow_pickle=False)
    rec = json.loads((path / "record.json").read_text())["discovery"]
    ys, X = z["ys"], z["X"]
    K = ys.shape[0]
    y = ys.mean(0)
    yc = y - y.mean(0)
    _, _, Vt = np.linalg.svd(yc, full_matrices=False)
    P = Vt[:N_LEAD].T
    pcs = yc @ P
    sd = pcs.std(0)
    dev = (ys - ys.mean(0)) @ P
    spread = np.sqrt((dev ** 2).sum(0).mean(0) / (K - 1))
    Xs = (X - X.mean(0)) / X.std(0)
    A = np.c_[Xs, np.ones(len(Xs))]
    B, *_ = np.linalg.lstsq(A, pcs / sd, rcond=None)
    info = np.array([float(v) for v in rec["info_final_nats"].split(";")])
    r = dict(r_hat=rec["r_hat"], m_fit=rec["m_fit"], info=info, info_total=float(info.sum()),
             spread=spread / sd, nonlin=1 - cv_r2(A, pcs / sd), B=B[:-1],
             calib=max(float(v) for v in rec["calib_ratio_loo"].split(";")) if rec.get("calib_ratio_loo") else np.nan,
             arch=rec.get("ensemble_arch", "").split(";")[0])
    if "eta_hat_raw" in z.files:
        ti = test_information(z["eta_raw"], z["eta_hat_raw"])
        r["test_info"] = ti
    return r


def main():
    roots = [Path(p) for p in sys.argv[1:]]
    R = {}
    for root in roots:
        for setting in ("m13", "rank"):
            for t in range(3):
                d = root / setting / "runs" / f"iterative_camels_d35_trial{t}"
                if (d / "ensemble.npz").exists():
                    R[(root.name, setting, t)] = one(d)
    for setting in ("m13", "rank"):
        print(f"\n################ setting: {setting}")
        for t in range(3):
            print(f"\n== split {t}")
            for root in roots:
                r = R.get((root.name, setting, t))
                if r is None:
                    print(f"  {root.name}: missing")
                    continue
                print(f"  {root.name}  [{r['arch']}]  rank {r['r_hat']}, latents {r['m_fit']}")
                print(f"     information per axis (validation, nats): {r['info'][:9]}  total {r['info_total']:.2f}")
                print(f"     member spread / range, six leading axes:  {r['spread']}")
                print(f"     nonlinear share (1 - CV R2 of a linear fit): {r['nonlin']}")
                print(f"     worst calibration ratio {r['calib']:.2f}")
                if "test_info" in r:
                    ti = np.sort(r["test_info"], axis=1)[:, ::-1]
                    print(f"     information per axis on the test simulations, median over members: {np.median(ti, 0)[:9]}")
                    print(f"       total per member: {ti.clip(min=0).sum(1)}  | axes above 0.5 nats per member: {(ti > 0.5).sum(1)}")
            if len(roots) == 2 and all((root.name, setting, t) in R for root in roots):
                a, b = (R[(root.name, setting, t)]["B"] for root in roots)
                print(f"  principal angles between the two runs' six-dimensional parameter subspaces (deg): "
                      f"{np.degrees(subspace_angles(a, b))[::-1]}")
    print("\n################ medians over splits")
    for root in roots:
        for setting in ("m13", "rank"):
            rs = [R[k] for k in R if k[0] == root.name and k[1] == setting]
            if rs:
                print(f"  {root.name:22s} {setting:5s} information {np.round([r['info_total'] for r in rs], 2)}  "
                      f"spread/range {np.median([r['spread'] for r in rs]):.3f}  "
                      f"nonlinear share {100 * np.median([r['nonlin'] for r in rs]):.2f}%")


if __name__ == "__main__":
    main()
