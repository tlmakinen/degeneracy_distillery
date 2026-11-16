"""Summarise aligned one-step ensembles (ensemble.npz + record.json per run).

Usage: python scripts/camels_ensemble_summary.py <dir holding m13/ and rank/> <loadings.npz to write>

Per run: principal axes of the ensemble-mean coordinates over the test
simulations, the member spread along each, and how linear each axis is in the
inputs. Across runs: canonical correlations between settings on the same split,
principal angles between the parameter subspaces across splits, the share of
each parameter inside the informative subspace, and a varimax basis for the
subspace shared by all runs. Needs numpy and scipy only.
"""
import sys, json, numpy as np
from scipy.linalg import subspace_angles
np.set_printoptions(precision=2, suppress=True, linewidth=220)
root = sys.argv[1]
def load(setting, t):
    d = f"{root}/{setting}/runs/iterative_camels_d35_trial{t}"
    z = np.load(f"{d}/ensemble.npz", allow_pickle=False)
    rec = json.load(open(f"{d}/record.json"))
    return z["ys"], z["y"], z["y_std"], z["X"], rec
def cc(P, Q):
    P = P - P.mean(0); Q = Q - Q.mean(0)
    qp, _ = np.linalg.qr(P); qq, _ = np.linalg.qr(Q)
    return np.linalg.svd(qp.T @ qq, compute_uv=False)
def cv_r2(A, c, folds=5, ridge=0.0, seed=0):
    n = len(c); idx = np.random.default_rng(seed).permutation(n); pred = np.zeros_like(c)
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        G = A[tr].T @ A[tr] + ridge * np.eye(A.shape[1])
        pred[te] = A[te] @ np.linalg.solve(G, A[tr].T @ c[tr])
    return 1 - ((c - pred) ** 2).mean(0) / c.var(0)
R = {}
for setting in ("m13", "rank"):
    for t in range(3):
        ys, y, y_std, X, rec = load(setting, t)
        names = rec["whittle"]["keep_names"]
        K, n, m = ys.shape
        yc = y - y.mean(0)
        U, s, Vt = np.linalg.svd(yc, full_matrices=False)
        pcs = yc @ Vt.T                                   # (n, m), sorted by variance
        sd = pcs.std(0)
        dev = (ys - ys.mean(0)) @ Vt.T                    # member deviations along each PC
        spread = np.sqrt((dev ** 2).sum(0).mean(0) / (K - 1))    # pooled over simulations
        spread_pt = np.sqrt((dev ** 2).sum(0) / (K - 1))         # (n, m) per simulation
        info_from_var = 0.5 * np.log(sd ** 2 / sd[-1] ** 2)
        print(f"\n===== {setting} split {t} (m={m})")
        print(" information per axis, from the run   :", rec["discovery"]["info_final_nats"].replace(";", "  "))
        print(" 0.5 log(var_a / var_last), PCs of y  :", info_from_var)
        print(" sd over simulations per PC           :", sd)
        print(" member spread per PC                 :", spread)
        print(" spread / sd  (error bar, fraction)   :", spread / sd)
        print(" error on the ensemble mean / sd      :", spread / sd / np.sqrt(K))
        print(" spread 10th-90th pct over sims / sd  :", np.percentile(spread_pt, 10, 0) / sd, "to", np.percentile(spread_pt, 90, 0) / sd)
        Xs = (X - X.mean(0)) / X.std(0)
        A1 = np.c_[Xs, np.ones(n)]
        iu = np.triu_indices(Xs.shape[1])
        A2 = np.c_[A1, (Xs[:, :, None] * Xs[:, None, :])[:, iu[0], iu[1]]]
        c = pcs / sd
        B, *_ = np.linalg.lstsq(A1, c, rcond=None)
        print(" linear R2 (5-fold CV)                :", cv_r2(A1, c))
        print(" quadratic R2 (5-fold CV, ridge)      :", cv_r2(A2, c, ridge=1.0))
        R[(setting, t)] = dict(B=B[:-1], pcs=pcs, names=names, sd=sd, X=X)
names = R[("m13", 0)]["names"]
print("\n--- same split: canonical correlations between top-k PCs of the 13-latent fit and of the 7-latent fit")
for t in range(3):
    for k in (5, 6):
        print(f" split {t} k={k}:", cc(R[('m13', t)]['pcs'][:, :k], R[('rank', t)]['pcs'][:, :k]))
print("\n--- across splits: principal angles (deg) between the parameter subspaces of the top-k PCs (linear loadings)")
for setting in ("m13", "rank"):
    for k in (5, 6, 7):
        for (i, j) in ((0, 1), (0, 2), (1, 2)):
            ang = np.degrees(subspace_angles(R[(setting, i)]["B"][:, :k], R[(setting, j)]["B"][:, :k]))[::-1]
            print(f" {setting} k={k} splits {i},{j}:", ang)
print("\n--- share of each parameter inside the informative subspace (diag of projector; 1 = fully constrained direction)")
for k in (5, 6):
    Ps = []
    for setting in ("m13", "rank"):
        for t in range(3):
            Q, _ = np.linalg.qr(R[(setting, t)]["B"][:, :k]); Ps.append(Q @ Q.T)
    Ps = np.array(Ps)
    print(f" k={k}")
    order = np.argsort(-Ps.mean(0).diagonal())
    for j in order:
        print(f"   {names[j]:28s} mean {Ps[:, j, j].mean():.2f}   per run (m13 x3, rank x3): {np.round(Ps[:, j, j], 2)}")
    w, V = np.linalg.eigh(Ps.mean(0))
    print("   eigenvalues of the mean projector (1 = direction shared by all six runs):", w[::-1])
np.savez(sys.argv[2], **{f"{s}_{t}_B": R[(s, t)]["B"] for s in ("m13", "rank") for t in range(3)})

print('\n--- varimax basis of the 6-dimensional subspace shared by all runs (loadings on standardised inputs)')
z = np.load(sys.argv[2]); k = 6
Ps = []
for key in z.files:
    Q, _ = np.linalg.qr(z[key][:, :k]); Ps.append(Q @ Q.T)
w, V = np.linalg.eigh(np.mean(Ps, 0)); L = V[:, ::-1][:, :k]
def varimax(L, it=500, tol=1e-10):
    p, q = L.shape; Rm = np.eye(q); d = 0
    for _ in range(it):
        Lr = L @ Rm
        u, s, vt = np.linalg.svd(L.T @ (Lr ** 3 - Lr @ np.diag((Lr ** 2).sum(0)) / p))
        Rm = u @ vt; d_new = s.sum()
        if d_new - d < tol: break
        d = d_new
    return L @ Rm
Lr = varimax(L)
order = np.argsort(-np.abs(Lr).max(0))
for c in order:
    v = Lr[:, c]; v = v * np.sign(v[np.argmax(np.abs(v))])
    top = [j for j in np.argsort(-np.abs(v)) if abs(v[j]) >= 0.15]
    print("  " + "  ".join(f"{names[j]} {v[j]:+.2f}" for j in top))
# how well each run's own 6-dim subspace reproduces each rotated direction
print("captured by each run (cos^2):")
print(np.array([[float(v @ P @ v) for v in Lr.T[order]] for P in Ps]))
