"""Minimal simulator + fishnets probe at d=8, with an optional PCA front-end.

Run on a GPU node -- the JAX CPU backend fails for this pipeline at d>=3:
    srun --partition=pscomp --gres=gpu:1 --time=01:00:00 --cpus-per-task=8 --mem=40G \
        bash -c 'source /home/makinen/venvs/degen/bin/activate && \
                 python fishnet_d8_probe.py'

Toggle N_PCA below:
    None -> current behaviour (raw 20-dim y)
    5    -> PCA to 5 comps (drops 15 noise dims, rank still free to exceed 1)
    1    -> DO NOT USE for rank claims: a scalar summary forces F to be rank 1
            by construction, making "intrinsic dim = 1" an artifact.
"""

import sys
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

sys.path.insert(0, "/home/makinen/repositories/degeneracy_distillery")
from scripts.heater_discovery_dim_scaling_sweep import (      # noqa: E402
    ChainHeaterCfg, chain_dataset, select_informative_axes,
)
from degeneracy_distillery.training_loop_fishnets import train_fishnets  # noqa: E402

# ----------------------------------------------------------------- knobs
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--d", type=int, default=8)
_ap.add_argument("--nsims", type=int, default=1000)
_ap.add_argument("--ntest", type=int, default=1000)
_ap.add_argument("--num-fishnets", type=int, default=20)
_ap.add_argument("--fishnet-epochs", type=int, default=5000)
_ap.add_argument("--n-pca", type=int, default=0, help="0/absent = no PCA")
_ap.add_argument("--seed", type=int, default=0)
_ap.add_argument("--outdir", default=None)
_a = _ap.parse_args()
D, NSIMS, NTEST = _a.d, _a.nsims, _a.ntest
NUM_FISHNETS, FISHNET_EPOCHS = _a.num_fishnets, _a.fishnet_epochs
N_PCA = _a.n_pca if _a.n_pca and _a.n_pca > 0 else None
SEED = _a.seed
OUTDIR = _a.outdir or f"./_probe_d{D}_pca{_a.n_pca}_f{NUM_FISHNETS}"

# ----------------------------------------------------------------- simulate
cfg = ChainHeaterCfg()
rng = np.random.default_rng(SEED)
theta_tr, y_tr = chain_dataset(NSIMS, D, cfg, rng)
theta_te, y_te = chain_dataset(NTEST, D, cfg, rng)
print(f"theta {theta_tr.shape}   y {y_tr.shape}   (data dim is n_t={cfg.n_t}, "
      f"fixed for every d)")

# ----------------------------------------------------------------- optional PCA
if N_PCA is not None:
    # Fit on the FIRST 100 training sims only, using y alone -- never theta.
    fit_n = min(100, len(y_tr))
    mu = y_tr[:fit_n].mean(0)
    _, S, Vt = np.linalg.svd(y_tr[:fit_n] - mu, full_matrices=False)
    comps = Vt[:N_PCA]
    ev = S ** 2 / (S ** 2).sum()
    print(f"\nPCA fit on {fit_n} sims -> keeping {N_PCA} of {cfg.n_t} comps")
    print("  explained variance:", np.round(ev[:min(6, len(ev))], 6))
    print(f"  cumulative kept:    {ev[:N_PCA].sum():.8f}")
    k = cfg.thermal_kernel
    print(f"  |cos(PC1, kernel)|: {abs(comps[0] @ k) / np.linalg.norm(k):.6f}")
    if N_PCA == 1:
        print("  !! WARNING: k=1 makes the Fisher rank-1 by construction.")
    y_tr = ((y_tr - mu) @ comps.T).astype(np.float32)
    y_te = ((y_te - mu) @ comps.T).astype(np.float32)
    print(f"  y is now {y_tr.shape}")

# ----------------------------------------------------------------- fishnets
# Same settings as the sweep and as heater_minimal_distillery.py.
embedding_net = nn.Sequential([nn.Dense(64), nn.gelu, nn.Dense(32), nn.gelu])
train_fishnets(
    jnp.asarray(theta_tr), jnp.asarray(y_tr),
    jnp.asarray(theta_te), jnp.asarray(y_te),
    num_models=NUM_FISHNETS,
    train_epochs=FISHNET_EPOCHS,
    patience=30,
    n_layers=[2, 5],
    hids_min=50,
    hids_max=300,
    embedding_net=embedding_net,
    lr=5e-5,
    train_batch_size=200,
    seed_model=201 + SEED,
    seed_train=999 + SEED,
    outdir=OUTDIR,
)

# ----------------------------------------------------------------- diagnose F
with np.load(f"{OUTDIR}/fishnets_outputs.npz") as f:
    Fs = np.asarray(f["Fs"])            # (members, n_test, d, d)
    print(f"\nFs {Fs.shape}  finite members: "
          f"{int(np.isfinite(Fs).all(axis=(1,2,3)).sum())}/{Fs.shape[0]}")
    F = Fs.mean(0)                      # ensemble-mean per-sample Fisher

# Per-sample relative spectrum -- the quantity the rank rule reads. The mean
# Fisher is full rank here even though every per-sample F is rank 1, because the
# informative direction rotates across the prior, so do NOT eigendecompose the
# mean and expect rank 1.
s = np.full(D, cfg.theta_max - cfg.theta_min)
Fn = F * s[None, :, None] * s[None, None, :]
ev = np.linalg.eigvalsh(0.5 * (Fn + np.swapaxes(Fn, -1, -2)))[:, ::-1]
rel = np.abs(ev) / np.maximum(ev[:, :1], 1e-300)
med = np.median(rel, axis=0)
print("\nper-sample MEDIAN relative spectrum (want [1, plateau, plateau, ...]):")
print("  ", np.array2string(med, precision=5))
gaps = med[:-1] / np.maximum(med[1:], 1e-300)
print(f"  largest multiplicative gap: {gaps.max():.1f} after index {gaps.argmax()}"
      f"  -> eigengap rank = {gaps.argmax() + 1}")
print(f"  noise plateau level (median of tail): {np.median(med[1:]):.2e}")
