"""Hidden rank-2 Rosenbrock banana in ``d`` unused coordinates.

The informative pair ``(I0, I1)`` is drawn from the trial seed. The
observation is eight noisy replicates of the banana mean. This adapter
also owns the three-step comparison arm. The driver skips that arm when
``d > 16``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

A_BOX = 3.0
N_REP = 8
NOISE = (0.25, 0.5)
THREESTEP_D_CAP = 16


def poly_r2(y: np.ndarray, feats: np.ndarray, deg: int = 3) -> float:
    y = np.asarray(y, dtype=np.float64).ravel()
    f = np.asarray(feats, dtype=np.float64)
    if f.ndim == 1:
        f = f[:, None]
    f = (f - f.mean(0)) / (f.std(0) + 1e-12)
    k = f.shape[1]
    cols = [f[:, j] for j in range(k)]
    if deg >= 2:
        cols += [f[:, i] * f[:, j] for i in range(k) for j in range(i, k)]
    if deg >= 3:
        cols += [
            f[:, i] * f[:, j] * f[:, l]
            for i in range(k)
            for j in range(i, k)
            for l in range(j, k)
        ]
    xf = np.column_stack(cols + [np.ones(len(y))])
    beta, *_ = np.linalg.lstsq(xf, y, rcond=None)
    return float(1.0 - ((y - xf @ beta) ** 2).mean() / (y.var() + 1e-12))


_MeanPoolEstimator = None


def _mean_pool_estimator(m: int, features: Sequence[int] = (32, 32)):
    global _MeanPoolEstimator
    if _MeanPoolEstimator is None:
        import flax.linen as nn

        class MeanPoolEstimator(nn.Module):
            m: int
            features: Sequence[int] = (32, 32)

            @nn.compact
            def __call__(self, x):
                h = x.mean(0) if x.ndim >= 2 else x
                skip = h
                for w in self.features:
                    h = nn.gelu(nn.Dense(w)(h))
                return (
                    nn.Dense(
                        self.m,
                        kernel_init=nn.initializers.zeros,
                        bias_init=nn.initializers.zeros,
                    )(h)
                    + nn.Dense(self.m)(skip)
                )

        _MeanPoolEstimator = MeanPoolEstimator
    return _MeanPoolEstimator(m=int(m), features=tuple(features))


class RosenbrockProblem:
    """Rank-2 banana hidden in ``d`` coordinates. ``use_log_features=False``."""

    def __init__(self, d: int = 2, seed: int = 0, hidden=None, **_ignored):
        self.d = int(d)
        self.name = "rosenbrock"
        self.param_names = tuple(f"theta_{i}" for i in range(self.d))
        self.theta_lo = np.full(self.d, -A_BOX, dtype=np.float64)
        self.theta_hi = np.full(self.d, A_BOX, dtype=np.float64)
        self.use_log_features = False
        self.m_probe = 4
        self.expected_rank = 2
        if hidden is not None:
            self.I0, self.I1 = int(hidden[0]), int(hidden[1])
        elif self.d == 2:
            self.I0, self.I1 = 0, 1
        else:
            rng = np.random.default_rng(1000 + self.d + int(seed))
            self.I0, self.I1 = (int(v) for v in rng.choice(self.d, size=2, replace=False))

    def sample(self, n: int, rng: np.random.Generator):
        th = rng.uniform(-A_BOX, A_BOX, size=(int(n), self.d))
        mu = np.stack(
            [th[:, self.I0], th[:, self.I1] - th[:, self.I0] ** 2],
            axis=1,
        )
        sd = np.asarray(NOISE, dtype=float)
        x = mu[:, None, :] + rng.normal(size=(int(n), N_REP, 2)) * sd
        return th, x

    def sample_prior(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(-A_BOX, A_BOX, size=(int(n), self.d))

    def estimator_factory(self, m: int):
        return _mean_pool_estimator(m)

    def truth_coords(self, theta: np.ndarray) -> np.ndarray:
        th = np.asarray(theta)
        return np.stack(
            [th[:, self.I0], th[:, self.I1] - th[:, self.I0] ** 2],
            axis=1,
        )

    def gates(self, payload: dict) -> dict[str, bool]:
        r_hat = int(payload.get("r_hat", -1))
        order = list(payload.get("screen_order", []))
        top2 = set(int(v) for v in order[:2]) if len(order) >= 2 else set()
        hidden = {self.I0, self.I1}
        return {
            "rank_correct": bool(r_hat == 2),
            "screen_set_correct": bool(top2 == hidden),
        }

    def sr_hints(self) -> dict:
        return {
            "allowed_symbols": "add,mul,div,pow,constant,variable,sqrt",
            "max_length": max(25, 4 * self.d + 8),
            "max_depth": max(10, self.d + 4),
        }

    def threestep_arm(
        self,
        th,
        x,
        th_te,
        x_te,
        *,
        seed: int,
        outdir: Path,
        num_fishnets: int = 3,
        fish_epochs: int = 250,
        flatten_epochs: int = 250,
        fish_eig_floor: float = 1e-2,
    ) -> dict:
        """Fishnets + flatten. Skipped by the driver when ``d > 16``."""
        if self.d > THREESTEP_D_CAP:
            return {"status": "skipped_by_budget", "r_hat": None}

        import jax
        import jax.numpy as jnp
        import os

        from degeneracy_distillery.sr_utils import fit_theta_scaler
        from degeneracy_distillery.training_loop_fishnets import train_fishnets
        from degeneracy_distillery.training_loop_flatten import fit_flattening

        th = np.asarray(th)
        x = np.asarray(x)
        th_te = np.asarray(th_te)
        x_te = np.asarray(x_te)
        d = th.shape[1]
        x_flat = x.reshape(x.shape[0], -1)
        x_te_flat = x_te.reshape(x_te.shape[0], -1)
        scaler = fit_theta_scaler(th, feature_range=(-3.0, 3.0))
        th_s = scaler.transform(th).astype(np.float32)
        th_te_s = scaler.transform(th_te).astype(np.float32)

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        fish_dir = outdir / f"fishnets_{self.name}_d{d}"
        train_fishnets(
            th_s,
            x_flat.astype(np.float32),
            th_te_s,
            x_te_flat.astype(np.float32),
            num_models=int(num_fishnets),
            hids_min=24,
            hids_max=64,
            n_layers=[2, 3],
            train_epochs=int(fish_epochs),
            train_min_epochs=min(80, int(fish_epochs)),
            patience=20,
            train_batch_size=min(25, th.shape[0]),
            lr=5e-5,
            seed_model=int(seed) + 201,
            seed_train=int(seed) + 999,
            outdir=str(fish_dir),
            update_pbar_every=50,
        )
        fish = np.load(fish_dir / "fishnets_outputs.npz")
        thetas = jnp.array(fish["theta"])
        fs_np = np.asarray(fish["Fs"])
        wts = np.asarray(fish["ensemble_weights"])
        finite = np.isfinite(fs_np).all(axis=(1, 2, 3))
        if not finite.any():
            raise RuntimeError("all fishnet members produced non-finite Fishers")
        fs_np, wts = fs_np[finite], wts[finite]
        f_mean = np.average(fs_np.mean(1), axis=0, weights=wts)
        f_eigs = np.sort(np.linalg.eigvalsh(0.5 * (f_mean + f_mean.T)))[::-1]
        f_eigs = np.maximum(f_eigs, 0.0)
        rel = f_eigs / (f_eigs[0] + 1e-12)
        r_fish = int(np.sum(rel > fish_eig_floor))

        cwd = Path.cwd()
        os.chdir(outdir)
        try:
            w, ensemble_w, _, flatten_model = fit_flattening(
                jnp.array(fs_np),
                thetas,
                ensemble_weights=wts,
                hidden_size=64,
                n_layers=3,
                batch_size=min(50, th.shape[0]),
                epochs_phase1=int(flatten_epochs),
                epochs_phase2=int(flatten_epochs),
                finetune_epochs=max(50, int(flatten_epochs) // 4),
                min_epochs=min(80, int(flatten_epochs)),
                patience=30,
                lr_phase1=1e-6,
                lr_schedule_initial=7e-5,
                lr_decay=0.3,
                lr_finetune=4e-6,
                Fisher_to_flatten="average",
                norm_factor=None,
                norm_method="median_det",
                flattener_activation="softplus",
                noise=1e-4,
                seed=int(seed),
                output_prefix=f"flatten_{self.name}_d{d}",
                use_whitening=True,
                nn_inv=False,
                forward_backward_mlp=True,
                l1_alpha=0.0,
                do_plot=False,
                return_model=True,
                save_flatten_model_pickle=False,
                update_pbar_every=50,
            )
        finally:
            os.chdir(cwd)

        th_te_j = jnp.asarray(th_te_s)
        etas = [
            np.asarray(jax.vmap(lambda t: flatten_model.apply(w_i, t))(th_te_j))
            for w_i in ensemble_w
        ]
        eta = np.average(np.stack(etas, 0), 0, weights=wts)
        truth = self.truth_coords(th_te)
        r2s = [poly_r2(truth[:, k], eta) for k in range(truth.shape[1])]
        n_fisher = int(d * (d + 1) / 2)
        return {
            "status": "ok",
            "r_hat": int(r_fish),
            "rank_correct": bool(r_fish == 2),
            "r2_true_min": float(np.min(r2s)),
            "r2_true_mean": float(np.mean(r2s)),
            "n_outputs": n_fisher,
            "n_finite_fishnets": int(finite.sum()),
            "jtj_eigengap_rank": int(r_fish),
        }
