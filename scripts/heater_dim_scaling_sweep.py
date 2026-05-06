"""Sweep validation log-prob vs ambient dimension for the chain-product heater.

The simulator is a generalisation of the rank-1 heater example to an
arbitrary number of multiplicative parameters:

    y(t) = (theta_1 * theta_2 * ... * theta_d) * (1 - exp(-t / tau)) + noise(t).

The intrinsic dimension of the data manifold is exactly **1** for any d --
only the scalar product P = prod_i theta_i enters the data. This sweep
demonstrates the practical consequence: training NPE-MAF on the raw
d-dimensional theta target degrades with d (curse of dimensionality acting
on a measure-zero manifold), while training on the analytic distilled
1-dimensional target is, by construction, insensitive to d. With a fixed
simulation budget the gap between the two widens exponentially, exactly
as predicted by Stone's minimax bound for nonparametric density
estimation.

Choice of distilled coordinate
------------------------------
The "obvious" distilled target ``eta = prod_i theta_i`` has range
``[theta_min^d, theta_max^d]``, which grows *exponentially* with d. After
the MAF's internal standardisation a fixed posterior width
``sigma_perp ~ sigma / ||k||`` becomes a near-delta in standardised
units, which the flow cannot represent from O(1000) samples. The
resulting distilled curve decays with d for purely numerical reasons
rather than statistical ones, smuggling the curse of dimensionality back
into the figure through the *coordinate* rather than the *intrinsic
dimension*.

The default distilled coordinate here is therefore the **standardised
log-product**

    eta(theta) = (sum_i log theta_i  -  d * mu_log)
                 / sqrt(d * var_log),

where ``mu_log`` and ``var_log`` are the analytic mean and variance of
``log theta`` under the prior. This is a strictly monotone function of
``prod_i theta_i`` (so it carries the same identifiable information),
but its induced prior tends to ``N(0, 1)`` by the CLT and is
d-independent in support and scale. Pass ``--distilled-coord product``
to recover the original (unstabilised) target for comparison.

Defaults match the k=1 heater pipeline (theta in [1, 2], n_t = 20,
nsims = 1000, tau = 1.0, sigma = 0.2). Reasonable runs are produced in a
few minutes per (d, trial) pair on a single GPU and a few times that on
CPU.

Note: we use an *analytic* distilled coordinate rather than re-running
the Distillery for every d. The k=1 experiment (see
``scripts/heater_minimal_distillery.py``) is the proof-of-concept that
the Distillery *recovers* such a coordinate from data; this sweep
quantifies the sample-efficiency gain that follows from using it.

Run from the repository root, e.g.::

    python scripts/heater_dim_scaling_sweep.py \\
        --dims 2 3 4 6 8 10 12 \\
        --num-trials 3 \\
        --out-dir heater_dim_scaling_run

Outputs in ``--out-dir``:

* ``metrics.csv`` / ``metrics.npz``: per-trial best val log-prob.
* ``metrics_aggregate.csv``: mean +/- std across trials, per d.
* ``log_prob_vs_d.{pdf,png}``: the headline figure.
* ``manifest.json``: configuration record (includes the chosen
  distilled coordinate and its analytic moments).

When ``--marginal-eta-eval`` is set, additionally:

* ``metrics.csv`` / ``metrics.npz`` gain ``raw_marg_eta_logprob`` and
  ``distilled_marg_eta_logprob`` columns -- both NPEs evaluated on the
  *same* 1-D marginal density on eta via Gaussian moment matching of
  posterior samples (apples-to-apples comparison; see Sec. "Marginal-on-eta
  evaluation" in the script body).
* ``metrics_aggregate.csv`` gains ``*_marg_eta_{mean,std,sem,n}`` columns.
* ``log_prob_vs_d_marginal_eta.{pdf,png}``: the apples-to-apples figure.
* ``training_histories.npz`` gains per-validation log-prob arrays keyed
  ``raw_marg_eta_arr_<tag>`` / ``distilled_marg_eta_arr_<tag>``.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    import ili
    from ili.dataloaders import NumpyLoader
    from ili.inference import InferenceRunner
except ImportError as exc:  # pragma: no cover - runtime env check
    raise SystemExit(
        "This script requires ltu-ili. In Colab:\n"
        "  pip install -q git+https://github.com/maho3/ltu-ili\n"
        "  pip install -q 'ltu-ili[pytorch]'"
    ) from exc


@dataclass
class ChainHeaterCfg:
    """Configuration for the chain-product heater simulator."""

    theta_min: float = 1.0
    theta_max: float = 2.0
    tau: float = 1.0
    t_max: float = 4.0
    n_t: int = 20
    sigma: float = 0.2

    @property
    def t_grid(self) -> np.ndarray:
        return np.linspace(0.0, self.t_max, self.n_t, dtype=np.float32)

    @property
    def thermal_kernel(self) -> np.ndarray:
        return (1.0 - np.exp(-self.t_grid / self.tau)).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # --- sweep ---
    parser.add_argument(
        "--dims", nargs="+", type=int, default=[2, 3, 4, 6, 8, 10, 12],
        help="Ambient dimensions to sweep. Intrinsic dim is always 1.",
    )
    parser.add_argument("--num-trials", type=int, default=3,
                        help="Independent training runs per dim (for error bars).")
    parser.add_argument("--seed", type=int, default=0)

    # --- simulator (defaults match the working k=1 heater run) ---
    parser.add_argument("--theta-min", type=float, default=1.0)
    parser.add_argument("--theta-max", type=float, default=2.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--t-max", type=float, default=4.0)
    parser.add_argument("--n-t", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--nsims", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)

    # --- NPE training (matches scripts/rosen_nsims_logprob_sweep.py) ---
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-features", type=int, default=50)
    parser.add_argument("--num-transforms", type=int, default=5)
    parser.add_argument("--repeats-maf", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    # --- distilled coordinate ---
    parser.add_argument(
        "--distilled-coord",
        choices=("standardised_log_product", "log_product", "product"),
        default="standardised_log_product",
        help=(
            "Choice of analytic distilled coordinate. "
            "'standardised_log_product' (default): "
            "(sum log theta - d*mu_log) / sqrt(d*var_log) "
            "-- d-invariant N(0, 1) prior by CLT, recommended. "
            "'log_product': sum_i log theta_i -- range scales linearly with d. "
            "'product': prod_i theta_i -- range scales exponentially with d "
            "(reproduces the original numerical pathology, kept for comparison)."
        ),
    )

    # --- apples-to-apples marginal-on-eta evaluation ---
    parser.add_argument(
        "--marginal-eta-eval", action="store_true",
        help=(
            "After training, evaluate both NPEs on the same 1-D marginal "
            "density log p_marg(eta_true | y) via Gaussian moment matching "
            "of posterior samples. Adds an apples-to-apples figure and CSV "
            "columns; ~10-20%% wall-clock surcharge."
        ),
    )
    parser.add_argument(
        "--n-marginal-samples", type=int, default=2000,
        help="Posterior samples per validation observation for the "
             "marginal-on-eta Gaussian fit.",
    )
    parser.add_argument(
        "--n-marginal-val", type=int, default=None,
        help="Number of held-out validation observations used for the "
             "marginal-on-eta evaluation (default: --n-test).",
    )

    # --- I/O ---
    parser.add_argument("--out-dir", type=Path, default=Path("heater_dim_scaling_run"))
    return parser.parse_args()


def chain_dataset(
    n: int, d: int, cfg: ChainHeaterCfg, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (theta, y) for ``y = (prod_i theta_i) * kernel + noise``."""
    theta = rng.uniform(cfg.theta_min, cfg.theta_max,
                        size=(n, d)).astype(np.float32)
    power = np.prod(theta, axis=1, keepdims=False).astype(np.float32)
    mean = power[:, None] * cfg.thermal_kernel
    noise = rng.normal(scale=cfg.sigma, size=mean.shape).astype(np.float32)
    data = (mean + noise).astype(np.float32)
    return theta, data


def log_theta_moments_uniform(a: float, b: float) -> tuple[float, float]:
    """Analytic mean and variance of ``log theta`` for ``theta ~ U[a, b]``.

    Returns ``(mu_log, var_log)``. Requires ``a > 0``.
    """
    if a <= 0.0:
        raise ValueError("log moments require a positive prior lower bound")
    width = b - a
    mu = (b * np.log(b) - b - a * np.log(a) + a) / width
    e_sq = (
        b * np.log(b) ** 2 - 2.0 * b * np.log(b) + 2.0 * b
        - a * np.log(a) ** 2 + 2.0 * a * np.log(a) - 2.0 * a
    ) / width
    var = e_sq - mu ** 2
    return float(mu), float(var)


def compute_distilled_target(
    theta: np.ndarray, coord: str, cfg: ChainHeaterCfg, d: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict[str, float]]:
    """Map ``theta`` to a chosen 1-D distilled target.

    Parameters
    ----------
    theta : array of shape (n, d)
    coord : one of {"product", "log_product", "standardised_log_product"}
    cfg, d : simulator config and current ambient dimension

    Returns
    -------
    eta : (n, 1) float32 array of distilled targets
    low, high : (1,) float32 arrays giving prior bounds for ili.utils.Uniform
    label : LaTeX label for plotting
    info : dict of analytic constants used (``mu_log``, ``var_log``)
    """
    info: dict[str, float] = {}
    if coord == "product":
        eta = np.prod(theta, axis=1, keepdims=True).astype(np.float32)
        low = np.array([cfg.theta_min ** d], dtype=np.float32)
        high = np.array([cfg.theta_max ** d], dtype=np.float32)
        label = r"distilled $\eta = \prod_i \theta_i \in \mathbb{R}$"
        return eta, low, high, label, info

    mu_log, var_log = log_theta_moments_uniform(cfg.theta_min, cfg.theta_max)
    info["mu_log"] = mu_log
    info["var_log"] = var_log
    log_theta_sum = np.log(theta).sum(axis=1, keepdims=True).astype(np.float32)

    if coord == "log_product":
        eta = log_theta_sum
        low = np.array([d * np.log(cfg.theta_min)], dtype=np.float32)
        high = np.array([d * np.log(cfg.theta_max)], dtype=np.float32)
        label = r"distilled $\eta = \sum_i \log \theta_i$"
        return eta, low, high, label, info

    if coord == "standardised_log_product":
        scale = np.sqrt(d * var_log)
        eta = ((log_theta_sum - d * mu_log) / scale).astype(np.float32)
        # +/- 5 sigma envelopes the Gaussian induced prior to numerical zero
        low = np.array([-5.0], dtype=np.float32)
        high = np.array([+5.0], dtype=np.float32)
        label = (
            r"distilled $\eta = (\sum_i \log\theta_i - d\mu_{\log})"
            r"/\sqrt{d\,\sigma^{2}_{\log}}$"
        )
        info["scale"] = float(scale)
        return eta, low, high, label, info

    raise ValueError(f"unknown distilled-coord choice: {coord!r}")


def make_eta_projector(coord: str, cfg: ChainHeaterCfg, d: int):
    """Return a callable that maps theta-samples ``(n, d)`` to ``(n,)`` eta.

    Uses the *same* analytic transform as ``compute_distilled_target`` so the
    raw NPE's posterior samples can be projected onto the same axis the
    distilled NPE was trained on. Constants are computed once and reused.
    """
    if coord == "product":
        return lambda theta: np.prod(theta, axis=1).astype(np.float32)
    mu_log, var_log = log_theta_moments_uniform(cfg.theta_min, cfg.theta_max)
    if coord == "log_product":
        return lambda theta: np.log(theta).sum(axis=1).astype(np.float32)
    if coord == "standardised_log_product":
        scale = float(np.sqrt(d * var_log))
        offset = float(d * mu_log)
        return lambda theta: ((np.log(theta).sum(axis=1) - offset) / scale).astype(np.float32)
    raise ValueError(f"unknown distilled-coord choice: {coord!r}")


def marginal_eta_log_probs(
    posterior: Any,
    x_val: np.ndarray,
    eta_true: np.ndarray,
    eta_projector,
    n_samples: int,
    device: str,
    *,
    progress: bool = True,
) -> np.ndarray:
    """Per-validation Gaussian-fit log p_marg(eta_true | y) from posterior samples.

    For each ``y_i`` in ``x_val``:

        1. Sample ``n_samples`` parameters from the trained posterior at ``y_i``.
        2. Project them through ``eta_projector`` to scalar etas.
           For the *distilled* NPE pass ``eta_projector = lambda s: s.reshape(-1)``;
           for the *raw* NPE pass the result of ``make_eta_projector``.
        3. Fit Gaussian moments ``(mu_i, sigma_i^2)`` to the eta-samples and
           return ``log N(eta_true_i | mu_i, sigma_i^2)``.

    The Gaussian fit is the natural Bayes-optimal estimator for this simulator:
    the conditional likelihood ``p(y | eta)`` is exactly Gaussian (linear in
    eta plus Gaussian noise), and the standardised-log-product prior is
    near-Gaussian by CLT, so the marginal posterior on eta is well-approximated
    by a Gaussian for any d.
    """
    n_val = int(x_val.shape[0])
    log_probs = np.empty(n_val, dtype=np.float32)
    x_val_t = torch.as_tensor(x_val.astype(np.float32), device=device)
    iterator = range(n_val)
    if progress:
        try:
            from tqdm import tqdm  # local import: optional dependency
            iterator = tqdm(iterator, desc="marg-eta", leave=False)
        except ImportError:
            pass

    log_2pi = float(np.log(2.0 * np.pi))
    with torch.no_grad():
        for i in iterator:
            x_i = x_val_t[i]
            try:
                samples = posterior.sample(
                    (n_samples,), x=x_i, show_progress_bars=False,
                )
            except TypeError:
                samples = posterior.sample((n_samples,), x=x_i)
            samples_np = samples.detach().cpu().numpy().astype(np.float32)
            if samples_np.ndim == 1:
                samples_np = samples_np[:, None]
            eta_samples = np.asarray(eta_projector(samples_np), dtype=np.float64)
            mu = float(eta_samples.mean())
            var = float(eta_samples.var(ddof=0)) + 1e-12
            log_probs[i] = -0.5 * (log_2pi + np.log(var)) \
                           - 0.5 * (float(eta_true[i]) - mu) ** 2 / var
    return log_probs


def make_runner(
    low: np.ndarray, high: np.ndarray, args: argparse.Namespace, device: str,
) -> InferenceRunner:
    prior = ili.utils.Uniform(low=low.tolist(), high=high.tolist(), device=device)
    nets = [
        ili.utils.load_nde_lampe(
            engine="NPE", model="maf",
            hidden_features=args.hidden_features,
            num_transforms=args.num_transforms,
            repeats=args.repeats_maf,
        )
    ]
    train_args = {
        "training_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_num_epochs": args.epochs,
        "stop_after_epochs": args.patience,
    }
    return InferenceRunner.load(
        backend="lampe", engine="NPE",
        prior=prior, nets=nets, device=device,
        train_args=train_args, proposal=None, out_dir=None,
    )


def train_npe(
    theta: np.ndarray, data: np.ndarray,
    low: np.ndarray, high: np.ndarray,
    args: argparse.Namespace, device: str, seed: int,
) -> tuple[float, np.ndarray, np.ndarray, Any]:
    """Train an NPE-MAF.

    Returns
    -------
    best_val_log_prob : float
    train_log_probs   : per-epoch training log-prob curve
    val_log_probs     : per-epoch validation log-prob curve
    posterior         : the trained posterior object (for downstream sampling)
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    runner = make_runner(low, high, args, device)
    loader = NumpyLoader(x=data.astype(np.float32),
                         theta=theta.astype(np.float32))
    posterior, summaries = runner(loader=loader)
    val_log_probs = np.asarray(summaries[0]["validation_log_probs"])
    train_log_probs = np.asarray(summaries[0]["training_log_probs"])
    return float(np.max(val_log_probs)), train_log_probs, val_log_probs, posterior


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    cfg = ChainHeaterCfg(
        theta_min=args.theta_min, theta_max=args.theta_max,
        tau=args.tau, t_max=args.t_max, n_t=args.n_t, sigma=args.sigma,
    )

    rows: list[dict[str, Any]] = []
    histories: dict[str, np.ndarray] = {}
    distilled_label: str = ""
    distilled_info: dict[str, float] = {}

    print(f"distilled coordinate: {args.distilled_coord}")
    if args.distilled_coord != "product":
        mu_log, var_log = log_theta_moments_uniform(cfg.theta_min, cfg.theta_max)
        print(f"  prior moments: mu_log = {mu_log:.6f}, "
              f"var_log = {var_log:.6f}, sigma_log = {np.sqrt(var_log):.6f}")
    if args.marginal_eta_eval:
        n_marg_val = int(args.n_marginal_val or args.n_test)
        print(f"marginal-eta-eval: ON  ({n_marg_val} val obs, "
              f"{args.n_marginal_samples} posterior samples / obs)")

    for d in args.dims:
        for trial in range(args.num_trials):
            seed = args.seed + 7919 * trial + 31 * d
            tag = f"d{d}_trial{trial}"
            print(f"\n=== {tag} (seed {seed}) ===")
            rng = np.random.default_rng(seed)

            theta_tr, data_tr = chain_dataset(args.nsims, d, cfg, rng)
            # raw NPE: target is theta in R^d
            low_raw = np.full(d, cfg.theta_min, dtype=np.float32)
            high_raw = np.full(d, cfg.theta_max, dtype=np.float32)
            print(f"[raw      ] target dim = {d}, training NPE-MAF...")
            raw_lp, raw_train, raw_val, raw_posterior = train_npe(
                theta_tr, data_tr, low_raw, high_raw, args, device, seed,
            )

            # distilled NPE: target is a 1-D analytic identifiable coord
            eta_tr, low_eta, high_eta, distilled_label, distilled_info = (
                compute_distilled_target(theta_tr, args.distilled_coord, cfg, d)
            )
            print(f"[distilled] target dim = 1 ({args.distilled_coord}), "
                  f"range [{float(low_eta[0]):.4g}, {float(high_eta[0]):.4g}], "
                  f"training NPE-MAF...")
            dist_lp, dist_train, dist_val, dist_posterior = train_npe(
                eta_tr, data_tr, low_eta, high_eta, args, device, seed + 1,
            )

            print(f"  raw       best val log_prob = {raw_lp:.4f}  "
                  f"(target dim {d})")
            print(f"  distilled best val log_prob = {dist_lp:.4f}  "
                  f"(target dim 1)")

            row: dict[str, Any] = {
                "d": d, "trial": trial, "seed": seed,
                "raw_log_prob": raw_lp,
                "distilled_log_prob": dist_lp,
            }

            if args.marginal_eta_eval:
                # held-out evaluation set, shared between the two NPEs
                marg_rng = np.random.default_rng(seed + 999_983)
                theta_eval, data_eval = chain_dataset(
                    int(args.n_marginal_val or args.n_test), d, cfg, marg_rng,
                )
                eta_true_eval, _, _, _, _ = compute_distilled_target(
                    theta_eval, args.distilled_coord, cfg, d,
                )
                eta_true_eval = eta_true_eval.reshape(-1).astype(np.float32)

                eta_proj_raw = make_eta_projector(args.distilled_coord, cfg, d)
                # distilled NPE samples are already 1-D in eta-coords
                eta_proj_dist = lambda s: np.asarray(s, dtype=np.float32).reshape(-1)

                print(f"[marg-eta] evaluating raw NPE...")
                raw_marg_arr = marginal_eta_log_probs(
                    raw_posterior, data_eval, eta_true_eval, eta_proj_raw,
                    args.n_marginal_samples, device,
                )
                print(f"[marg-eta] evaluating distilled NPE...")
                dist_marg_arr = marginal_eta_log_probs(
                    dist_posterior, data_eval, eta_true_eval, eta_proj_dist,
                    args.n_marginal_samples, device,
                )
                row["raw_marg_eta_logprob"] = float(np.mean(raw_marg_arr))
                row["distilled_marg_eta_logprob"] = float(np.mean(dist_marg_arr))
                row["raw_marg_eta_se"] = float(
                    np.std(raw_marg_arr, ddof=1) / np.sqrt(len(raw_marg_arr))
                )
                row["distilled_marg_eta_se"] = float(
                    np.std(dist_marg_arr, ddof=1) / np.sqrt(len(dist_marg_arr))
                )
                histories[f"raw_marg_eta_arr_{tag}"] = raw_marg_arr
                histories[f"distilled_marg_eta_arr_{tag}"] = dist_marg_arr
                print(f"  raw       <log p_marg(eta|y)> = "
                      f"{row['raw_marg_eta_logprob']:.4f}")
                print(f"  distilled <log p_marg(eta|y)> = "
                      f"{row['distilled_marg_eta_logprob']:.4f}")

            rows.append(row)
            histories[f"raw_train_{tag}"] = raw_train
            histories[f"raw_val_{tag}"] = raw_val
            histories[f"distilled_train_{tag}"] = dist_train
            histories[f"distilled_val_{tag}"] = dist_val

            # release posteriors before next iteration to keep GPU memory flat
            del raw_posterior, dist_posterior
            if device == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False)

    npz_payload = {
        "d": df["d"].to_numpy(),
        "trial": df["trial"].to_numpy(),
        "seed": df["seed"].to_numpy(),
        "raw_log_prob": df["raw_log_prob"].to_numpy(),
        "distilled_log_prob": df["distilled_log_prob"].to_numpy(),
    }
    if args.marginal_eta_eval:
        npz_payload["raw_marg_eta_logprob"] = df["raw_marg_eta_logprob"].to_numpy()
        npz_payload["distilled_marg_eta_logprob"] = df["distilled_marg_eta_logprob"].to_numpy()
        npz_payload["raw_marg_eta_se"] = df["raw_marg_eta_se"].to_numpy()
        npz_payload["distilled_marg_eta_se"] = df["distilled_marg_eta_se"].to_numpy()
    np.savez(out_dir / "metrics.npz", **npz_payload)
    np.savez(out_dir / "training_histories.npz", **histories)

    agg_specs: dict[str, tuple[str, str]] = {
        "raw_mean": ("raw_log_prob", "mean"),
        "raw_std": ("raw_log_prob", "std"),
        "raw_n": ("raw_log_prob", "count"),
        "dist_mean": ("distilled_log_prob", "mean"),
        "dist_std": ("distilled_log_prob", "std"),
        "dist_n": ("distilled_log_prob", "count"),
    }
    if args.marginal_eta_eval:
        agg_specs.update({
            "raw_marg_eta_mean": ("raw_marg_eta_logprob", "mean"),
            "raw_marg_eta_std": ("raw_marg_eta_logprob", "std"),
            "raw_marg_eta_n": ("raw_marg_eta_logprob", "count"),
            "dist_marg_eta_mean": ("distilled_marg_eta_logprob", "mean"),
            "dist_marg_eta_std": ("distilled_marg_eta_logprob", "std"),
            "dist_marg_eta_n": ("distilled_marg_eta_logprob", "count"),
        })

    agg = df.groupby("d").agg(**agg_specs).reset_index()
    agg["raw_sem"] = agg["raw_std"] / np.sqrt(np.maximum(agg["raw_n"], 1))
    agg["dist_sem"] = agg["dist_std"] / np.sqrt(np.maximum(agg["dist_n"], 1))
    if args.marginal_eta_eval:
        agg["raw_marg_eta_sem"] = (
            agg["raw_marg_eta_std"] / np.sqrt(np.maximum(agg["raw_marg_eta_n"], 1))
        )
        agg["dist_marg_eta_sem"] = (
            agg["dist_marg_eta_std"] / np.sqrt(np.maximum(agg["dist_marg_eta_n"], 1))
        )
    agg.to_csv(out_dir / "metrics_aggregate.csv", index=False)

    print("\n=== Aggregate (mean +/- sem) ===")
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    try:  # pragma: no cover - plotting is best-effort
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
        ax.errorbar(
            agg["d"], agg["raw_mean"], yerr=agg["raw_sem"],
            marker="s", capsize=3, label=r"raw $\theta\in\mathbb{R}^{d}$",
        )
        ax.errorbar(
            agg["d"], agg["dist_mean"], yerr=agg["dist_sem"],
            marker="o", capsize=3,
            label=distilled_label or "distilled (1-D)",
        )
        ax.set_xlabel(r"ambient dim $d$  (intrinsic dim $= 1$)")
        ax.set_ylabel(r"best validation $\log p(\,\cdot\,|\,y)$")
        ax.set_title(
            r"Heater chain-product: NPE-MAF val log-prob vs ambient dim"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(out_dir / "log_prob_vs_d.pdf")
        fig.savefig(out_dir / "log_prob_vs_d.png", dpi=200)
        print(f"\nSaved figure to {out_dir / 'log_prob_vs_d.pdf'}")

        if args.marginal_eta_eval:
            fig2, ax2 = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
            ax2.errorbar(
                agg["d"], agg["raw_marg_eta_mean"], yerr=agg["raw_marg_eta_sem"],
                marker="s", capsize=3,
                label=r"raw $\theta$, projected onto $\eta$",
            )
            ax2.errorbar(
                agg["d"], agg["dist_marg_eta_mean"], yerr=agg["dist_marg_eta_sem"],
                marker="o", capsize=3,
                label=r"distilled (native $\eta$)",
            )
            ax2.set_xlabel(r"ambient dim $d$  (intrinsic dim $= 1$)")
            ax2.set_ylabel(
                r"$\langle\,\log\hat{p}_{\mathrm{marg}}(\eta_{\mathrm{true}}\mid y)\,\rangle$"
            )
            ax2.set_title(
                "Heater chain-product: marginal log-prob on the same $\\eta$ axis"
            )
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            fig2.savefig(out_dir / "log_prob_vs_d_marginal_eta.pdf")
            fig2.savefig(out_dir / "log_prob_vs_d_marginal_eta.png", dpi=200)
            print(f"Saved figure to {out_dir / 'log_prob_vs_d_marginal_eta.pdf'}")
    except Exception as e:
        print(f"plot failed: {e}")

    manifest = {
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "cfg": asdict(cfg),
        "distilled_coord": args.distilled_coord,
        "distilled_info": distilled_info,
        "distilled_label_latex": distilled_label,
        "marginal_eta_eval": bool(args.marginal_eta_eval),
        "files": {
            "metrics.csv": "per-trial val log-prob for raw and distilled NPE "
                           "(plus *_marg_eta_logprob columns when --marginal-eta-eval)",
            "metrics.npz": "same as metrics.csv in numpy format",
            "metrics_aggregate.csv": "mean/std/sem across trials, per d",
            "training_histories.npz": "full train/val curves keyed by 'raw_*'/'distilled_*' "
                                      "(plus '*_marg_eta_arr_*' per-validation arrays)",
            "log_prob_vs_d.pdf": "headline figure: best val log-prob vs ambient dim",
            "log_prob_vs_d_marginal_eta.pdf":
                "apples-to-apples figure: marginal log-prob on the same eta axis "
                "(only present when --marginal-eta-eval)",
        },
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
