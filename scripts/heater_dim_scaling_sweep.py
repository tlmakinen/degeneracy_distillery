"""Sweep validation log-prob vs ambient dimension for the chain-product heater.

The simulator is a generalisation of the rank-1 heater example to an
arbitrary number of multiplicative parameters:

    y(t) = (theta_1 * theta_2 * ... * theta_d) * (1 - exp(-t / tau)) + noise(t).

The intrinsic dimension of the data manifold is exactly **1** for any d --
only the scalar product P = prod_i theta_i enters the data. This sweep
demonstrates the practical consequence: training NPE-MAF on the raw
d-dimensional theta target degrades with d (curse of dimensionality acting
on a measure-zero manifold), while training on the analytic distilled
1-dimensional target eta = prod_i theta_i is, by construction, insensitive
to d. With a fixed simulation budget the gap between the two widens
exponentially, exactly as predicted by Stone's minimax bound for
nonparametric density estimation.

Defaults match the k=1 heater pipeline (theta in [1, 2], n_t = 20,
nsims = 1000, tau = 1.0, sigma = 0.2). Reasonable runs are produced in a
few minutes per (d, trial) pair on a single GPU and a few times that on
CPU.

Note: we use the *analytic* distilled coordinate eta = prod_i theta_i
rather than re-running the Distillery for every d. The k=1 experiment
(see scripts/heater_minimal_distillery.py) is the proof-of-concept that
the Distillery *recovers* this coordinate from data; this sweep
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
* ``manifest.json``: configuration record.
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
) -> tuple[float, np.ndarray, np.ndarray]:
    """Train an NPE-MAF and return (best val log_prob, train_curve, val_curve)."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    runner = make_runner(low, high, args, device)
    loader = NumpyLoader(x=data.astype(np.float32),
                         theta=theta.astype(np.float32))
    _, summaries = runner(loader=loader)
    val_log_probs = np.asarray(summaries[0]["validation_log_probs"])
    train_log_probs = np.asarray(summaries[0]["training_log_probs"])
    return float(np.max(val_log_probs)), train_log_probs, val_log_probs


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
            raw_lp, raw_train, raw_val = train_npe(
                theta_tr, data_tr, low_raw, high_raw, args, device, seed,
            )

            # distilled NPE: target is eta = prod_i theta_i in R^1
            eta_tr = np.prod(theta_tr, axis=1, keepdims=True).astype(np.float32)
            low_eta = np.array([cfg.theta_min ** d], dtype=np.float32)
            high_eta = np.array([cfg.theta_max ** d], dtype=np.float32)
            print(f"[distilled] target dim = 1, training NPE-MAF...")
            dist_lp, dist_train, dist_val = train_npe(
                eta_tr, data_tr, low_eta, high_eta, args, device, seed + 1,
            )

            print(f"  raw       best val log_prob = {raw_lp:.4f}  "
                  f"(target dim {d})")
            print(f"  distilled best val log_prob = {dist_lp:.4f}  "
                  f"(target dim 1)")

            rows.append({
                "d": d, "trial": trial, "seed": seed,
                "raw_log_prob": raw_lp,
                "distilled_log_prob": dist_lp,
            })
            histories[f"raw_train_{tag}"] = raw_train
            histories[f"raw_val_{tag}"] = raw_val
            histories[f"distilled_train_{tag}"] = dist_train
            histories[f"distilled_val_{tag}"] = dist_val

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False)
    np.savez(out_dir / "metrics.npz",
             d=df["d"].to_numpy(),
             trial=df["trial"].to_numpy(),
             seed=df["seed"].to_numpy(),
             raw_log_prob=df["raw_log_prob"].to_numpy(),
             distilled_log_prob=df["distilled_log_prob"].to_numpy())
    np.savez(out_dir / "training_histories.npz", **histories)

    agg = (
        df.groupby("d")
          .agg(
              raw_mean=("raw_log_prob", "mean"),
              raw_std=("raw_log_prob", "std"),
              raw_n=("raw_log_prob", "count"),
              dist_mean=("distilled_log_prob", "mean"),
              dist_std=("distilled_log_prob", "std"),
              dist_n=("distilled_log_prob", "count"),
          )
          .reset_index()
    )
    # sem (standard error of the mean) for error bars
    agg["raw_sem"] = agg["raw_std"] / np.sqrt(np.maximum(agg["raw_n"], 1))
    agg["dist_sem"] = agg["dist_std"] / np.sqrt(np.maximum(agg["dist_n"], 1))
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
            label=r"distilled $\eta=\prod_i \theta_i\in\mathbb{R}$",
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
    except Exception as e:
        print(f"plot failed: {e}")

    manifest = {
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "cfg": asdict(cfg),
        "files": {
            "metrics.csv": "per-trial val log-prob for raw and distilled NPE",
            "metrics.npz": "same as metrics.csv in numpy format",
            "metrics_aggregate.csv": "mean/std/sem across trials, per d",
            "training_histories.npz": "full train/val curves keyed by 'raw_*'/'distilled_*'",
            "log_prob_vs_d.pdf": "headline figure: log-prob vs ambient dim",
        },
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
