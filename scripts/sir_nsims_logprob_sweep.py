"""Train SIR NPEs in theta and eta coordinates over a simulation-count sweep.

This script is adapted from ``notebooks/sir_sampling.ipynb`` and is intended to
be run as a command-line job, for example in Google Colab:

    python sir_nsims_logprob_sweep.py --out-dir sir_sweep_results

It saves:
  * ``metrics.csv``: best validation log_prob by nsims, method, and ensemble member.
  * ``metrics.npz``: the same metrics table as NumPy arrays.
  * ``metrics_aggregate.csv``: mean/std best validation log_prob by nsims and method.
  * ``metrics_aggregate.npz``: the same aggregate table as NumPy arrays.
  * ``fom_comparison.csv/.npz``: theta-vs-eta FoM comparison for the target nsims case.
  * ``training_histories.npz``: full train/validation curves for every run.
  * ``posterior_samples.npz``: seed-matched posterior samples for example test cases.
  * ``manifest.json``: configuration and file descriptions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.integrate import odeint
from tqdm import tqdm

try:
    import ili
    from ili.dataloaders import NumpyLoader
    from ili.inference import InferenceRunner
except ImportError as exc:  # pragma: no cover - this is a runtime environment check.
    raise SystemExit(
        "This script requires ltu-ili. In Colab, install it with:\n"
        "  pip install -q git+https://github.com/maho3/ltu-ili\n"
        "  pip install -q 'ltu-ili[pytorch]'"
    ) from exc


THETA_LABELS = (r"$\beta$", r"$\gamma$", r"$I_0 / 10$")
ETA_LABELS = (r"$\eta_0$", r"$\eta_1$", r"$\eta_2$")

THETA_PRIOR_LOW = np.array([0.1, 0.05, 0.0], dtype=np.float32)
THETA_PRIOR_HIGH = np.array([1.0, 0.5, 5.0], dtype=np.float32)

DEFAULT_THETA_TO_ETA_EXPRS = (
    "(-0.061*X1 + 0.823*X2 + 0.072)/(0.434*X1 + 0.923*sqrt(X2))",
    "0.713*sqrt(X2)",
    "X3",
)
DEFAULT_ETA_TO_THETA_EXPRS = (
    "(1.6189*X2**2 - 1.2945*X1*X2 + 0.072)/(0.434*X1 + 0.061)",
    "1.9671*X2**2",
    "X3",
)


@dataclass
class SirConfig:
    n_pop: int = 1000
    i0_mean: float = 8.0
    r0_init: float = 0.0
    n_timepoints: int = 30
    t_max: float = 50.0
    noise_std: float = 0.05
    delta: float = 0.15
    beta_min: float = 0.1
    beta_max: float = 1.0
    gamma_min: float = 0.05
    gamma_max: float = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep SIR NPE validation log_prob versus number of simulations."
    )
    parser.add_argument("--nsims", nargs="+", type=int, default=[100, 500, 1000, 5000, 10000])
    parser.add_argument("--out-dir", type=Path, default=Path("sir_nsims_logprob_sweep"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument(
        "--training-data",
        type=Path,
        default=None,
        help=(
            "Optional path to an .npz file holding a precomputed training pool, e.g. the "
            "concatenated train+test sets from the distillery. The script reads (theta, "
            "data) from the file using the first matching key pair among "
            "('theta','data'), ('theta_train','data_train'), or "
            "('theta_combined','data_combined'). Theta must already be in the script's "
            "3-D form (beta, gamma, I0/10). For nsims <= file size, MAF training data is "
            "taken from the file (first nsims rows). For max(nsims) > file size, "
            "additional sims are generated to fill the deficit. When this flag is not "
            "given, the legacy generate-everything path is used."
        ),
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.05,
        help="SIR observation noise std added to I(t)/N (default: 0.05, matches legacy notebook).",
    )
    parser.add_argument("--n-examples", type=int, default=8)
    parser.add_argument("--n-posterior-samples", type=int, default=10000)
    parser.add_argument(
        "--fom-nsims",
        type=int,
        default=1000,
        help="Simulation count at which to save the theta-vs-eta FoM comparison.",
    )
    parser.add_argument(
        "--run-coverage",
        action="store_true",
        help="Optionally export posterior coverage diagnostics for the coverage nsims case.",
    )
    parser.add_argument(
        "--coverage-nsims",
        type=int,
        default=None,
        help="Simulation count for coverage export; defaults to --fom-nsims.",
    )
    parser.add_argument(
        "--coverage-n-test",
        type=int,
        default=1000,
        help="Number of held-out test observations for coverage export.",
    )
    parser.add_argument(
        "--coverage-num-samples",
        type=int,
        default=1000,
        help="Posterior samples per test observation for coverage export.",
    )
    parser.add_argument(
        "--coverage-seed",
        type=int,
        default=None,
        help="Seed for selecting coverage test observations; defaults to seed + 20000.",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--validation-smoothing-window",
        type=int,
        default=10,
        help="Moving-average window for selecting the best validation epoch; set to 1 to disable smoothing.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--repeats-maf", type=int, default=2)
    parser.add_argument("--hidden-features", type=int, default=50)
    parser.add_argument("--num-transforms", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--theta-to-eta",
        nargs=3,
        default=DEFAULT_THETA_TO_ETA_EXPRS,
        metavar=("ETA0", "ETA1", "ETA2"),
        help="Three coordinate expressions using X1, X2, X3 for theta=(beta,gamma,I0/10).",
    )
    parser.add_argument(
        "--eta-to-theta",
        nargs=3,
        default=DEFAULT_ETA_TO_THETA_EXPRS,
        metavar=("THETA0", "THETA1", "THETA2"),
        help="Three inverse expressions using X1, X2, X3 for eta coordinates.",
    )
    parser.add_argument(
        "--jacobian-correction-samples",
        type=int,
        default=512,
        help="Number of eta training points used to estimate the mean log|dtheta/deta|.",
    )
    return parser.parse_args()


def make_expression_transform(expressions: tuple[str, str, str] | list[str]):
    """Build a NumPy vectorized transform from expressions in variables X1, X2, X3."""
    allowed_names = {
        "abs": np.abs,
        "arccos": np.arccos,
        "arcsin": np.arcsin,
        "arctan": np.arctan,
        "cos": np.cos,
        "exp": np.exp,
        "log": np.log,
        "maximum": np.maximum,
        "minimum": np.minimum,
        "pi": np.pi,
        "sin": np.sin,
        "sqrt": np.sqrt,
        "tan": np.tan,
    }
    compiled = [compile(expr, f"<coordinate:{idx}>", "eval") for idx, expr in enumerate(expressions)]

    def transform(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        local_names = {"X1": x[:, 0], "X2": x[:, 1], "X3": x[:, 2]}
        cols = [
            eval(code, {"__builtins__": {}}, allowed_names | local_names)
            for code in compiled
        ]
        return np.column_stack(cols).astype(np.float32)

    return transform


def sir_odes(y: np.ndarray, _t: np.ndarray, beta: float, gamma: float, n_pop: int) -> list[float]:
    s, i, r = y
    dsdt = -beta * s * i / n_pop
    didt = beta * s * i / n_pop - gamma * i
    drdt = gamma * i
    return [dsdt, didt, drdt]


def simulate_sir(
    beta: float,
    gamma: float,
    i0: float,
    rng: np.random.Generator,
    t_obs: np.ndarray,
    cfg: SirConfig,
) -> np.ndarray:
    s0 = cfg.n_pop - i0
    y0 = [s0, i0, cfg.r0_init]
    solution = odeint(sir_odes, y0, t_obs, args=(beta, gamma, cfg.n_pop))
    infected = solution[:, 1] / cfg.n_pop
    return infected + rng.normal(0.0, cfg.noise_std, size=infected.shape)


def generate_sir_dataset(
    n_keep: int,
    rng: np.random.Generator,
    cfg: SirConfig,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate supercritical SIR simulations, with I0 normalized as in the notebook."""
    t_obs = np.linspace(0.0, cfg.t_max, cfg.n_timepoints)
    theta_chunks = []
    data_chunks = []
    pbar = tqdm(total=n_keep, desc=desc)

    while sum(chunk.shape[0] for chunk in theta_chunks) < n_keep:
        n_remaining = n_keep - sum(chunk.shape[0] for chunk in theta_chunks)
        n_draw = max(4 * n_remaining, 2048)
        beta = rng.uniform(cfg.beta_min, cfg.beta_max, n_draw)
        gamma = rng.uniform(cfg.gamma_min, cfg.gamma_max, n_draw)
        i0 = rng.poisson(cfg.i0_mean, n_draw).astype(np.float64)
        theta = np.stack([beta, gamma, i0], axis=1)
        keep = beta / gamma >= 1.0 + cfg.delta
        theta = theta[keep][:n_remaining]

        if theta.size == 0:
            continue

        data = np.array(
            [
                simulate_sir(row[0], row[1], row[2], rng, t_obs, cfg)
                for row in theta
            ],
            dtype=np.float32,
        )
        theta[:, 2] /= 10.0
        theta_chunks.append(theta.astype(np.float32))
        data_chunks.append(data)
        pbar.update(theta.shape[0])

    pbar.close()
    return np.concatenate(theta_chunks, axis=0), np.concatenate(data_chunks, axis=0)


def load_training_pool(path: Path, n_timepoints: int) -> tuple[np.ndarray, np.ndarray]:
    """Load a precomputed (theta, data) training pool from an .npz file.

    Resolution order:

    1. If the archive contains all four keys ``theta_train``, ``data_train``,
       ``theta_test``, ``data_test`` (e.g. saved by the distillery notebook),
       the train and test sets are concatenated along axis 0, train first,
       so for nsims <= len(train) only the train sims are used and for
       len(train) < nsims <= len(train) + len(test) the test sims fill in.
    2. Otherwise, the first matching pair among ``('theta','data')``,
       ``('theta_train','data_train')``, ``('theta_combined','data_combined')``
       is used as-is.

    Theta must have shape (N, 3) corresponding to (beta, gamma, I0/10), and data
    must have shape (N, n_timepoints).
    """
    archive = np.load(path)
    available = list(archive.files)

    train_test_keys = ("theta_train", "data_train", "theta_test", "data_test")
    if all(key in available for key in train_test_keys):
        theta_train = np.asarray(archive["theta_train"], dtype=np.float32)
        data_train = np.asarray(archive["data_train"], dtype=np.float32)
        theta_test = np.asarray(archive["theta_test"], dtype=np.float32)
        data_test = np.asarray(archive["data_test"], dtype=np.float32)
        if theta_train.shape[1:] != theta_test.shape[1:]:
            raise ValueError(
                f"theta_train and theta_test have incompatible shapes: "
                f"{theta_train.shape} vs {theta_test.shape}."
            )
        if data_train.shape[1:] != data_test.shape[1:]:
            raise ValueError(
                f"data_train and data_test have incompatible shapes: "
                f"{data_train.shape} vs {data_test.shape}."
            )
        theta = np.concatenate([theta_train, theta_test], axis=0)
        data = np.concatenate([data_train, data_test], axis=0)
        chosen_desc = (
            f"concatenated theta_train{theta_train.shape[0]} + "
            f"theta_test{theta_test.shape[0]}"
        )
    else:
        candidate_keys = [
            ("theta", "data"),
            ("theta_train", "data_train"),
            ("theta_combined", "data_combined"),
        ]
        chosen: tuple[str, str] | None = None
        for theta_key, data_key in candidate_keys:
            if theta_key in available and data_key in available:
                chosen = (theta_key, data_key)
                break
        if chosen is None:
            raise ValueError(
                f"Could not find a recognized (theta, data) key pair in {path}. "
                f"Available keys: {available}. Expected either all of "
                f"{train_test_keys} or one of {candidate_keys}."
            )
        theta = np.asarray(archive[chosen[0]], dtype=np.float32)
        data = np.asarray(archive[chosen[1]], dtype=np.float32)
        chosen_desc = f"{chosen[0]}/{chosen[1]}"

    expected_dim = THETA_PRIOR_LOW.size
    if theta.ndim != 2 or theta.shape[1] != expected_dim:
        raise ValueError(
            f"Loaded theta has shape {theta.shape} (from {chosen_desc}); expected "
            f"(N, {expected_dim}) with columns (beta, gamma, I0/10). If your "
            f"distillery saved a 2-D theta = (beta, gamma), append the I0/10 "
            f"column before saving."
        )
    if data.ndim != 2 or data.shape[0] != theta.shape[0]:
        raise ValueError(
            f"Loaded data has shape {data.shape} (from {chosen_desc}); expected "
            f"({theta.shape[0]}, {n_timepoints})."
        )
    if data.shape[1] != n_timepoints:
        raise ValueError(
            f"Loaded data has {data.shape[1]} timepoints (from {chosen_desc}); "
            f"expected {n_timepoints}."
        )
    return theta, data


def in_theta_prior(theta: np.ndarray) -> np.ndarray:
    finite = np.isfinite(theta).all(axis=1)
    above = (theta >= THETA_PRIOR_LOW).all(axis=1)
    below = (theta <= THETA_PRIOR_HIGH).all(axis=1)
    return finite & above & below


def inverse_logdet_jacobian(
    eta: np.ndarray,
    eta_to_theta_fn,
    max_points: int,
    seed: int,
) -> float:
    """Estimate mean log|det(dtheta/deta)| for converting eta log_prob to theta log_prob."""
    rng = np.random.default_rng(seed)
    if eta.shape[0] > max_points:
        eta = eta[rng.choice(eta.shape[0], size=max_points, replace=False)]

    logdets = []
    for point in eta:
        jac = np.empty((3, 3), dtype=np.float64)
        for dim in range(3):
            step = 1e-4 * max(1.0, abs(float(point[dim])))
            plus = point.astype(np.float64).copy()
            minus = point.astype(np.float64).copy()
            plus[dim] += step
            minus[dim] -= step
            jac[:, dim] = (
                eta_to_theta_fn(plus[None, :])[0].astype(np.float64)
                - eta_to_theta_fn(minus[None, :])[0].astype(np.float64)
            ) / (2.0 * step)
        det = np.linalg.det(jac)
        if np.isfinite(det) and abs(det) > 0.0:
            logdets.append(math.log(abs(det)))

    if not logdets:
        raise RuntimeError("Could not compute any finite eta inverse-Jacobian determinants.")
    return float(np.mean(logdets))


def make_runner(
    low: np.ndarray,
    high: np.ndarray,
    args: argparse.Namespace,
    device: str,
) -> InferenceRunner:
    prior = ili.utils.Uniform(low=low.tolist(), high=high.tolist(), device=device)
    nets = [
        ili.utils.load_nde_lampe(
            engine="NPE",
            model="maf",
            hidden_features=args.hidden_features,
            num_transforms=args.num_transforms,
            repeats=args.repeats_maf,
        )
    ]
    train_args = {
        "training_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_num_epochs": args.epochs,
    }
    return InferenceRunner.load(
        backend="lampe",
        engine="NPE",
        prior=prior,
        nets=nets,
        device=device,
        train_args=train_args,
        proposal=None,
        out_dir=None,
    )


def train_posterior(
    data: np.ndarray,
    params: np.ndarray,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
    args: argparse.Namespace,
    device: str,
    seed: int,
) -> tuple[Any, list[dict[str, Any]]]:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    runner = make_runner(prior_low, prior_high, args, device)
    loader = NumpyLoader(x=data.astype(np.float32), theta=params.astype(np.float32))
    posterior, summaries = runner(loader=loader)
    return posterior, summaries


def sample_one_observation(
    posterior: Any,
    x_obs: np.ndarray,
    n_samples: int,
    device: str,
) -> np.ndarray:
    x_tensor = torch.as_tensor(x_obs.astype(np.float32), device=device)
    with torch.no_grad():
        try:
            samples = posterior.sample((n_samples,), x=x_tensor, show_progress_bars=False)
        except TypeError:
            samples = posterior.sample((n_samples,), x=x_tensor)
    return samples.detach().cpu().numpy().astype(np.float32)


def smooth_curve(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding; window=1 returns values unchanged."""
    values = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return values.copy()
    if window > values.size:
        window = values.size
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def summarize_curves(
    summaries: list[dict[str, Any]],
    nsims: int,
    method: str,
    logdet_correction: float,
    validation_smoothing_window: int,
) -> list[dict[str, Any]]:
    rows = []
    for member, summary in enumerate(summaries):
        val = np.asarray(summary["validation_log_probs"], dtype=np.float64)
        train = np.asarray(summary["training_log_probs"], dtype=np.float64)
        val_smooth = smooth_curve(val, validation_smoothing_window)
        best_epoch = int(np.nanargmax(val_smooth))
        best_smoothed = float(val_smooth[best_epoch])
        rows.append(
            {
                "nsims": nsims,
                "method": method,
                "ensemble_member": member,
                "best_epoch": best_epoch,
                "validation_smoothing_window": validation_smoothing_window,
                "best_validation_log_prob_raw": float(val[best_epoch]),
                "best_validation_log_prob_smoothed": best_smoothed,
                "best_validation_log_prob_raw_theta_density": float(val[best_epoch] - logdet_correction),
                "best_validation_log_prob_theta_density": float(best_smoothed - logdet_correction),
                "final_validation_log_prob_raw": float(val[-1]),
                "final_training_log_prob_raw": float(train[-1]),
                "logdet_dtheta_deta_correction": logdet_correction,
            }
        )
    return rows


def dataframe_to_npz(path: Path, frame: pd.DataFrame) -> None:
    arrays = {column: frame[column].to_numpy() for column in frame.columns}
    np.savez_compressed(path, **arrays)


def beta_gamma_fom(samples: np.ndarray) -> tuple[float, int]:
    """Return 1/sqrt(det(cov(beta, gamma))) after dropping invalid samples."""
    samples_2d = samples[:, :2]
    valid = np.isfinite(samples_2d).all(axis=1)
    samples_2d = samples_2d[valid]
    if samples_2d.shape[0] < 3:
        return np.nan, int(samples_2d.shape[0])

    cov = np.cov(samples_2d, rowvar=False)
    det = np.linalg.det(cov)
    if not np.isfinite(det) or det <= 0.0:
        return np.nan, int(samples_2d.shape[0])
    return float(1.0 / np.sqrt(det)), int(samples_2d.shape[0])


def build_fom_comparison(
    posterior_arrays: dict[str, np.ndarray],
    nsims: int,
) -> pd.DataFrame:
    theta_key = f"n{nsims}_theta_theta_samples"
    eta_key = f"n{nsims}_eta_theta_samples"
    theta_samples = posterior_arrays[theta_key]
    eta_samples = posterior_arrays[eta_key]
    theta_true = posterior_arrays["theta_true"]
    example_indices = posterior_arrays["example_indices"]

    rows = []
    for example_number in range(theta_samples.shape[0]):
        fom_theta, n_valid_theta = beta_gamma_fom(theta_samples[example_number])
        fom_eta, n_valid_eta = beta_gamma_fom(eta_samples[example_number])
        rows.append(
            {
                "nsims": nsims,
                "example_number": example_number,
                "example_index": int(example_indices[example_number]),
                "theta_true_beta": float(theta_true[example_number, 0]),
                "theta_true_gamma": float(theta_true[example_number, 1]),
                "theta_true_i0_over_10": float(theta_true[example_number, 2]),
                "fom_theta": fom_theta,
                "fom_eta": fom_eta,
                "fom_eta_over_theta": float(fom_eta / fom_theta),
                "n_valid_theta_samples": n_valid_theta,
                "n_valid_eta_samples": n_valid_eta,
            }
        )
    return pd.DataFrame(rows)



def compute_coverage_diagnostics(
    posterior_samples: np.ndarray,
    theta_true: np.ndarray,
    seed: int,
    nbins: int = 10,
) -> dict[str, np.ndarray]:
    """Compute exportable data behind ltu-ili's one-dimensional coverage plots.

    Parameters
    ----------
    posterior_samples
        Array with shape (num_samples, n_data, n_params), matching ltu-ili.
    theta_true
        True parameters with shape (n_data, n_params).
    seed
        Seed used only for the reference uniform coverage bands.
    nbins
        Rank-histogram bins.
    """
    posterior_samples = np.asarray(posterior_samples, dtype=np.float32)
    theta_true = np.asarray(theta_true, dtype=np.float32)
    num_samples, n_data, n_params = posterior_samples.shape

    finite = np.isfinite(posterior_samples).all(axis=-1)
    valid_counts = finite.sum(axis=0).astype(np.int64)
    ranks = np.zeros((n_data, n_params), dtype=np.int64)
    percentiles = np.full((n_data, n_params), np.nan, dtype=np.float32)

    for data_idx in range(n_data):
        valid = finite[:, data_idx]
        if valid.sum() == 0:
            continue
        ranks[data_idx] = (posterior_samples[valid, data_idx, :] < theta_true[data_idx]).sum(axis=0)
        percentiles[data_idx] = ranks[data_idx] / valid.sum()

    empirical_cdf = np.linspace(0.0, 1.0, n_data, dtype=np.float32)
    predicted_percentiles = np.full((n_data, n_params), np.nan, dtype=np.float32)
    rank_hist_counts = np.zeros((n_params, nbins), dtype=np.int64)
    rank_hist_edges = np.linspace(0.0, 1.0, nbins + 1, dtype=np.float32)

    for param_idx in range(n_params):
        finite_pct = percentiles[np.isfinite(percentiles[:, param_idx]), param_idx]
        if finite_pct.size:
            predicted_percentiles[: finite_pct.size, param_idx] = np.sort(finite_pct)
            rank_hist_counts[param_idx], _ = np.histogram(finite_pct, bins=rank_hist_edges)

    rng = np.random.default_rng(seed)
    uniform_curves = np.sort(rng.uniform(0.0, 1.0, size=(200, n_data)), axis=1)
    uniform_bands = np.percentile(uniform_curves, [5, 16, 84, 95], axis=0).astype(np.float32)

    sample_mean = np.nanmean(posterior_samples, axis=0).astype(np.float32)
    sample_std = np.nanstd(posterior_samples, axis=0).astype(np.float32)

    return {
        "posterior_samples": posterior_samples,
        "theta_true": theta_true,
        "ranks": ranks,
        "percentiles": percentiles,
        "predicted_percentiles": predicted_percentiles,
        "empirical_cdf": empirical_cdf,
        "uniform_bands_percentiles": np.array([5, 16, 84, 95], dtype=np.int64),
        "uniform_bands": uniform_bands,
        "rank_hist_counts": rank_hist_counts,
        "rank_hist_edges": rank_hist_edges,
        "sample_mean": sample_mean,
        "sample_std": sample_std,
        "valid_counts": valid_counts,
    }


def save_coverage_outputs(
    out_dir: Path,
    coverage_arrays: dict[str, np.ndarray],
) -> None:
    if coverage_arrays:
        np.savez_compressed(out_dir / "coverage_outputs.npz", **coverage_arrays)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    cfg = SirConfig(noise_std=args.noise_std)
    theta_to_eta = make_expression_transform(args.theta_to_eta)
    eta_to_theta = make_expression_transform(args.eta_to_theta)
    max_nsims = max(args.nsims)
    rng = np.random.default_rng(args.seed)

    print(f"Using device: {device}")
    print(f"SIR noise_std = {cfg.noise_std}")
    if args.training_data is not None:
        print(f"Loading training pool from {args.training_data}")
        loaded_theta, loaded_data = load_training_pool(args.training_data, cfg.n_timepoints)
        n_loaded = loaded_theta.shape[0]
        n_from_file = min(n_loaded, max_nsims)
        n_extra = max(0, max_nsims - n_loaded)
        print(
            f"Loaded {n_loaded} sims; using first {n_from_file} for the training pool. "
            f"Need {n_extra} additional sims to reach max_nsims={max_nsims}."
        )
        theta_pool_pre = loaded_theta[:n_from_file]
        data_pool_pre = loaded_data[:n_from_file]
        if n_extra > 0:
            extra_theta, extra_data = generate_sir_dataset(
                n_extra, rng, cfg, "extra train sims"
            )
            theta_pool = np.concatenate([theta_pool_pre, extra_theta], axis=0).astype(np.float32)
            data_pool = np.concatenate([data_pool_pre, extra_data], axis=0).astype(np.float32)
        else:
            theta_pool = theta_pool_pre.astype(np.float32)
            data_pool = data_pool_pre.astype(np.float32)
        print(f"Generating {args.n_test} test simulations.")
        theta_test, data_test = generate_sir_dataset(args.n_test, rng, cfg, "test sims")
    else:
        print(
            f"Generating {max_nsims} training simulations and {args.n_test} test simulations."
        )
        theta_pool, data_pool = generate_sir_dataset(max_nsims, rng, cfg, "train sims")
        theta_test, data_test = generate_sir_dataset(args.n_test, rng, cfg, "test sims")

    if args.n_examples > args.n_test:
        raise ValueError("--n-examples cannot exceed --n-test.")
    example_rng = np.random.default_rng(args.seed + 10_000)
    example_indices = example_rng.choice(args.n_test, size=args.n_examples, replace=False)

    coverage_nsims = args.fom_nsims if args.coverage_nsims is None else args.coverage_nsims
    if args.run_coverage and coverage_nsims not in args.nsims:
        raise ValueError("Coverage nsims must be included in --nsims.")
    coverage_indices = np.array([], dtype=np.int64)
    if args.run_coverage:
        coverage_n_test = min(args.coverage_n_test, args.n_test)
        coverage_seed = args.seed + 20_000 if args.coverage_seed is None else args.coverage_seed
        coverage_rng = np.random.default_rng(coverage_seed)
        coverage_indices = coverage_rng.choice(args.n_test, size=coverage_n_test, replace=False)

    np.savez_compressed(
        args.out_dir / "dataset_reference.npz",
        theta_test=theta_test.astype(np.float32),
        data_test=data_test.astype(np.float32),
        example_indices=example_indices.astype(np.int64),
        theta_examples=theta_test[example_indices].astype(np.float32),
        data_examples=data_test[example_indices].astype(np.float32),
    )

    metrics_rows: list[dict[str, Any]] = []
    history_arrays: dict[str, np.ndarray] = {}
    coverage_arrays: dict[str, np.ndarray] = {}
    if args.run_coverage:
        coverage_arrays = {
            "coverage_nsims": np.asarray(coverage_nsims, dtype=np.int64),
            "coverage_indices": coverage_indices.astype(np.int64),
            "coverage_num_samples": np.asarray(args.coverage_num_samples, dtype=np.int64),
            "theta_true": theta_test[coverage_indices].astype(np.float32),
            "data_obs": data_test[coverage_indices].astype(np.float32),
        }

    posterior_arrays: dict[str, np.ndarray] = {
        "theta_true": theta_test[example_indices].astype(np.float32),
        "data_obs": data_test[example_indices].astype(np.float32),
        "example_indices": example_indices.astype(np.int64),
        "nsims": np.asarray(args.nsims, dtype=np.int64),
    }

    for nsims in args.nsims:
        train_theta = theta_pool[:nsims]
        train_data = data_pool[:nsims]

        for method in ("theta", "eta"):
            start = time.time()
            run_seed = args.seed + nsims * 10 + (0 if method == "theta" else 1)
            print(f"\nTraining method={method}, nsims={nsims}, seed={run_seed}")

            if method == "theta":
                train_params = train_theta
                prior_low = THETA_PRIOR_LOW
                prior_high = THETA_PRIOR_HIGH
                logdet_correction = 0.0
            else:
                train_params = theta_to_eta(train_theta)
                prior_low = train_params.min(axis=0)
                prior_high = train_params.max(axis=0)
                logdet_correction = inverse_logdet_jacobian(
                    train_params,
                    eta_to_theta_fn=eta_to_theta,
                    max_points=args.jacobian_correction_samples,
                    seed=run_seed,
                )

            posterior, summaries = train_posterior(
                train_data,
                train_params,
                prior_low,
                prior_high,
                args,
                device,
                seed=run_seed,
            )

            metrics_rows.extend(
                summarize_curves(
                    summaries=summaries,
                    nsims=nsims,
                    method=method,
                    logdet_correction=logdet_correction,
                    validation_smoothing_window=args.validation_smoothing_window,
                )
            )

            for member, summary in enumerate(summaries):
                prefix = f"n{nsims}_{method}_member{member}"
                training_log_probs = np.asarray(summary["training_log_probs"], dtype=np.float32)
                validation_log_probs = np.asarray(summary["validation_log_probs"], dtype=np.float32)
                validation_log_probs_smoothed = smooth_curve(
                    validation_log_probs, args.validation_smoothing_window
                ).astype(np.float32)
                history_arrays[f"{prefix}_training_log_probs"] = training_log_probs
                history_arrays[f"{prefix}_validation_log_probs"] = validation_log_probs
                history_arrays[f"{prefix}_validation_log_probs_smoothed"] = validation_log_probs_smoothed
                history_arrays[f"{prefix}_training_log_probs_theta_density"] = (
                    training_log_probs - logdet_correction
                ).astype(np.float32)
                history_arrays[f"{prefix}_validation_log_probs_theta_density"] = (
                    validation_log_probs - logdet_correction
                ).astype(np.float32)
                history_arrays[f"{prefix}_validation_log_probs_smoothed_theta_density"] = (
                    validation_log_probs_smoothed - logdet_correction
                ).astype(np.float32)


            if args.run_coverage and nsims == coverage_nsims:
                coverage_theta_samples = []
                coverage_valid_masks = []
                for idx in tqdm(coverage_indices, desc=f"coverage samples {method} n={nsims}"):
                    raw_samples = sample_one_observation(
                        posterior,
                        data_test[idx],
                        args.coverage_num_samples,
                        device,
                    )
                    if method == "theta":
                        theta_samples = raw_samples
                        valid_mask = in_theta_prior(theta_samples)
                    else:
                        theta_samples = eta_to_theta(raw_samples)
                        valid_mask = in_theta_prior(theta_samples)
                    theta_samples = theta_samples.copy()
                    theta_samples[~valid_mask] = np.nan
                    coverage_theta_samples.append(theta_samples.astype(np.float32))
                    coverage_valid_masks.append(valid_mask)

                coverage_samples_ltu = np.stack(coverage_theta_samples, axis=1).astype(np.float32)
                coverage_valid_ltu = np.stack(coverage_valid_masks, axis=1)
                diagnostics = compute_coverage_diagnostics(
                    coverage_samples_ltu,
                    theta_test[coverage_indices],
                    seed=run_seed,
                )
                coverage_prefix = f"n{nsims}_{method}"
                coverage_arrays[f"{coverage_prefix}_coverage_valid_mask"] = coverage_valid_ltu
                for diag_name, diag_value in diagnostics.items():
                    coverage_arrays[f"{coverage_prefix}_coverage_{diag_name}"] = diag_value
                save_coverage_outputs(args.out_dir, coverage_arrays)

            raw_examples = []
            theta_examples = []
            valid_masks = []
            for idx in tqdm(example_indices, desc=f"posterior samples {method} n={nsims}"):
                raw_samples = sample_one_observation(
                    posterior,
                    data_test[idx],
                    args.n_posterior_samples,
                    device,
                )
                if method == "theta":
                    theta_samples = raw_samples
                    valid_mask = np.isfinite(theta_samples).all(axis=1)
                else:
                    theta_samples = eta_to_theta(raw_samples)
                    valid_mask = in_theta_prior(theta_samples)
                    theta_samples = theta_samples.copy()
                    theta_samples[~valid_mask] = np.nan

                raw_examples.append(raw_samples)
                theta_examples.append(theta_samples.astype(np.float32))
                valid_masks.append(valid_mask)

            prefix = f"n{nsims}_{method}"
            posterior_arrays[f"{prefix}_raw_samples"] = np.stack(raw_examples).astype(np.float32)
            posterior_arrays[f"{prefix}_theta_samples"] = np.stack(theta_examples).astype(np.float32)
            posterior_arrays[f"{prefix}_valid_mask"] = np.stack(valid_masks)

            elapsed = time.time() - start
            metrics = pd.DataFrame(metrics_rows)
            metrics.to_csv(args.out_dir / "metrics.csv", index=False)
            dataframe_to_npz(args.out_dir / "metrics.npz", metrics)
            aggregate = (
                metrics.groupby(["nsims", "method"], as_index=False)
                .agg(
                    best_validation_log_prob_theta_density_mean=(
                        "best_validation_log_prob_theta_density",
                        "mean",
                    ),
                    best_validation_log_prob_theta_density_std=(
                        "best_validation_log_prob_theta_density",
                        "std",
                    ),
                    best_validation_log_prob_raw_mean=("best_validation_log_prob_raw", "mean"),
                    best_validation_log_prob_raw_std=("best_validation_log_prob_raw", "std"),
                    n_ensemble_members=("ensemble_member", "count"),
                )
            )
            aggregate.to_csv(args.out_dir / "metrics_aggregate.csv", index=False)
            dataframe_to_npz(args.out_dir / "metrics_aggregate.npz", aggregate)
            fom_theta_key = f"n{args.fom_nsims}_theta_theta_samples"
            fom_eta_key = f"n{args.fom_nsims}_eta_theta_samples"
            if fom_theta_key in posterior_arrays and fom_eta_key in posterior_arrays:
                fom_comparison = build_fom_comparison(posterior_arrays, args.fom_nsims)
                fom_comparison.to_csv(args.out_dir / "fom_comparison.csv", index=False)
                dataframe_to_npz(args.out_dir / "fom_comparison.npz", fom_comparison)
            save_coverage_outputs(args.out_dir, coverage_arrays)
            np.savez_compressed(args.out_dir / "training_histories.npz", **history_arrays)
            np.savez_compressed(args.out_dir / "posterior_samples.npz", **posterior_arrays)
            print(f"Finished method={method}, nsims={nsims} in {elapsed / 60.0:.1f} min")

    manifest = {
        "config": {
            "args": vars(args) | {
                "out_dir": str(args.out_dir),
                "training_data": (
                    str(args.training_data) if args.training_data is not None else None
                ),
                "device_resolved": device,
            },
            "sir": asdict(cfg),
            "theta_labels": THETA_LABELS,
            "eta_labels": ETA_LABELS,
            "theta_prior_low": THETA_PRIOR_LOW.tolist(),
            "theta_prior_high": THETA_PRIOR_HIGH.tolist(),
            "theta_to_eta_expressions": list(args.theta_to_eta),
            "eta_to_theta_expressions": list(args.eta_to_theta),
        },
        "outputs": {
            "metrics_csv": "metrics.csv",
            "metrics_npz": "metrics.npz",
            "metrics_aggregate_csv": "metrics_aggregate.csv",
            "metrics_aggregate_npz": "metrics_aggregate.npz",
            "fom_comparison_csv": "fom_comparison.csv",
            "fom_comparison_npz": "fom_comparison.npz",
            "training_histories_npz": "training_histories.npz",
            "posterior_samples_npz": "posterior_samples.npz",
            "dataset_reference_npz": "dataset_reference.npz",
            "coverage_outputs_npz": "coverage_outputs.npz",
        },
        "notes": [
            "For eta runs, best_validation_log_prob_raw is in eta density units.",
            "best_validation_log_prob_theta_density uses the smoothed validation curve when --validation-smoothing-window > 1.",
            "best_validation_log_prob_raw is the raw validation value at the epoch selected by the smoothed curve.",
            "best_validation_log_prob_theta_density subtracts mean log|det(dtheta/deta)|.",
            "posterior *_theta_samples arrays are mapped to physical theta coordinates.",
            "Invalid eta->theta samples outside the original theta prior are set to NaN.",
            "Coverage outputs are saved only when --run-coverage is set.",
            "Coverage ranks and predicted_percentiles follow ltu-ili PosteriorCoverage marginal-rank diagnostics.",
            "FoM is 1/sqrt(det(cov(beta, gamma))) for --fom-nsims, default 1000.",
        ],
    }
    with open(args.out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Results written to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
