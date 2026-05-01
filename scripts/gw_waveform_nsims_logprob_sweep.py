"""Train GW waveform NPEs in theta and eta coordinates over an nsims sweep.

This script is adapted from ``notebooks/gw_experiment_waveform.ipynb`` and is
intended to be run as a command-line job, for example in Google Colab:

    python scripts/gw_waveform_nsims_logprob_sweep.py --out-dir gw_waveform_sweep_results

It saves:
  * ``metrics.csv`` and ``metrics.npz``: best validation log_prob by nsims/method/member.
  * ``metrics_aggregate.csv`` and ``metrics_aggregate.npz``: plot-ready summaries.
  * ``fom_comparison.csv`` and ``fom_comparison.npz``: theta-vs-eta FoM comparison.
  * ``training_histories.npz``: full train/validation curves for every run.
  * ``posterior_samples.npz``: seed-matched posterior samples for example test cases.
  * ``dataset_reference.npz``: test data, examples, PCA metadata, and frequency grid.
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
from sklearn.decomposition import PCA
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


THETA_LABELS = (r"$m_1$", r"$m_2$")
ETA_LABELS = (r"$\eta_0$", r"$\eta_1$")

DEFAULT_THETA_TO_ETA_EXPRS = (
    "0.001*m1 - 0.123",
    "(0.369*m1*(0.026*m1)**(0.187*m1) + 0.16*m2)"
    "/((0.026*m1)**(0.187*m1)*(0.128*m2)**(0.051*m2))",
)
DEFAULT_ETA_TO_THETA_EXPRS = (
    "0.001*m1 - 0.123",
    "(0.369*m1*(0.026*m1)**(0.187*m1) + 0.16*m2)"
    "/((0.026*m1)**(0.187*m1)*(0.128*m2)**(0.051*m2))",
)


@dataclass
class GwWaveformConfig:
    m_sun_sec: float = 4.925491025543576e-6
    mpc_sec: float = 1.0292712503e14
    m1_min: float = 5.0
    m1_max: float = 50.0
    m2_min: float = 5.0
    m2_max: float = 50.0
    d_l_mpc: float = 200.0
    f_low: float = 20.0
    df: float = 0.5
    n_pca: int = 40
    pca_bank_size: int = 5000

    @property
    def prior_low(self) -> np.ndarray:
        return np.array([self.m1_min, self.m2_min], dtype=np.float32)

    @property
    def prior_high(self) -> np.ndarray:
        return np.array([self.m1_max, self.m2_max], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep GW waveform NPE validation log_prob versus number of simulations."
    )
    parser.add_argument("--nsims", nargs="+", type=int, default=[100, 500, 1000, 5000, 10000])
    parser.add_argument("--out-dir", type=Path, default=Path("gw_waveform_nsims_logprob_sweep"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--n-examples", type=int, default=8)
    parser.add_argument("--n-posterior-samples", type=int, default=10000)
    parser.add_argument(
        "--fom-nsims",
        type=int,
        default=1000,
        help="Simulation count at which to save the theta-vs-eta FoM comparison.",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--repeats-maf", type=int, default=2)
    parser.add_argument("--hidden-features", type=int, default=50)
    parser.add_argument("--num-transforms", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--theta-to-eta",
        nargs=2,
        default=DEFAULT_THETA_TO_ETA_EXPRS,
        metavar=("ETA0", "ETA1"),
        help="Two expressions using m1,m2 or X1,X2 for theta=(m1,m2).",
    )
    parser.add_argument(
        "--eta-to-theta",
        nargs=2,
        default=DEFAULT_ETA_TO_THETA_EXPRS,
        metavar=("THETA0", "THETA1"),
        help="Two inverse expressions using m1,m2 or X1,X2 for eta coordinates.",
    )
    parser.add_argument(
        "--jacobian-correction-samples",
        type=int,
        default=512,
        help="Number of eta training points used to estimate mean log|dtheta/deta|.",
    )
    return parser.parse_args()


def make_expression_transform(expressions: tuple[str, str] | list[str]):
    """Build a NumPy vectorized transform from expressions in m1,m2 or X1,X2."""
    allowed_names = {
        "E": math.e,
        "abs": np.abs,
        "arccos": np.arccos,
        "arcsin": np.arcsin,
        "arctan": np.arctan,
        "cos": np.cos,
        "e": math.e,
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
        local_names = {"X1": x[:, 0], "X2": x[:, 1], "m1": x[:, 0], "m2": x[:, 1]}
        cols = [
            eval(code, {"__builtins__": {}}, allowed_names | local_names)
            for code in compiled
        ]
        return np.column_stack(cols).astype(np.float32)

    return transform


def chirp_mass(m1: np.ndarray | float, m2: np.ndarray | float) -> np.ndarray | float:
    return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)


def symmetric_mass_ratio(m1: np.ndarray | float, m2: np.ndarray | float) -> np.ndarray | float:
    return (m1 * m2) / (m1 + m2) ** 2


def aligo_psd(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    f0 = 215.0
    x = f / f0
    psd = 1e-49 * (x ** (-4.14) + 2.0 + 2.0 * x**2)
    return np.where(f >= 10.0, psd, np.inf)


def frequency_grid(cfg: GwWaveformConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f_isco_lightest = 1.0 / (6**1.5 * np.pi * (cfg.m1_min + cfg.m2_min) * cfg.m_sun_sec)
    freqs = np.arange(cfg.f_low, min(f_isco_lightest, 1024.0), cfg.df)
    psd = aligo_psd(freqs)
    whiten_factor = np.sqrt(4.0 * cfg.df) / np.sqrt(psd)
    return freqs.astype(np.float64), psd.astype(np.float64), whiten_factor.astype(np.float64)


def taylorf2_waveform(
    m1: float,
    m2: float,
    freqs: np.ndarray,
    cfg: GwWaveformConfig,
) -> np.ndarray:
    """TaylorF2 inspiral waveform, non-spinning 2PN phase, zero above f_ISCO."""
    m_sec = (m1 + m2) * cfg.m_sun_sec
    eta = m1 * m2 / (m1 + m2) ** 2
    mc_sec = chirp_mass(m1, m2) * cfg.m_sun_sec
    d_l_sec = cfg.d_l_mpc * cfg.mpc_sec

    f_isco = 1.0 / (6**1.5 * np.pi * m_sec)
    active = (freqs > 0) & (freqs < f_isco)

    v = np.zeros_like(freqs)
    v[active] = (np.pi * m_sec * freqs[active]) ** (1.0 / 3)

    amp = np.zeros_like(freqs)
    amp[active] = (
        np.sqrt(5.0 / 24)
        * np.pi ** (-2.0 / 3)
        * mc_sec ** (5.0 / 6)
        / d_l_sec
        * freqs[active] ** (-7.0 / 6)
    )

    c1 = 3715.0 / 756 + 55.0 / 9 * eta
    c15 = -16.0 * np.pi
    c2 = 15379365.0 / 508032 + 27145.0 / 504 * eta + 3085.0 / 72 * eta**2

    phase = np.zeros_like(freqs)
    vm = v[active]
    phase[active] = 3.0 / (128 * eta) * vm ** (-5) * (1 + c1 * vm**2 + c15 * vm**3 + c2 * vm**4)
    return amp * np.exp(1j * phase)


def sample_masses(n_sims: int, rng: np.random.Generator, cfg: GwWaveformConfig) -> np.ndarray:
    chunks = []
    while sum(chunk.shape[0] for chunk in chunks) < n_sims:
        n_remaining = n_sims - sum(chunk.shape[0] for chunk in chunks)
        n_draw = max(4 * n_remaining, 2048)
        m1 = rng.uniform(cfg.m1_min, cfg.m1_max, n_draw)
        m2 = rng.uniform(cfg.m2_min, cfg.m2_max, n_draw)
        keep = m1 >= m2
        theta = np.stack([m1[keep], m2[keep]], axis=1)[:n_remaining]
        if theta.size:
            chunks.append(theta.astype(np.float32))
    return np.concatenate(chunks, axis=0)


def clean_waveform_vector(
    theta: np.ndarray,
    freqs: np.ndarray,
    whiten_factor: np.ndarray,
    cfg: GwWaveformConfig,
) -> np.ndarray:
    h = taylorf2_waveform(float(theta[0]), float(theta[1]), freqs, cfg)
    hw = h * whiten_factor
    return np.concatenate([hw.real, hw.imag]).astype(np.float32)


def generate_gw_dataset(
    theta: np.ndarray,
    pca: PCA,
    freqs: np.ndarray,
    whiten_factor: np.ndarray,
    rng: np.random.Generator,
    cfg: GwWaveformConfig,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.empty((theta.shape[0], cfg.n_pca), dtype=np.float32)
    snr = np.empty(theta.shape[0], dtype=np.float32)
    for i in tqdm(range(theta.shape[0]), desc=desc):
        hvec = clean_waveform_vector(theta[i], freqs, whiten_factor, cfg)
        snr[i] = np.linalg.norm(hvec)
        noisy = hvec + rng.normal(size=hvec.shape).astype(np.float32)
        data[i] = pca.transform(noisy.reshape(1, -1)).flatten().astype(np.float32)
    return data, snr


def build_pca(
    theta: np.ndarray,
    freqs: np.ndarray,
    whiten_factor: np.ndarray,
    rng: np.random.Generator,
    cfg: GwWaveformConfig,
) -> tuple[PCA, np.ndarray]:
    n_bank = min(cfg.pca_bank_size, theta.shape[0])
    idx = rng.choice(theta.shape[0], size=n_bank, replace=False)
    bank = np.empty((n_bank, 2 * len(freqs)), dtype=np.float32)
    for j, i in enumerate(tqdm(idx, desc="PCA bank")):
        bank[j] = clean_waveform_vector(theta[i], freqs, whiten_factor, cfg)
    pca = PCA(n_components=cfg.n_pca).fit(bank)
    return pca, pca.explained_variance_ratio_.cumsum().astype(np.float32)


def in_theta_prior(theta: np.ndarray, cfg: GwWaveformConfig) -> np.ndarray:
    finite = np.isfinite(theta).all(axis=1)
    above = (theta >= cfg.prior_low).all(axis=1)
    below = (theta <= cfg.prior_high).all(axis=1)
    ordered = theta[:, 0] >= theta[:, 1]
    return finite & above & below & ordered


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
        jac = np.empty((2, 2), dtype=np.float64)
        for dim in range(2):
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
        return 0.0
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


def summarize_curves(
    summaries: list[dict[str, Any]],
    nsims: int,
    method: str,
    logdet_correction: float,
) -> list[dict[str, Any]]:
    rows = []
    for member, summary in enumerate(summaries):
        val = np.asarray(summary["validation_log_probs"], dtype=np.float64)
        train = np.asarray(summary["training_log_probs"], dtype=np.float64)
        best_epoch = int(np.nanargmax(val))
        rows.append(
            {
                "nsims": nsims,
                "method": method,
                "ensemble_member": member,
                "best_epoch": best_epoch,
                "best_validation_log_prob_raw": float(val[best_epoch]),
                "best_validation_log_prob_theta_density": float(val[best_epoch] - logdet_correction),
                "final_validation_log_prob_raw": float(val[-1]),
                "final_training_log_prob_raw": float(train[-1]),
                "logdet_dtheta_deta_correction": logdet_correction,
            }
        )
    return rows


def dataframe_to_npz(path: Path, frame: pd.DataFrame) -> None:
    arrays = {column: frame[column].to_numpy() for column in frame.columns}
    np.savez_compressed(path, **arrays)


def theta_fom(samples: np.ndarray) -> tuple[float, int]:
    """Return 1/sqrt(det(cov(m1, m2))) after dropping invalid samples."""
    valid = np.isfinite(samples).all(axis=1)
    samples = samples[valid]
    if samples.shape[0] < 3:
        return np.nan, int(samples.shape[0])

    cov = np.cov(samples[:, :2], rowvar=False)
    det = np.linalg.det(cov)
    if not np.isfinite(det) or det <= 0.0:
        return np.nan, int(samples.shape[0])
    return float(1.0 / np.sqrt(det)), int(samples.shape[0])


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
        fom_theta, n_valid_theta = theta_fom(theta_samples[example_number])
        fom_eta, n_valid_eta = theta_fom(eta_samples[example_number])
        rows.append(
            {
                "nsims": nsims,
                "example_number": example_number,
                "example_index": int(example_indices[example_number]),
                "theta_true_m1": float(theta_true[example_number, 0]),
                "theta_true_m2": float(theta_true[example_number, 1]),
                "fom_theta": fom_theta,
                "fom_eta": fom_eta,
                "fom_eta_over_theta": float(fom_eta / fom_theta),
                "n_valid_theta_samples": n_valid_theta,
                "n_valid_eta_samples": n_valid_eta,
            }
        )
    return pd.DataFrame(rows)


def save_metrics_outputs(
    out_dir: Path,
    metrics_rows: list[dict[str, Any]],
    posterior_arrays: dict[str, np.ndarray],
    fom_nsims: int,
) -> None:
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    dataframe_to_npz(out_dir / "metrics.npz", metrics)

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
    aggregate.to_csv(out_dir / "metrics_aggregate.csv", index=False)
    dataframe_to_npz(out_dir / "metrics_aggregate.npz", aggregate)

    fom_theta_key = f"n{fom_nsims}_theta_theta_samples"
    fom_eta_key = f"n{fom_nsims}_eta_theta_samples"
    if fom_theta_key in posterior_arrays and fom_eta_key in posterior_arrays:
        fom_comparison = build_fom_comparison(posterior_arrays, fom_nsims)
        fom_comparison.to_csv(out_dir / "fom_comparison.csv", index=False)
        dataframe_to_npz(out_dir / "fom_comparison.npz", fom_comparison)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    cfg = GwWaveformConfig()
    theta_to_eta = make_expression_transform(args.theta_to_eta)
    eta_to_theta = make_expression_transform(args.eta_to_theta)
    max_nsims = max(args.nsims)
    rng = np.random.default_rng(args.seed)
    freqs, psd, whiten_factor = frequency_grid(cfg)

    print(f"Using device: {device}")
    print(f"Frequency grid: {cfg.f_low}-{freqs[-1]:.0f} Hz, {len(freqs)} bins, df={cfg.df}")
    print(f"Generating {max_nsims} training masses and {args.n_test} test masses.")

    theta_pool = sample_masses(max_nsims, rng, cfg)
    theta_test = sample_masses(args.n_test, rng, cfg)
    theta_for_pca = np.concatenate([theta_pool, theta_test], axis=0)
    print(f"Building PCA basis from {min(cfg.pca_bank_size, theta_for_pca.shape[0])} clean waveforms.")
    pca, cumvar = build_pca(theta_for_pca, freqs, whiten_factor, rng, cfg)

    print("Generating noisy PCA-compressed waveform observations.")
    data_pool, snr_pool = generate_gw_dataset(theta_pool, pca, freqs, whiten_factor, rng, cfg, "train waveforms")
    data_test, snr_test = generate_gw_dataset(theta_test, pca, freqs, whiten_factor, rng, cfg, "test waveforms")

    if args.n_examples > args.n_test:
        raise ValueError("--n-examples cannot exceed --n-test.")
    example_rng = np.random.default_rng(args.seed + 10_000)
    example_indices = example_rng.choice(args.n_test, size=args.n_examples, replace=False)

    np.savez_compressed(
        args.out_dir / "dataset_reference.npz",
        theta_test=theta_test.astype(np.float32),
        data_test=data_test.astype(np.float32),
        snr_test=snr_test.astype(np.float32),
        example_indices=example_indices.astype(np.int64),
        theta_examples=theta_test[example_indices].astype(np.float32),
        data_examples=data_test[example_indices].astype(np.float32),
        freqs=freqs.astype(np.float32),
        psd=psd.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        pca_explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        pca_cumulative_variance=cumvar.astype(np.float32),
    )

    metrics_rows: list[dict[str, Any]] = []
    history_arrays: dict[str, np.ndarray] = {}
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
                prior_low = cfg.prior_low
                prior_high = cfg.prior_high
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
                )
            )

            for member, summary in enumerate(summaries):
                prefix = f"n{nsims}_{method}_member{member}"
                history_arrays[f"{prefix}_training_log_probs"] = np.asarray(
                    summary["training_log_probs"], dtype=np.float32
                )
                history_arrays[f"{prefix}_validation_log_probs"] = np.asarray(
                    summary["validation_log_probs"], dtype=np.float32
                )

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
                    valid_mask = in_theta_prior(theta_samples, cfg)
                else:
                    theta_samples = eta_to_theta(raw_samples)
                    valid_mask = in_theta_prior(theta_samples, cfg)
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
            save_metrics_outputs(args.out_dir, metrics_rows, posterior_arrays, args.fom_nsims)
            np.savez_compressed(args.out_dir / "training_histories.npz", **history_arrays)
            np.savez_compressed(args.out_dir / "posterior_samples.npz", **posterior_arrays)
            print(f"Finished method={method}, nsims={nsims} in {elapsed / 60.0:.1f} min")

    manifest = {
        "config": {
            "args": vars(args) | {"out_dir": str(args.out_dir), "device_resolved": device},
            "gw_waveform": asdict(cfg),
            "theta_labels": THETA_LABELS,
            "eta_labels": ETA_LABELS,
            "theta_prior_low": cfg.prior_low.tolist(),
            "theta_prior_high": cfg.prior_high.tolist(),
            "theta_to_eta_expressions": list(args.theta_to_eta),
            "eta_to_theta_expressions": list(args.eta_to_theta),
            "frequency_bins": int(len(freqs)),
            "pca_cumulative_variance_final": float(cumvar[-1]),
            "snr_train_min_median_max": [
                float(np.min(snr_pool)),
                float(np.median(snr_pool)),
                float(np.max(snr_pool)),
            ],
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
        },
        "notes": [
            "For eta runs, best_validation_log_prob_raw is in eta density units.",
            "best_validation_log_prob_theta_density subtracts mean log|det(dtheta/deta)|.",
            "posterior *_theta_samples arrays are mapped to physical (m1, m2) coordinates.",
            "Invalid samples outside m1/m2 bounds or with m1 < m2 are set to NaN for eta runs.",
            "FoM is 1/sqrt(det(cov(m1, m2))) for --fom-nsims, default 1000.",
        ],
    }
    with open(args.out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Results written to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
