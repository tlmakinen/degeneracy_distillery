"""Train 2D weak-lensing NPEs in theta and eta coordinates over an nsims sweep.

This script mirrors the GW and SIR sweeps but operates on the 2-D weak-lensing
problem with parameters ``theta = (Omega_m, sigma_8)`` and a 2-pt summary
extracted from tomographic shear maps.  The dataset is loaded from a numpy
archive (``--data-path``, e.g. mounted in Google Drive at
``/content/drive/MyDrive/Colab Notebooks/wl-sbi/...``), Gaussian noise is
re-injected on each load (so we can sweep ``--noise-scale`` on top of the
fixed simulator output), and the noisy maps are compressed to ``log C_ell``.

Run from the repo root, e.g. in Colab::

    python scripts/wl_2d_nsims_logprob_sweep.py \
        --out-dir wl_2d_nsims_logprob_sweep \
        --noise-scale 0.25 \
        --compare-conventional-coords

It saves:
  * ``metrics.csv`` / ``metrics.npz``: best validation log_prob by nsims, method,
    and ensemble member.
  * ``metrics_aggregate.csv`` / ``.npz``: mean/std summaries.
  * ``fom_comparison.csv`` / ``.npz``: theta-vs-eta (and conventional) FoM comparison
    at ``--fom-nsims``.
  * ``training_histories.npz``: full train/validation curves.
  * ``posterior_samples.npz``: seed-matched posterior samples for the example
    test cases (in eta, raw, and physical theta units).
  * ``dataset_reference.npz``: noisy compressed test data, examples, prior
    bounds, and the noise/binning configuration that produced them.
  * ``coverage_outputs.npz`` (optional): coverage diagnostics in theta units.
  * ``manifest.json``: configuration and file descriptions.

The default learned/inverse expressions are the rank-1 distillery output for
the (Omega_m, sigma_8) Fisher; the default conventional coordinates are the
conventional ``(Omega_m, S_8 = sigma_8 * sqrt(Omega_m / 0.3))`` mapping.
Both are configurable via CLI.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    import jax
    import jax.numpy as jnp
    import jax.random as jr
except ImportError as exc:  # pragma: no cover - this is a runtime environment check.
    raise SystemExit(
        "This script requires JAX. In Colab it is preinstalled; otherwise install with:\n"
        "  pip install -q -U 'jax[cpu]' (or the cuda variant matching your runtime)."
    ) from exc

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


THETA_LABELS = (r"$\Omega_m$", r"$\sigma_8$")
ETA_LABELS = (r"$\eta_0$", r"$\eta_1$")
CONVENTIONAL_LABELS = (r"$\Omega_m$", r"$S_8$")

# Learned distillery expressions for theta = (Omega_m, sigma_8).
DEFAULT_THETA_TO_ETA_EXPRS = (
    "-0.9*X1**0.143792 + 1.0*X1*X2",
    "0.5*X1 - 0.5",
)
DEFAULT_ETA_TO_THETA_EXPRS = (
    "2.0*X2 + 1.0",
    "0.5*(1.0*X1 + 0.9*(2.0*X2 + 1.0)**0.143792)/(1.0*X2 + 0.5)",
)

# Conventional comparison: (Omega_m, S_8 = sigma_8 * sqrt(Omega_m/0.3)).
DEFAULT_THETA_TO_CONVENTIONAL_EXPRS = (
    "X1",
    "sqrt(X1 / 0.3) * X2",
)
DEFAULT_CONVENTIONAL_TO_THETA_EXPRS = (
    "X1",
    "X2 / sqrt(X1 / 0.3)",
)

# Per-tomo-bin observation noise variances, matching the user's data-loader.
DEFAULT_NOISEVARS = (0.00045021, 0.00087473, 0.00134725, 0.00183411)


@dataclass
class WlConfig:
    """2-pt and noise configuration matching the user's data-loader."""

    L: float = 250.0          # Mpc/h (patch size)
    Lz: float = 4000.0        # Mpc/h (LOS depth)
    N: int = 128              # transverse pixel count per side
    Nz: int = 512             # LOS sampling
    cls_outbins: int = 6      # number of multipole bins
    num_tomo: int = 4         # number of tomographic bins
    cl_cut: int = -1          # high-ell cut
    noise_scale: float = 0.25
    noisevars: tuple[float, ...] = field(default_factory=lambda: tuple(DEFAULT_NOISEVARS))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_floats(token: str) -> list[float]:
    """Parse a comma-separated list of floats."""
    return [float(t) for t in token.split(",") if t.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep weak-lensing NPE validation log_prob versus number of training "
            "simulations, comparing theta, learned-eta, and conventional-coordinate baselines."
        )
    )

    # --- sweep / IO ---
    parser.add_argument("--nsims", nargs="+", type=int, default=[100, 500, 1000, 2500, 5000])
    parser.add_argument("--out-dir", type=Path, default=Path("wl_2d_nsims_logprob_sweep"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(
            "/content/drive/MyDrive/Colab Notebooks/wl-sbi/"
            "prior_S8_L_250_N_128_Nz_512.npz"
        ),
        help=(
            "Path to the .npz archive holding the noise-free WL dataset. "
            "Expected keys: 'prior_sims' (shape (N, num_tomo, N_pix, N_pix)) "
            "and 'prior_theta' (shape (N, 2) with columns (Omega_m, S_8))."
        ),
    )
    parser.add_argument(
        "--prior-theta-mode",
        choices=("S8", "sigma8"),
        default="S8",
        help=(
            "Whether the loaded 'prior_theta' columns are (Omega_m, S_8) or "
            "(Omega_m, sigma_8). Defaults to S8 (matches the user's data-loader); "
            "the script always trains in (Omega_m, sigma_8)."
        ),
    )

    # --- noise / data settings ---
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.25,
        help="Multiplicative scale on the per-bin noise std added to each tomo map.",
    )
    parser.add_argument(
        "--noisevars",
        type=_parse_floats,
        default=list(DEFAULT_NOISEVARS),
        help=(
            "Comma-separated per-tomo-bin shape-noise variances. Length must equal "
            "--num-tomo. Defaults to the user's "
            f"{tuple(DEFAULT_NOISEVARS)}."
        ),
    )
    parser.add_argument("--num-tomo", type=int, default=4)
    parser.add_argument("--cls-outbins", type=int, default=6)
    parser.add_argument("--cl-cut", type=int, default=-1)
    parser.add_argument("--patch-size-mpc-h", type=float, default=250.0,
                        help="Transverse patch size L in Mpc/h.")
    parser.add_argument("--los-depth-mpc-h", type=float, default=4000.0,
                        help="LOS depth Lz in Mpc/h.")
    parser.add_argument("--pixel-count", type=int, default=128,
                        help="Transverse pixel count per side, must match the file.")
    parser.add_argument("--los-pixels", type=int, default=512,
                        help="LOS sampling Nz, must match the file.")
    parser.add_argument(
        "--noise-chunk-size",
        type=int,
        default=64,
        help="Batch size for vmapped noise + 2pt over the loaded simulations.",
    )

    # --- test set / examples ---
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--n-examples", type=int, default=8)
    parser.add_argument("--n-posterior-samples", type=int, default=10000)
    parser.add_argument(
        "--fom-nsims",
        type=int,
        default=1000,
        help="Simulation count at which to save posterior FoM comparisons.",
    )

    # --- prior bounds ---
    parser.add_argument(
        "--theta-prior-low",
        type=_parse_floats,
        default=None,
        help=(
            "Comma-separated theta lower bounds (Omega_m, sigma_8). "
            "If unset, inferred from the loaded theta with a 1%% margin."
        ),
    )
    parser.add_argument(
        "--theta-prior-high",
        type=_parse_floats,
        default=None,
        help=(
            "Comma-separated theta upper bounds (Omega_m, sigma_8). "
            "If unset, inferred from the loaded theta with a 1%% margin."
        ),
    )

    # --- coverage ---
    parser.add_argument(
        "--run-coverage",
        action="store_true",
        help="Optionally export posterior coverage diagnostics for the coverage nsims case.",
    )
    parser.add_argument(
        "--coverage-all-nsims",
        action="store_true",
        help="When --run-coverage is set, run coverage diagnostics at every --nsims value.",
    )
    parser.add_argument(
        "--coverage-nsims",
        type=int,
        default=None,
        help="Simulation count for coverage export; defaults to --fom-nsims. Ignored by --coverage-all-nsims.",
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

    # --- training ---
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--validation-smoothing-window",
        type=int,
        default=10,
        help="Moving-average window for selecting the best validation epoch; set to 1 to disable.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--repeats-maf", type=int, default=2)
    parser.add_argument("--hidden-features", type=int, default=50)
    parser.add_argument("--num-transforms", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    # --- coordinate expressions ---
    parser.add_argument(
        "--theta-to-eta",
        nargs=2,
        default=DEFAULT_THETA_TO_ETA_EXPRS,
        metavar=("ETA0", "ETA1"),
        help="Two expressions in X1=Omega_m, X2=sigma_8 mapping theta -> eta.",
    )
    parser.add_argument(
        "--eta-to-theta",
        nargs=2,
        default=DEFAULT_ETA_TO_THETA_EXPRS,
        metavar=("THETA0", "THETA1"),
        help="Two expressions in X1=eta_0, X2=eta_1 mapping eta -> theta.",
    )
    parser.add_argument(
        "--theta-to-conventional",
        nargs=2,
        default=DEFAULT_THETA_TO_CONVENTIONAL_EXPRS,
        metavar=("CONVENTIONAL0", "CONVENTIONAL1"),
        help="Two expressions in X1=Omega_m, X2=sigma_8 mapping theta -> conventional coords.",
    )
    parser.add_argument(
        "--conventional-to-theta",
        nargs=2,
        default=DEFAULT_CONVENTIONAL_TO_THETA_EXPRS,
        metavar=("THETA0", "THETA1"),
        help="Two inverse expressions mapping the conventional coords back to (Omega_m, sigma_8).",
    )
    parser.add_argument(
        "--compare-conventional-coords",
        action="store_true",
        help="Also train/evaluate a conventional-coordinate baseline (default: (Omega_m, S_8)).",
    )
    parser.add_argument(
        "--jacobian-correction-samples",
        type=int,
        default=512,
        help="Number of training points used to estimate mean log|d theta / d coord|.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Coordinate-expression helpers
# -----------------------------------------------------------------------------

def make_expression_transform(expressions: tuple[str, str] | list[str]):
    """Build a NumPy-vectorized transform from expressions in X1, X2."""
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
        "logAbs": lambda x: np.log(np.abs(x)),
        "maximum": np.maximum,
        "minimum": np.minimum,
        "pi": np.pi,
        "sin": np.sin,
        "sqrt": np.sqrt,
        "tan": np.tan,
    }
    compiled = [
        compile(expr.replace("^", "**"), f"<coordinate:{idx}>", "eval")
        for idx, expr in enumerate(expressions)
    ]

    def transform(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        local_names = {"X1": x[:, 0], "X2": x[:, 1]}
        cols = [
            eval(code, {"__builtins__": {}}, allowed_names | local_names)
            for code in compiled
        ]
        return np.column_stack(cols).astype(np.float32)

    return transform


def check_inverse(
    theta: np.ndarray,
    forward_fn,
    inverse_fn,
    name: str,
    atol: float = 1e-4,
) -> None:
    """Verify inverse(forward(theta)) recovers theta."""
    fwd = forward_fn(theta)
    rec = inverse_fn(fwd)
    rel_err = np.abs(rec - theta) / np.maximum(np.abs(theta), 1e-12)
    max_rel_err = float(np.nanmax(rel_err))
    if not np.isfinite(max_rel_err) or max_rel_err > atol:
        raise ValueError(
            f"{name}: inverse consistency check failed. "
            f"max relative error={max_rel_err:.3e}, threshold={atol:.3e}."
        )


# -----------------------------------------------------------------------------
# Data loading: noise injection + 2-pt compression
# -----------------------------------------------------------------------------

def indices_vector(num_tomo: int) -> list[list[int]]:
    """Auto- and cross-bin indices for the tomographic 2pt calculation."""
    indices = []
    for cat_a in range(num_tomo):
        for cat_b in range(cat_a, num_tomo):
            indices.append([cat_a, cat_b])
    return indices


@partial(jax.jit, static_argnames=("size_mpc_h",))
def _compute_auto_cross_angular_power_spectrum(
    field1: jnp.ndarray,
    field2: jnp.ndarray,
    distance: float,
    size_mpc_h: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Auto/cross 2-D angular power spectrum (matches the user's data-loader)."""
    nx, ny = field1.shape
    npix = nx * ny

    f1 = jnp.fft.fftn(field1)
    f2 = jnp.fft.fftn(field2)

    theta = size_mpc_h / distance
    ell_fundamental = 2.0 * jnp.pi / theta

    power_2d = (jnp.abs(f1 * jnp.conj(f2)).astype(jnp.float32)) / (npix ** 2.0)

    kx = jnp.fft.fftfreq(nx, d=1.0) * nx
    ky = jnp.fft.fftfreq(ny, d=1.0) * ny
    k_modes = jnp.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    ell_vals = (k_modes * ell_fundamental).flatten()

    ell_bins = jnp.arange(0.5, ny // 2 + 1, 1.0) * ell_fundamental
    ell_centers = 0.5 * (ell_bins[1:] + ell_bins[:-1])
    binned, _ = jnp.histogram(ell_vals, weights=power_2d.flatten(), bins=ell_bins)
    counts, _ = jnp.histogram(ell_vals, bins=ell_bins)
    cl = (binned / counts) * theta ** 2
    return ell_centers, cl


def make_cls_compressor(cfg: WlConfig):
    """Return a JAX function mapping (num_tomo, N, N) maps -> flat log C_ell vector."""
    indices = jnp.array(indices_vector(cfg.num_tomo))
    chi_grid = (jnp.arange(cfg.Nz) + 0.5) * cfg.Lz / float(cfg.Nz)
    chi_source = chi_grid[-1]
    cl_cut = cfg.cl_cut
    num_bins = cfg.cls_outbins
    patch_size = cfg.L

    def cls_for_one(tomo_data: jnp.ndarray) -> jnp.ndarray:
        def get_spec(index):
            ell, cl = _compute_auto_cross_angular_power_spectrum(
                tomo_data[index[0]],
                tomo_data[index[1]],
                chi_source,
                patch_size,
            )
            return jnp.histogram(ell[:cl_cut], weights=cl[:cl_cut], bins=num_bins)[0]

        return jax.vmap(get_spec)(indices).flatten()

    return jax.jit(cls_for_one)


def make_noise_simulator(cfg: WlConfig):
    """Return a JAX function adding white shape noise to a (num_tomo, N, N) map."""
    noisevars = jnp.asarray(cfg.noisevars, dtype=jnp.float32).reshape(-1, 1, 1)
    noise_scale = float(cfg.noise_scale)
    n = int(cfg.N)
    num_tomo = int(cfg.num_tomo)

    def add_noise(key: jnp.ndarray, sim: jnp.ndarray) -> jnp.ndarray:
        noise = jr.normal(key, shape=(num_tomo, n, n)) * noise_scale * jnp.sqrt(noisevars)
        return sim + noise

    return jax.jit(add_noise)


def load_and_compress(
    data_path: Path,
    cfg: WlConfig,
    seed: int,
    chunk_size: int,
    prior_theta_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the WL archive, add JAX noise, and return ``(theta_sigma8, log_cls)``.

    The archive is expected to expose:
      - ``prior_sims`` of shape ``(N_total, num_tomo, N, N)``;
      - ``prior_theta`` of shape ``(N_total, 2)`` with columns either
        ``(Omega_m, S_8)`` (default) or ``(Omega_m, sigma_8)``.
    """
    archive = np.load(data_path)
    if "prior_sims" not in archive.files or "prior_theta" not in archive.files:
        raise ValueError(
            f"Expected keys 'prior_sims' and 'prior_theta' in {data_path}, "
            f"got {archive.files}."
        )

    sims = jnp.asarray(archive["prior_sims"], dtype=jnp.float32)
    raw_theta = np.asarray(archive["prior_theta"], dtype=np.float32)

    expected_shape = (cfg.num_tomo, cfg.N, cfg.N)
    if sims.shape[1:] != expected_shape:
        raise ValueError(
            f"prior_sims has shape {sims.shape}, expected (N, {expected_shape}). "
            f"Adjust --num-tomo / --pixel-count to match."
        )
    if raw_theta.ndim != 2 or raw_theta.shape[1] != 2:
        raise ValueError(
            f"prior_theta has shape {raw_theta.shape}, expected (N, 2)."
        )
    if raw_theta.shape[0] != sims.shape[0]:
        raise ValueError(
            f"prior_theta has {raw_theta.shape[0]} rows but prior_sims has "
            f"{sims.shape[0]}; they must match."
        )

    if prior_theta_mode == "S8":
        omega_m = raw_theta[:, 0]
        s8 = raw_theta[:, 1]
        sigma_8 = s8 / np.sqrt(np.clip(omega_m / 0.3, 1e-12, None))
        theta_sigma8 = np.stack([omega_m, sigma_8], axis=1).astype(np.float32)
    elif prior_theta_mode == "sigma8":
        theta_sigma8 = raw_theta.astype(np.float32)
    else:
        raise ValueError(f"Unknown --prior-theta-mode: {prior_theta_mode!r}.")

    add_noise = make_noise_simulator(cfg)
    cls_for_one = make_cls_compressor(cfg)

    n_total = int(sims.shape[0])
    master_key = jr.PRNGKey(int(seed))
    keys = jr.split(master_key, num=n_total)

    log_cls_chunks = []
    for start in tqdm(range(0, n_total, chunk_size), desc="noise + 2pt"):
        end = min(start + chunk_size, n_total)
        sims_chunk = sims[start:end]
        keys_chunk = keys[start:end]
        noisy = jax.vmap(add_noise)(keys_chunk, sims_chunk)
        cls = jax.vmap(cls_for_one)(noisy)
        log_cls_chunks.append(np.asarray(jnp.log(cls), dtype=np.float32))

    log_cls = np.concatenate(log_cls_chunks, axis=0)
    return theta_sigma8, log_cls


# -----------------------------------------------------------------------------
# Prior helpers and Jacobian estimation
# -----------------------------------------------------------------------------

def in_theta_prior(
    theta: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    finite = np.isfinite(theta).all(axis=1)
    above = (theta >= low).all(axis=1)
    below = (theta <= high).all(axis=1)
    return finite & above & below


def inverse_logdet_jacobian(
    coords: np.ndarray,
    coords_to_theta_fn,
    max_points: int,
    seed: int,
) -> float:
    """Estimate mean log|det(d theta / d coords)|."""
    rng = np.random.default_rng(seed)
    if coords.shape[0] > max_points:
        coords = coords[rng.choice(coords.shape[0], size=max_points, replace=False)]

    logdets = []
    for point in coords:
        jac = np.empty((2, 2), dtype=np.float64)
        for dim in range(2):
            step = 1e-4 * max(1.0, abs(float(point[dim])))
            plus = point.astype(np.float64).copy()
            minus = point.astype(np.float64).copy()
            plus[dim] += step
            minus[dim] -= step
            jac[:, dim] = (
                coords_to_theta_fn(plus[None, :])[0].astype(np.float64)
                - coords_to_theta_fn(minus[None, :])[0].astype(np.float64)
            ) / (2.0 * step)
        det = np.linalg.det(jac)
        if np.isfinite(det) and abs(det) > 0.0:
            logdets.append(math.log(abs(det)))

    if not logdets:
        raise RuntimeError("Could not compute any finite inverse-Jacobian determinants.")
    return float(np.mean(logdets))


# -----------------------------------------------------------------------------
# NPE training / sampling
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Metrics / FoM
# -----------------------------------------------------------------------------

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
                "logdet_dtheta_dcoord_correction": logdet_correction,
            }
        )
    return rows


def dataframe_to_npz(path: Path, frame: pd.DataFrame) -> None:
    arrays = {column: frame[column].to_numpy() for column in frame.columns}
    np.savez_compressed(path, **arrays)


def theta_fom(samples: np.ndarray) -> tuple[float, int]:
    """Return 1/sqrt(det(cov(Omega_m, sigma_8))) after dropping invalid samples."""
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
    if theta_key not in posterior_arrays:
        return pd.DataFrame()
    theta_samples = posterior_arrays[theta_key]
    theta_true = posterior_arrays["theta_true"]
    example_indices = posterior_arrays["example_indices"]
    comparison_methods = [
        method
        for method in ("eta", "conventional_coordinates")
        if f"n{nsims}_{method}_theta_samples" in posterior_arrays
    ]

    rows = []
    for example_number in range(theta_samples.shape[0]):
        fom_theta_, n_valid_theta = theta_fom(theta_samples[example_number])
        row = {
            "nsims": nsims,
            "example_number": example_number,
            "example_index": int(example_indices[example_number]),
            "theta_true_omega_m": float(theta_true[example_number, 0]),
            "theta_true_sigma_8": float(theta_true[example_number, 1]),
            "fom_theta": fom_theta_,
            "n_valid_theta_samples": n_valid_theta,
        }
        for method in comparison_methods:
            method_samples = posterior_arrays[f"n{nsims}_{method}_theta_samples"]
            fom_method, n_valid_method = theta_fom(method_samples[example_number])
            row[f"fom_{method}"] = fom_method
            row[f"fom_{method}_over_theta"] = float(fom_method / fom_theta_)
            row[f"n_valid_{method}_samples"] = n_valid_method
        rows.append(row)
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

    if f"n{fom_nsims}_theta_theta_samples" in posterior_arrays:
        fom_comparison = build_fom_comparison(posterior_arrays, fom_nsims)
        if not fom_comparison.empty:
            fom_comparison.to_csv(out_dir / "fom_comparison.csv", index=False)
            dataframe_to_npz(out_dir / "fom_comparison.npz", fom_comparison)


# -----------------------------------------------------------------------------
# Coverage diagnostics
# -----------------------------------------------------------------------------

def compute_coverage_diagnostics(
    posterior_samples: np.ndarray,
    theta_true: np.ndarray,
    seed: int,
    nbins: int = 10,
) -> dict[str, np.ndarray]:
    """Compute exportable data behind ltu-ili's one-dimensional coverage plots."""
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


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    if len(args.noisevars) != args.num_tomo:
        raise ValueError(
            f"--noisevars has {len(args.noisevars)} entries but --num-tomo is "
            f"{args.num_tomo}; they must match."
        )

    cfg = WlConfig(
        L=args.patch_size_mpc_h,
        Lz=args.los_depth_mpc_h,
        N=args.pixel_count,
        Nz=args.los_pixels,
        cls_outbins=args.cls_outbins,
        num_tomo=args.num_tomo,
        cl_cut=args.cl_cut,
        noise_scale=args.noise_scale,
        noisevars=tuple(args.noisevars),
    )

    print(f"Using device: {device}")
    print(f"Loading WL dataset from {args.data_path}")
    print(
        f"Noise: scale={cfg.noise_scale} on per-tomo vars={list(cfg.noisevars)}; "
        f"compressing with {cfg.num_tomo} tomo bins x {cfg.cls_outbins} ell bins."
    )

    theta_all, data_all = load_and_compress(
        args.data_path,
        cfg,
        seed=args.seed,
        chunk_size=args.noise_chunk_size,
        prior_theta_mode=args.prior_theta_mode,
    )
    n_total = theta_all.shape[0]
    print(f"Loaded {n_total} simulations; each compressed to {data_all.shape[1]} log C_ell bins.")

    max_nsims = max(args.nsims)
    if max_nsims + args.n_test > n_total:
        raise ValueError(
            f"max(nsims)={max_nsims} + n_test={args.n_test} > available={n_total}; "
            "reduce --nsims or --n-test."
        )

    index = np.arange(n_total)
    ind_train, ind_test = train_test_split(index, test_size=args.n_test, random_state=42)
    if ind_train.shape[0] < max_nsims:
        raise ValueError(
            f"After 50/50-style split, train pool size {ind_train.shape[0]} is "
            f"below max(nsims)={max_nsims}; reduce --n-test or --nsims."
        )
    ind_train = ind_train[:max_nsims]
    theta_pool = theta_all[ind_train]
    data_pool = data_all[ind_train]
    theta_test = theta_all[ind_test]
    data_test = data_all[ind_test]

    if args.theta_prior_low is None:
        margin = 0.01
        low = theta_all.min(axis=0).astype(np.float32)
        span = (theta_all.max(axis=0) - theta_all.min(axis=0)).astype(np.float32)
        theta_prior_low = (low - margin * span).astype(np.float32)
    else:
        theta_prior_low = np.asarray(args.theta_prior_low, dtype=np.float32)

    if args.theta_prior_high is None:
        margin = 0.01
        high = theta_all.max(axis=0).astype(np.float32)
        span = (theta_all.max(axis=0) - theta_all.min(axis=0)).astype(np.float32)
        theta_prior_high = (high + margin * span).astype(np.float32)
    else:
        theta_prior_high = np.asarray(args.theta_prior_high, dtype=np.float32)

    if theta_prior_low.shape != (2,) or theta_prior_high.shape != (2,):
        raise ValueError("Theta prior bounds must have shape (2,).")

    print(
        f"Theta prior box: low={theta_prior_low.tolist()} "
        f"high={theta_prior_high.tolist()}."
    )

    theta_to_eta = make_expression_transform(args.theta_to_eta)
    eta_to_theta = make_expression_transform(args.eta_to_theta)
    check_inverse(
        np.concatenate([theta_pool[:128], theta_test[:128]], axis=0),
        theta_to_eta,
        eta_to_theta,
        "theta <-> eta",
    )
    if args.compare_conventional_coords:
        theta_to_conventional = make_expression_transform(args.theta_to_conventional)
        conventional_to_theta = make_expression_transform(args.conventional_to_theta)
        check_inverse(
            np.concatenate([theta_pool[:128], theta_test[:128]], axis=0),
            theta_to_conventional,
            conventional_to_theta,
            "theta <-> conventional",
        )
    else:
        theta_to_conventional = None
        conventional_to_theta = None

    if args.n_examples > theta_test.shape[0]:
        raise ValueError("--n-examples cannot exceed the number of test observations.")
    example_rng = np.random.default_rng(args.seed + 10_000)
    example_indices = example_rng.choice(theta_test.shape[0], size=args.n_examples, replace=False)

    coverage_nsims = args.fom_nsims if args.coverage_nsims is None else args.coverage_nsims
    coverage_nsims_values = list(args.nsims) if args.coverage_all_nsims else [coverage_nsims]
    if args.run_coverage:
        missing = sorted(set(coverage_nsims_values) - set(args.nsims))
        if missing:
            raise ValueError(f"Coverage nsims must be included in --nsims: {missing}")

    coverage_indices = np.array([], dtype=np.int64)
    if args.run_coverage:
        coverage_n_test = min(args.coverage_n_test, theta_test.shape[0])
        coverage_seed = args.seed + 20_000 if args.coverage_seed is None else args.coverage_seed
        coverage_rng = np.random.default_rng(coverage_seed)
        coverage_indices = coverage_rng.choice(
            theta_test.shape[0], size=coverage_n_test, replace=False
        )

    np.savez_compressed(
        args.out_dir / "dataset_reference.npz",
        theta_test=theta_test.astype(np.float32),
        data_test=data_test.astype(np.float32),
        example_indices=example_indices.astype(np.int64),
        theta_examples=theta_test[example_indices].astype(np.float32),
        data_examples=data_test[example_indices].astype(np.float32),
        theta_prior_low=theta_prior_low,
        theta_prior_high=theta_prior_high,
        train_indices=ind_train.astype(np.int64),
        test_indices=ind_test.astype(np.int64),
        noise_scale=np.asarray(cfg.noise_scale, dtype=np.float32),
        noisevars=np.asarray(cfg.noisevars, dtype=np.float32),
        cls_outbins=np.asarray(cfg.cls_outbins, dtype=np.int64),
        num_tomo=np.asarray(cfg.num_tomo, dtype=np.int64),
        cl_cut=np.asarray(cfg.cl_cut, dtype=np.int64),
    )

    metrics_rows: list[dict[str, Any]] = []
    history_arrays: dict[str, np.ndarray] = {}
    coverage_arrays: dict[str, np.ndarray] = {}
    if args.run_coverage:
        coverage_arrays = {
            "coverage_nsims": np.asarray(coverage_nsims_values, dtype=np.int64),
            "coverage_all_nsims": np.asarray(args.coverage_all_nsims),
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
    methods = ["theta", "eta"]
    if args.compare_conventional_coords:
        methods.append("conventional_coordinates")

    for nsims in args.nsims:
        train_theta = theta_pool[:nsims]
        train_data = data_pool[:nsims]

        for method_idx, method in enumerate(methods):
            start = time.time()
            run_seed = args.seed + nsims * 10 + method_idx
            print(f"\nTraining method={method}, nsims={nsims}, seed={run_seed}")

            if method == "theta":
                train_params = train_theta
                prior_low = theta_prior_low
                prior_high = theta_prior_high
                logdet_correction = 0.0
            elif method == "eta":
                train_params = theta_to_eta(train_theta)
                prior_low = train_params.min(axis=0)
                prior_high = train_params.max(axis=0)
                logdet_correction = inverse_logdet_jacobian(
                    train_params,
                    coords_to_theta_fn=eta_to_theta,
                    max_points=args.jacobian_correction_samples,
                    seed=run_seed,
                )
            elif method == "conventional_coordinates":
                assert theta_to_conventional is not None and conventional_to_theta is not None
                train_params = theta_to_conventional(train_theta)
                prior_low = train_params.min(axis=0)
                prior_high = train_params.max(axis=0)
                logdet_correction = inverse_logdet_jacobian(
                    train_params,
                    coords_to_theta_fn=conventional_to_theta,
                    max_points=args.jacobian_correction_samples,
                    seed=run_seed,
                )
            else:
                raise ValueError(f"Unknown method: {method!r}")

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

            if args.run_coverage and nsims in coverage_nsims_values:
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
                    elif method == "eta":
                        theta_samples = eta_to_theta(raw_samples)
                    elif method == "conventional_coordinates":
                        theta_samples = conventional_to_theta(raw_samples)
                    valid_mask = np.isfinite(theta_samples).all(axis=1)
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
                    valid_mask = in_theta_prior(theta_samples, theta_prior_low, theta_prior_high)
                elif method == "eta":
                    theta_samples = eta_to_theta(raw_samples)
                    valid_mask = in_theta_prior(theta_samples, theta_prior_low, theta_prior_high)
                elif method == "conventional_coordinates":
                    theta_samples = conventional_to_theta(raw_samples)
                    valid_mask = in_theta_prior(theta_samples, theta_prior_low, theta_prior_high)
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
            save_coverage_outputs(args.out_dir, coverage_arrays)
            np.savez_compressed(args.out_dir / "training_histories.npz", **history_arrays)
            np.savez_compressed(args.out_dir / "posterior_samples.npz", **posterior_arrays)
            print(f"Finished method={method}, nsims={nsims} in {elapsed / 60.0:.1f} min")

    manifest = {
        "config": {
            "args": vars(args) | {
                "out_dir": str(args.out_dir),
                "data_path": str(args.data_path),
                "device_resolved": device,
            },
            "wl": asdict(cfg),
            "theta_labels": THETA_LABELS,
            "eta_labels": ETA_LABELS,
            "conventional_labels": CONVENTIONAL_LABELS,
            "theta_prior_low": theta_prior_low.tolist(),
            "theta_prior_high": theta_prior_high.tolist(),
            "theta_to_eta_expressions": list(args.theta_to_eta),
            "eta_to_theta_expressions": list(args.eta_to_theta),
            "theta_to_conventional_expressions": list(args.theta_to_conventional),
            "conventional_to_theta_expressions": list(args.conventional_to_theta),
            "compare_conventional_coords": bool(args.compare_conventional_coords),
            "methods": methods,
            "coverage_nsims_values": coverage_nsims_values if args.run_coverage else [],
            "coverage_all_nsims": bool(args.coverage_all_nsims),
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
            "Training is done in (Omega_m, sigma_8); 'prior_theta' from the data file "
            "is converted from (Omega_m, S_8) to (Omega_m, sigma_8) when "
            "--prior-theta-mode=S8 (default).",
            "For eta and conventional runs, best_validation_log_prob_raw is in the alternate "
            "coordinate density; the theta-density variant subtracts the mean "
            "log|det(d theta / d coord)| estimated by finite differences.",
            "Use --compare-conventional-coords to add the (Omega_m, S_8) baseline.",
            "Noise is re-injected on every load; sweep --noise-scale (and rerun with "
            "different --out-dir values) to study the noisier regime.",
            "FoM is 1/sqrt(det(cov(Omega_m, sigma_8))) for --fom-nsims, default 1000.",
            "Coverage outputs are saved only when --run-coverage is set; "
            "--coverage-all-nsims runs them at every --nsims value.",
        ],
    }
    with open(args.out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Results written to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
