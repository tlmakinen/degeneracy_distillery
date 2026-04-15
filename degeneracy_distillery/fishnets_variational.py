"""
Variational / MDN extension for FishNets-style Fisher inference.

Goal: after training **one** network, approximate the Fisher uncertainty that you would
otherwise get from an **ensemble**, by sampling Fisher matrices from a conditional density
q(z | x) (mixture of Gaussians on Cholesky logits z) and mapping z -> F with
`construct_fisher_matrix_log_cholesky`. See `sample_fisher_mdn` and
`sample_student_fisher_mdn`.

Training uses a two-stage pipeline:
  1) Pretrain a point FishNet teacher (exposes logits z before the Cholesky map).
  2) Train a student with the usual Gaussian KL on (theta_hat, F_mean) plus an MDN
     term that fits q(z|x) to teacher targets (or, for richer uncertainty, replace those
     targets with multi-sample / ensemble / bootstrap Fishers).

Inference vs training: you avoid **many forward passes through many large nets** at
inference time; you still need a training signal for the *width* of q(F|x). A single
point teacher tends to collapse the MDN unless you add jitter, simulation replicates at
fixed theta, or one-off ensemble distillation.
"""
from __future__ import annotations

import os
import shutil
from typing import Callable, Sequence, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flax import linen as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

try:
    from .fishnets import (
        SoftSquelch,
        construct_fisher_matrix_log_cholesky,
        optimized_smooth_leaky,
        resMLP,
    )
    from .io_utils import create_results_dict
except ImportError:
    from fishnets import (
        SoftSquelch,
        construct_fisher_matrix_log_cholesky,
        optimized_smooth_leaky,
        resMLP,
    )
    from io_utils import create_results_dict


def _fisher_kl_term(mle: jnp.ndarray, F: jnp.ndarray, theta_batched: jnp.ndarray) -> jnp.ndarray:
    res = theta_batched - mle
    sign, logdet = jnp.linalg.slogdet(F)
    del sign
    logdet = jnp.clip(logdet, -50, 50)
    quad = jnp.einsum("ij,ij->i", res, jnp.einsum("ijk,ik->ij", F, res))
    return 0.5 * jnp.mean(quad - logdet, axis=0)


def mdn_log_prob_diagonal(
    pi_logits: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    z: jnp.ndarray,
) -> jnp.ndarray:
    """
    Mixture of Gaussians with diagonal covariances per component.

    pi_logits: (batch, K)
    mu, sigma: (batch, K, D)
    z: (batch, D)
    Returns log p(z) per batch element (batch,).
    """
    log_pi = jax.nn.log_softmax(pi_logits, axis=-1)
    z_exp = z[:, None, :]
    var = sigma**2 + 1e-8
    log_nd = -0.5 * (z_exp - mu) ** 2 / var - 0.5 * jnp.log(2 * jnp.pi * var)
    log_p_k = jnp.sum(log_nd, axis=-1) + log_pi
    return jax.scipy.special.logsumexp(log_p_k, axis=-1)


def sample_fisher_mdn(
    key: jnp.ndarray,
    pi_logits: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    n_p: int,
    num_samples: int = 1,
) -> jnp.ndarray:
    """
    Sample Fisher matrices from the MDN over Cholesky logits (one categorical draw per
    sample, then a diagonal Gaussian draw in z, then PD map).

    Parameters
    ----------
    key : PRNGKey
    pi_logits : (K,) or (batch, K) mixture logits (unnormalized).
    mu, sigma : (K, D) or (batch, K, D) with D = n_p * (n_p + 1) // 2.
    n_p : parameter dimension (F is n_p x n_p).
    num_samples : independent draws per observation.

    Returns
    -------
    F_samples
        Shape (num_samples, n_p, n_p) for a single x, or (batch, num_samples, n_p, n_p)
        when pi_logits is batched.
    """
    squeeze_batch = False
    if pi_logits.ndim == 1:
        pi_logits = pi_logits[None, :]
        mu = mu[None, ...]
        sigma = sigma[None, ...]
        squeeze_batch = True

    def draw_for_one(obs_key: jnp.ndarray, logits: jnp.ndarray, mu_b: jnp.ndarray, sig_b: jnp.ndarray):
        sks = jr.split(obs_key, num_samples)

        def one_draw(sk: jnp.ndarray):
            k0, k1 = jr.split(sk)
            comp = jr.categorical(k0, logits)
            z = mu_b[comp] + sig_b[comp] * jr.normal(k1, shape=mu_b[comp].shape)
            return construct_fisher_matrix_log_cholesky(z, n_p)

        return jax.vmap(one_draw)(sks)

    batch_keys = jr.split(key, pi_logits.shape[0])
    out = jax.vmap(draw_for_one)(batch_keys, pi_logits, mu, sigma)
    if squeeze_batch:
        out = out[0]
    return out


def forward_variational_student(model: nn.Module, w, x: jnp.ndarray, n_p: int | None = None):
    """
    Run the trained variational student: x is (batch, data_dim) or (data_dim,) for one obs.

    Parameters
    ----------
    n_p
        Parameter dimension; if None, inferred from ``t_hat.shape[-1]`` (set explicitly
        when jitting if shape inference is ambiguous).

    Returns
    -------
    theta_hat, F_mean, z_mean, z_mix_mean, pi, pi_logits, mu, sigma
        ``pi`` is mixture weights (softmax of ``pi_logits``).
    """
    squeeze = x.ndim == 1
    if squeeze:
        x = x[None, :]
    t_hat, z_mean, pi_logits, mu, sigma = jax.vmap(lambda xx: model.apply(w, xx))(x)
    n_p_i = int(n_p) if n_p is not None else int(t_hat.shape[-1])
    F_mean = jax.vmap(lambda z: construct_fisher_matrix_log_cholesky(z, n_p_i))(z_mean)
    pi = jax.nn.softmax(pi_logits, axis=-1)
    if pi_logits.ndim == 1:
        z_mix_mean = jnp.sum(pi[:, None] * mu, axis=0)
    else:
        z_mix_mean = jnp.einsum("bk,bkd->bd", pi, mu)
    if squeeze:
        t_hat = t_hat[0]
        F_mean = F_mean[0]
        z_mean = z_mean[0]
        z_mix_mean = z_mix_mean[0]
        pi = pi[0]
        pi_logits = pi_logits[0]
        mu = mu[0]
        sigma = sigma[0]
    return t_hat, F_mean, z_mean, z_mix_mean, pi, pi_logits, mu, sigma


def sample_student_fisher_mdn(
    key: jnp.ndarray,
    model: nn.Module,
    w,
    x: jnp.ndarray,
    num_samples: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    One-call inference: forward pass + Fisher samples from q(z|x).

    Returns
    -------
    theta_hat, F_samples
        For a single observation x (data_dim,), theta_hat has shape (n_p,) and
        F_samples (num_samples, n_p, n_p). For a batch x (B, data_dim), theta_hat is
        (B, n_p) and F_samples is (B, num_samples, n_p, n_p).
    """
    t_hat, _F, _zm, _zmm, _pi, pi_logits, mu, sigma = forward_variational_student(model, w, x)
    n_p = int(t_hat.shape[-1])
    if t_hat.ndim == 1:
        F_s = sample_fisher_mdn(key, pi_logits, mu, sigma, n_p, num_samples=num_samples)
        return t_hat, F_s
    keys = jr.split(key, t_hat.shape[0])
    F_s = jax.vmap(
        lambda k, pl, m, s: sample_fisher_mdn(k, pl, m, s, n_p, num_samples=num_samples)
    )(keys, pi_logits, mu, sigma)
    return t_hat, F_s


class FishnetTeacherWithZ(nn.Module):
    """Same mapping as Fishnet_from_embedding but returns raw Cholesky logits z."""

    n_p: int
    hidden: int
    act: Callable
    act_fisher: Callable = nn.gelu
    sharpness: float = 5.0
    threshold: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        t = self.act(nn.Dense(self.hidden)(x))
        fisher_h = nn.Dense(self.hidden)(x)
        fisher_h = SoftSquelch(threshold=self.threshold, sharpness=self.sharpness)(fisher_h)
        t = nn.Dense(self.n_p)(t)
        z = nn.Dense((self.n_p * (self.n_p + 1)) // 2)(fisher_h)
        F = construct_fisher_matrix_log_cholesky(z, self.n_p)
        return t, z, F


class VariationalFishnetMDN(nn.Module):
    """
    Heads on top of a fixed-size embedding: MLE, mean Cholesky logits, and MDN over z.
    """

    n_p: int
    hidden: int
    act: Callable
    num_components: int
    sharpness: float = 5.0
    threshold: float = 1.0

    @property
    def d_z(self) -> int:
        return (self.n_p * (self.n_p + 1)) // 2

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        t = self.act(nn.Dense(self.hidden)(x))
        fisher_h = nn.Dense(self.hidden)(x)
        fisher_h = SoftSquelch(threshold=self.threshold, sharpness=self.sharpness)(fisher_h)
        t = nn.Dense(self.n_p)(t)
        z_mean = nn.Dense(self.d_z)(fisher_h)

        h_mdn = self.act(nn.Dense(self.hidden)(x))
        pi_logits = nn.Dense(self.num_components)(h_mdn)
        flat_mu = nn.Dense(self.num_components * self.d_z)(h_mdn)
        flat_sig = nn.Dense(self.num_components * self.d_z)(h_mdn)
        if flat_mu.ndim == 1:
            mu = flat_mu.reshape(self.num_components, self.d_z)
            sigma_raw = flat_sig.reshape(self.num_components, self.d_z)
        else:
            bsz = flat_mu.shape[0]
            mu = flat_mu.reshape(bsz, self.num_components, self.d_z)
            sigma_raw = flat_sig.reshape(bsz, self.num_components, self.d_z)
        sigma = jax.nn.softplus(sigma_raw) + 1e-4
        return t, z_mean, pi_logits, mu, sigma


def _sample_architecture(
    key: jnp.ndarray,
    n: int,
    hids_min: int,
    hids_max: int,
    n_layers: Union[int, Sequence[int]],
    acts: list,
) -> tuple[list[int], Callable, jnp.ndarray, jnp.ndarray]:
    """One ensemble-style draw: hidden widths, act, sharpness, threshold."""
    hids_range = np.arange(hids_min, hids_max)
    key, k1, k2, k3, k4 = jr.split(key, 5)
    hidden = int(jr.choice(k1, hids_range, replace=True))
    if isinstance(n_layers, (list, tuple, np.ndarray)):
        if len(n_layers) != 2:
            raise ValueError("n_layers range must have exactly two values: [min_layers, max_layers].")
        min_layers, max_layers = int(n_layers[0]), int(n_layers[1])
        n_layers_model = int(jr.randint(k2, shape=(), minval=min_layers, maxval=max_layers + 1))
    else:
        n_layers_model = int(n_layers)
    act = acts[n % len(acts)]
    sharpness_val = jr.normal(k3, shape=(1,)) * 0.7 + 5.0
    threshold_val = jr.normal(k4, shape=(1,)) * 0.7 + 1.0
    feat = [hidden] * n_layers_model
    return feat, act, sharpness_val, threshold_val


def train_fishnets_variational(
    theta,
    data,
    theta_test,
    data_test,
    data_shape: int | None = None,
    hids_min: int = 10,
    hids_max: int = 300,
    n_layers: Union[int, Sequence[int]] = 3,
    num_models: int = 20,
    seed_model: int = 201,
    seed_train: int = 999,
    train_batch_size: int = 200,
    train_epochs: int = 4000,
    train_min_epochs: int = 100,
    patience: int = 20,
    lr: float = 5e-5,
    acts: list | None = None,
    scaler_type: str = "minmax",
    embedding_net: nn.Module | None = None,
    outdir: str = "fishnets-variational-log",
    *,
    num_mdn_components: int = 5,
    pretrain_epochs: int | None = None,
    beta_nll: float = 1.0,
):
    """
    Train a single variational FishNet student with KL(F_mean) + beta * NLL_MDN(z_teacher|x).

    After training, draw Fisher matrices from q(F|x) without an ensemble using
    ``sample_student_fisher_mdn`` or ``sample_fisher_mdn`` on the stored MDN parameters.

    The API mirrors `train_fishnets` in `training_loop_fishnets.py`. Arguments not used in
    the same way:
      - `num_models` is kept for API compatibility; only one student is trained. A single
        teacher architecture is drawn using the same rules as ensemble member 0.

    Suggestions (conceptual):
      1) **Targets for -log p(F)**: The MDN needs a conditional distribution to learn.
         Using one deterministic teacher gives a narrow target; better targets are
         empirical distributions from N simulations at the same theta, bootstrap Fishers,
         or Cholesky logits pooled from an ensemble of FishNets at each x.
      2) **Pretrain then refine**: Freezing a strong point estimate (your current FishNet
         or ensemble mean) as teacher stabilizes the MDN; tune `beta_nll` so KL and NLL
         scales match early in training.
      3) **Structured q(F)**: MDN on raw Cholesky logits ignores positive-definite
         structure beyond the mean branch; alternatives include a Wishart / LKJ factor
         or low-rank + diagonal corrections.
      4) **Single objective**: One can instead parameterize q as Gaussian on z with
         learned full covariance (low-rank + diag) and use only NLL + KL on theta with
         F = construct(z_mean) if you want a simpler variational family.
      5) **Calibration**: Check coverage of predictive intervals for F entries against
         held-out ensemble or simulation-based ground truth.

    Returns:
      student_w, teacher_w, student_model, teacher_model, data_scaler, outputs
    """
    print("saving to", outdir)
    if os.path.exists(outdir):
        for filename in os.listdir(outdir):
            file_path = os.path.join(outdir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except OSError as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(outdir)

    n_params = theta.shape[-1]
    if data_shape is None:
        data_shape = data.shape[-1]

    if scaler_type.lower() == "minmax":
        data_scaler = MinMaxScaler(feature_range=(0, 1))
    elif scaler_type.lower() == "standard":
        data_scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler_type: '{scaler_type}'. Options are 'minmax' or 'standard'.")

    data = data_scaler.fit_transform(data.reshape(-1, data_shape)).reshape(data.shape)
    data_test = data_scaler.transform(data_test.reshape(-1, data_shape)).reshape(data_test.shape)

    data = jnp.squeeze(data)
    data_test_flat = data_test.reshape(-1, data_shape)

    mish = lambda x: x * nn.tanh(nn.softplus(x))
    if acts is None:
        acts = [
            nn.relu,
            nn.relu,
            nn.relu,
            nn.leaky_relu,
            nn.leaky_relu,
            nn.leaky_relu,
            nn.leaky_relu,
            nn.swish,
            nn.swish,
            nn.swish,
            mish,
            mish,
            optimized_smooth_leaky,
            optimized_smooth_leaky,
            optimized_smooth_leaky,
            nn.gelu,
            nn.gelu,
            nn.gelu,
            nn.gelu,
            nn.gelu,
            nn.gelu,
            nn.gelu,
            nn.gelu,
        ]
    idx_acts = np.random.choice(np.arange(len(acts)), size=(num_models,))
    acts = [acts[i] for i in idx_acts]

    key = jr.PRNGKey(seed_model)
    key, arch_key, k_init_t, k_init_s = jr.split(key, 4)
    feat, act, sharpness_val, threshold_val = _sample_architecture(
        arch_key, 0, hids_min, hids_max, n_layers, acts
    )
    hidden0 = feat[0]
    print("Architecture (teacher & student trunk):", feat, "| act:", act)

    def build_teacher():
        tail = FishnetTeacherWithZ(
            n_p=n_params,
            hidden=hidden0,
            act=act,
            act_fisher=nn.gelu,
            sharpness=sharpness_val,
            threshold=threshold_val,
        )
        if embedding_net is not None:
            return nn.Sequential([embedding_net, resMLP(feat, act=act), tail])
        return nn.Sequential([resMLP(feat, act=act), tail])

    def build_student():
        tail = VariationalFishnetMDN(
            n_p=n_params,
            hidden=hidden0,
            act=act,
            num_components=num_mdn_components,
            sharpness=sharpness_val,
            threshold=threshold_val,
        )
        if embedding_net is not None:
            return nn.Sequential([embedding_net, resMLP(feat, act=act), tail])
        return nn.Sequential([resMLP(feat, act=act), tail])

    teacher_model = build_teacher()
    student_model = build_student()

    teacher_w = teacher_model.init(k_init_t, data[0])
    student_w = student_model.init(k_init_s, data[0])

    key = jr.PRNGKey(seed_train)
    key, shuffle_key = jr.split(key)
    n_train_samples = theta.shape[0]
    shuffle_idx = jr.permutation(shuffle_key, jnp.arange(n_train_samples))
    theta = theta[shuffle_idx]
    data = data[shuffle_idx]

    if pretrain_epochs is None:
        pretrain_epochs = max(200, train_epochs // 8)

    tx = optax.adam(learning_rate=lr)

    # --- Phase 1: pretrain teacher (point FishNet with z exposed) ---
    def teacher_loss_fn(w, x_b, theta_b):
        def fwd(x):
            t, _z, F = teacher_model.apply(w, x)
            return t, F

        mle, F = jax.vmap(fwd)(x_b)
        return _fisher_kl_term(mle, F, theta_b)

    teacher_tx = tx
    teacher_opt_state = teacher_tx.init(teacher_w)
    teacher_vag = jax.value_and_grad(teacher_loss_fn)

    @jax.jit
    def teacher_step(w, opt_state, x_b, theta_b):
        loss, grads = teacher_vag(w, x_b, theta_b)
        updates, opt_state = teacher_tx.update(grads, opt_state, w)
        w = optax.apply_updates(w, updates)
        return w, opt_state, loss

    n_train = (theta.reshape(-1, n_params).shape[0] // train_batch_size) * train_batch_size
    n_batch = n_train // train_batch_size

    print(f"Pretraining teacher for {pretrain_epochs} epochs...")
    key_t = jr.PRNGKey(seed_train + 7)
    pbar_pre = tqdm(range(pretrain_epochs), desc="Teacher pretrain", leave=True)
    for j in pbar_pre:
        key_t, rng = jr.split(key_t)
        randidx = jr.permutation(rng, jnp.arange(theta.shape[0]), independent=True)[:n_train]
        epoch_loss = 0.0
        for bi in range(n_batch):
            sl = slice(bi * train_batch_size, (bi + 1) * train_batch_size)
            idx = randidx[sl]
            x_b = data.reshape(-1, data_shape)[idx]
            th_b = theta.reshape(-1, n_params)[idx]
            teacher_w, teacher_opt_state, loss = teacher_step(
                teacher_w, teacher_opt_state, x_b, th_b
            )
            epoch_loss += loss
        epoch_loss /= n_batch
        pbar_pre.set_postfix(loss=float(epoch_loss))

    # --- Phase 2: student with KL + MDN NLL ---
    def student_loss_fn_batched(w, x_b, theta_b, t_w):
        t_hat, z_mean, pi_logits, mu, sigma = jax.vmap(lambda xx: student_model.apply(w, xx))(x_b)
        _tt, z_t, _Ft = jax.vmap(lambda xx: teacher_model.apply(t_w, xx))(x_b)
        F_mean = jax.vmap(lambda z: construct_fisher_matrix_log_cholesky(z, n_params))(z_mean)
        kl = _fisher_kl_term(t_hat, F_mean, theta_b)
        log_p = mdn_log_prob_diagonal(pi_logits, mu, sigma, z_t)
        nll = -jnp.mean(log_p)
        return kl + beta_nll * nll

    student_opt_state = tx.init(student_w)
    student_vag = jax.value_and_grad(student_loss_fn_batched)

    @jax.jit
    def student_step(w, opt_state, x_b, theta_b, t_w):
        loss, grads = student_vag(w, x_b, theta_b, t_w)
        updates, opt_state = tx.update(grads, opt_state, w)
        w = optax.apply_updates(w, updates)
        return w, opt_state, loss

    @jax.jit
    def val_student_loss(w, x_b, theta_b, t_w):
        loss, _grads = student_vag(w, x_b, theta_b, t_w)
        return loss

    print(f"Training variational student for up to {train_epochs} epochs...")
    key_s = jr.PRNGKey(seed_train + 1)
    best_loss = jnp.inf
    best_w = student_w
    patience_counter = 0
    pbar = tqdm(range(train_epochs), desc="Student train", leave=True)

    for j in pbar:
        key_s, rng = jr.split(key_s)
        randidx = jr.permutation(rng, jnp.arange(theta.shape[0]), independent=True)[:n_train]
        epoch_loss = 0.0
        for bi in range(n_batch):
            sl = slice(bi * train_batch_size, (bi + 1) * train_batch_size)
            idx = randidx[sl]
            x_b = data.reshape(-1, data_shape)[idx]
            th_b = theta.reshape(-1, n_params)[idx]
            student_w, student_opt_state, loss = student_step(
                student_w, student_opt_state, x_b, th_b, teacher_w
            )
            epoch_loss += loss
        epoch_loss /= n_batch

        vidx = jnp.arange(min(theta_test.shape[0], data_test_flat.shape[0]))
        x_val = data_test_flat[vidx]
        th_val = theta_test.reshape(-1, n_params)[vidx]
        val_loss = val_student_loss(student_w, x_val, th_val, teacher_w)

        pbar.set_postfix(train=float(epoch_loss), val=float(val_loss))

        if val_loss < best_loss:
            best_loss = val_loss
            best_w = student_w
            patience_counter = 0
        else:
            patience_counter += 1

        if (patience_counter > patience) and (j + 1 > train_min_epochs):
            print(f"\nEarly stopping at epoch {j}")
            break

    student_w = best_w

    @jax.jit
    def predict_student(w, x):
        return forward_variational_student(student_model, w, x, n_p=n_params)

    t_pred, F_pred, _zm, z_mix_mean, pi, _plog, mu, sigma = predict_student(
        student_w, data_test_flat
    )

    # Shape Fs as (1, n_test, n_p, n_p) to mirror ensemble dimension
    Fs = F_pred[None, ...]
    mle = t_pred[None, ...]

    outputs = create_results_dict(
        theta=theta_test,
        Fs=Fs,
        mle=mle,
        ensemble_weights=jnp.array([1.0]),
        x=data_test_flat,
        z_mix_mean=z_mix_mean,
        mdn_pi=pi,
        mdn_mu=mu,
        mdn_sigma=sigma,
    )

    outname = os.path.join(outdir, "fishnets_variational_outputs")
    np.savez(outname, **dict(outputs))
    print("Saved:", outname + ".npz")

    return student_w, teacher_w, student_model, teacher_model, data_scaler, outputs


if __name__ == "__main__":
    n_samples = 5000
    n_test = 2000
    n_params = 2
    n_d = 50

    MIN_VAR = 0.2
    MAX_VAR = 20.0
    MIN_MU = -5.0
    MAX_MU = 5.0

    @jax.jit
    def simulator(key, th):
        return th[0] + jr.normal(key, shape=(n_d,)) * jnp.sqrt(th[1])

    key = jr.PRNGKey(0)
    key1, key2 = jr.split(key)
    mu_ = jr.uniform(key1, shape=(n_samples,), minval=MIN_MU, maxval=MAX_MU)
    sigma_ = jr.uniform(key2, shape=(n_samples,), minval=MIN_VAR, maxval=MAX_VAR)
    theta_ = jnp.stack([mu_, sigma_], axis=-1)

    key1, key2 = jr.split(key1)
    mu_test = jr.uniform(key1, shape=(n_test,), minval=MIN_MU, maxval=MAX_MU)
    sigma_test = jr.uniform(key2, shape=(n_test,), minval=MIN_VAR, maxval=MAX_VAR)
    theta_test = jnp.stack([mu_test, sigma_test], axis=-1)

    keys = jr.split(key, num=n_samples)
    data = jax.vmap(simulator)(keys, theta_)
    keys = jr.split(key2, num=n_test)
    data_test = jax.vmap(simulator)(keys, theta_test)

    train_fishnets_variational(
        theta_,
        data,
        theta_test,
        data_test,
        data_shape=n_d,
        hids_min=32,
        hids_max=128,
        n_layers=3,
        num_models=20,
        seed_model=201,
        seed_train=999,
        train_batch_size=128,
        train_epochs=2000,
        train_min_epochs=50,
        patience=15,
        lr=1e-4,
        outdir="fishnets-variational-demo",
        num_mdn_components=6,
        pretrain_epochs=400,
        beta_nll=0.5,
    )
