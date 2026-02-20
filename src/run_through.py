import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
import optax

import numpy as np
import math
from typing import Sequence
from tqdm import tqdm
import yaml,os,sys
import argparse

from sklearn.preprocessing import MinMaxScaler
from scipy.stats import linregress

#import tensorflow_probability.substrates.jax as tfp

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fishnets import *
from training_loop_fishnets import train_fishnets, predicted_fishers, predicted_mle

# Function to load yaml configuration file
def load_config(config_name, config_path="./"):
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)

    return config


# Global covariance matrix (set in main() from command line args)
Sigma = None


def mu_x_from_y(y):
  x0 = y[0]
  x1 = y[1] - x0**2

  return jnp.array([x0, x1])

def mu_y_from_x(x):
  return jnp.array([x[0], x[1] + x[0]**2])


#@jax.jit
# def Fisher2(θ, nd):
#     #cov = jnp.eye(dim)
#     cov = Sigma
#     invC = jnp.linalg.inv(cov)

#     dμ_dθ = jax.jacrev(mu_x_from_y)(θ)
#     # do the simple case first
#     return nd * jnp.einsum("ij,ik,kl->jl", dμ_dθ, invC, dμ_dθ)


#@jax.jit
def Fisher(θ, nd):
    #cov = jnp.eye(dim)
    cov = (Sigma)
    invC = jnp.linalg.inv(cov)

    mu_x = θ

    #dμ_dθ = jax.jacrev(mu_x_from_y)(θ)
    # do the simple case first
    return nd * jnp.array([[(1/cov[0,0] + 4*mu_x[0]**2 / cov[1,1]), -2*mu_x[0]/cov[1,1]], 
                        [(-2*mu_x[0] / cov[1,1]), 1/cov[1,1]]])

#@jax.jit


# -------------- CREATE PLOT --------------

def make_fisher_plot(network_fishers, filename):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,4), sharex="col", sharey=True)
    (ax1, ax2) = axs

    dets1 =  jax.vmap(jnp.linalg.det)(jnp.array(fishers)).reshape(xs.shape)

    dets1 = 0.5*np.log(dets1)
    levels = 10 #np.linspace(dets1.min(), dets1.max(), 10) #[1.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    ax1.contourf(xs, ys, dets1, cmap='viridis', levels=levels)
    dets2 = jax.vmap(jnp.linalg.det)(network_fishers).reshape(xs.shape)
    cs2 = ax2.contourf(xs, ys, 0.5*np.log(dets2), cmap='viridis', levels=levels)
    plt.colorbar(cs2)

    ax1.set_title(r'$ \frac{1}{2} \ln \det F_{\rm true}(\theta)$')
    ax1.set_xlabel(r'$\mu_1$')
    ax2.set_ylabel(r'$\mu_2$')
    ax2.set_title(r'$ \frac{1}{2} \ln \det \langle F_{\rm NN}(\theta) \rangle $')
    plt.tight_layout()
    if not filename.endswith(('.png', '.pdf', '.jpg', '.svg')):
        filename = filename + '.png'
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    print(f"Saved grid plot to: {filename}")

    plt.close()




def make_fisher_plot_twopanel(network_fishers, filename):

    fig = plt.figure(figsize=(12, 7))

    # exact fishers
    dets1 =  jax.vmap(jnp.linalg.det)(jnp.array(network_fishers)).reshape(xs.shape)
    dets1 = 0.5*np.log(dets1)
    levels = 10

    ax1 = fig.add_subplot(121)
    im1 = ax1.contourf(xs, ys, dets1, cmap='viridis', levels=levels)

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')

    # network fishers
    ax2 = fig.add_subplot(122)
    dets2 = jax.vmap(jnp.linalg.det)(network_fishers).reshape(xs.shape)
    im2 = ax2.contourf(xs, ys, 0.5*np.log(dets2), cmap='viridis', levels=levels)

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')

    # labels
    ax1.set_title(r'$ \frac{1}{2} \ln \det F_{\rm true}(\theta)$')
    ax1.set_xlabel(r'$\mu_1$')
    ax2.set_ylabel(r'$\mu_2$')
    ax2.set_title(r'$ \frac{1}{2} \ln \det \langle F_{\rm NN}(\theta) \rangle $')
    plt.tight_layout()
    if not filename.endswith(('.png', '.pdf', '.jpg', '.svg')):
        filename = filename + '.png'
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    print(f"Saved grid plot to: {filename}")

    plt.close()




# plot all the other models as well
# look at each model separately
# for i in range(num_models):

#     if i < 14:
#         make_fisher_plot_twopanel(all_fisher_preds[i], "saturation_test_model_%d"%(i+1))





def main():
    # -------------- COMMAND LINE ARGUMENTS --------------
    parser = argparse.ArgumentParser(
        description="Train fishnets on a 2D Gaussian with nonlinear mean parameterization."
    )
    parser.add_argument(
        "--sigma",
        type=float,
        nargs=2,
        default=[1.0, 2.0], # default
        metavar=("S1", "S2"),
        help="Diagonal standard deviations for Sigma = diag([S1^2, S2^2]). Default: [1.0, 2.0], \
            Stress test: [4.0, 1.0]"
    )
    parser.add_argument(
        "--grid-plot",
        action="store_true",
        help="Generate grid plots of Fisher predictions"
    )
    parser.add_argument(
        "--n-d",
        type=int,
        default=50,
        help="Number of data points per dimension. Default: 50"
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=5000,
        help="Number of training simulations. Default: 5000"
    )
    parser.add_argument(
        "--n-nets",
        type=int,
        default=20,
        help="Number of networks to train. Default: 5000"
    )
    parser.add_argument(
        "--shape-norm",
        type=str,
        default="correlation",
        choices=["correlation", "trace"],
        help="Normalization method for shape comparison. Default: correlation (Option 1), Alternative: trace (Option 2)"
    )
    args = parser.parse_args()
    
    # Set global Sigma from command line
    global Sigma
    Sigma = jnp.diag(jnp.array(args.sigma)**2)
    print("Sigma =", Sigma)
    
    do_grid_plot = args.grid_plot


    dim = 2 # dimension of the multivariate normal distribution
    n_d = args.n_d
    n_params = 2 #config["n_params"]
    data_shape = n_d * dim
    input_shape = (data_shape,)
    nsims = args.n_sims

    print("running with n_d=%d"%(n_d))
    print("generating %d train simulations"%(nsims))

    MAX_MU = 3 #config["MAX_MU"]
    MIN_MU =  -3 #config["MIN_MU"]
    SCALE_THETA = False #bool(config["SCALE_THETA"])
    scale_theta_to = (0,1) #config["scale_theta_to"]



    xmin = jnp.array([MIN_MU]*dim)
    xmax = jnp.array([MAX_MU]*dim)


    def simulator(key, θ, nd=n_d):
        # HERE WE SIMULATE IN Y and transform into X
        x = mu_x_from_y(θ)

        # EDIT: JAX COMPUTES USING CHOLESKY FACTOR; CHANGE TO SQRT(SIGMA)
        cov = (Sigma)
        # could include some invertible transformation to the mean
        return jr.multivariate_normal(key, mean=x, cov=cov, shape=(nd,)).reshape(-1)


    # -------------- MAKE SOME DATA --------------
    key = jr.PRNGKey(0)
    key1,key2 = jr.split(key)

    # simulate in (0,1) space --> transformation happens in simulator

    mu_= jr.uniform(key1, shape=(nsims,dim), minval=MIN_MU, maxval=MAX_MU)
    theta_ = mu_

    # make test set
    key1,key2 = jr.split(key1)
    mu_test = jr.uniform(key1, shape=(nsims,dim), minval=MIN_MU, maxval=MAX_MU)

    theta_test = mu_test #jnp.stack([mu_test, sigma_test], axis=-1)

    # create data
    keys = jr.split(key, num=nsims)
    data = jax.vmap(simulator)(keys, theta_)

    keys = jr.split(key2, num=nsims)
    data_test = jax.vmap(simulator)(keys, theta_test)
    theta = theta_.copy()

    # Train ensemble using the parent function
    print("\n" + "="*60)
    print("TRAINING FISHNETS ENSEMBLE")
    print("="*60 + "\n")
    
    mish = lambda x: x * nn.tanh(nn.softplus(x))
    acts = [nn.relu, nn.relu, nn.relu,
            nn.leaky_relu, nn.leaky_relu, nn.leaky_relu,
            optimized_smooth_leaky, optimized_smooth_leaky,
            nn.swish, nn.swish, nn.swish,
            mish, mish,
            nn.gelu, nn.gelu, nn.gelu, nn.gelu, nn.gelu, nn.gelu, nn.gelu, nn.gelu]
    
    ws, ensemble_weights, models, data_scaler = train_fishnets(
        theta=theta,
        data=data,
        theta_test=theta_test,
        data_test=data_test,
        data_shape=data_shape,
        hids_min=100,
        hids_max=300,
        n_layers=3,
        num_models=args.n_nets,
        seed_model=201,
        seed_train=999,
        train_batch_size=200,
        train_epochs=4000,
        train_min_epochs=100,
        patience=200,
        lr=5e-5,
        acts=acts,
        scaler_type='minmax',
        outdir="fishnets-log"
    )
    
    num_models = len(models)


# ---------------------------------------------------------------------------------------------
        # LOOK AT GRID OF FISHERS
    if do_grid_plot:
        num = 10

        xs1 = jnp.linspace(MIN_MU, MAX_MU, num) # MEAN
        ys1 = jnp.linspace(MIN_MU, MAX_MU, num) #jnp.logspace(-1.0, 0.0, num) # VARIANCE


        xs,ys = jnp.meshgrid(xs1, ys1)

        fishers = []

        for _mu,_sigma in zip(xs.ravel(), ys.ravel()):
            fishers.append(Fisher(jnp.array([_mu,_sigma]), nd=n_d))


        fishers_pred = []
        key = jr.PRNGKey(99)


        print("GETTING PREDICTED FISHERS OVER GRID")

        def predicted_fisher_grid(key, model, w, num_sims_avg=200):

            fishers_pred = []

            def _getf(d):
                return model.apply(w, d)[1]


            for _mu,_sigma in tqdm(zip(xs.ravel(), ys.ravel())):
                # generate many data realization at each gridpoint
                key, rng = jr.split(key)
                keys = jr.split(key, num_sims_avg)

                sims = jax.vmap(simulator)(keys, jnp.tile(jnp.array([[_mu], [_sigma]]), num_sims_avg).T) #[:, :, jnp.newaxis]
                #sims /= datascale
                sims = data_scaler.transform(sims.reshape(-1, data_shape)).reshape(sims.shape)


                fpreds = jax.vmap(_getf)(sims)

                fishers_pred.append(jnp.mean(fpreds, axis=0))

            return jnp.array(fishers_pred)

        key = jr.PRNGKey(4) # keep key the same across sims


        #num_for_fisher_grid = 10

        all_fisher_preds = [predicted_fisher_grid(key, models[i], ws[i]) for i in range(num_models)]
        all_fisher_preds = jnp.array(all_fisher_preds)
        avg_fisher_preds = jnp.average(all_fisher_preds, axis=0, weights=ensemble_weights)
        std_fisher_preds = all_fisher_preds.std(0)

        # MAKE THE PLOT FOR THE AVERAGE FISHERS
        make_fisher_plot_twopanel(avg_fisher_preds, filename="saturation_test")

    # ---------------------------------------------------------------------------------------------
    # GET TEST DATASET
    # EXPORT ENSEMBLE ESTIMATES FOR FISHER

    key = jr.PRNGKey(10000)

    key1,key2 = jr.split(key)

    mu_ = jr.uniform(key1, shape=(10000,), minval=MIN_MU, maxval=MAX_MU)
    sigma_ = jr.uniform(key2, shape=(10000,), minval=MIN_MU, maxval=MAX_MU)

    theta_test = jnp.stack([mu_, sigma_], axis=-1)

    keys = jr.split(key, num=10000)
    data_test = jax.vmap(simulator)(keys, theta_test) #[:, :, jnp.newaxis]
    data_test = data_scaler.transform(data_test.reshape(-1, data_shape)).reshape(data_test.shape)
    

    ensemble_F_predictions = jnp.array([predicted_fishers(models[i], ws[i], data_test) for i in range(num_models)])
    ensemble_mle_predictions = jnp.array([predicted_mle(models[i], ws[i], data_test) for i in range(num_models)])
    
    # calculate true fisher at the same theta
    _f = lambda t: Fisher(t, nd=n_d)
    F_true_out = jax.vmap(_f)(theta_test)

    # save everything
    outname = "simple_test_regression_outputs"


    np.savez(outname,
            #data=data_test,
            theta=theta_test,
            F_network_ensemble=ensemble_F_predictions,
            mle_network_ensemble=ensemble_mle_predictions,
            ensemble_weights=ensemble_weights,
            F_true=F_true_out,
            n_d=n_d,
            )


    # make the predictive plots
    plt.figure(figsize=(8, 4))
    
    plt.subplot(121)
    plt.scatter(theta_test[:, 0], ensemble_mle_predictions[0, :, 0], alpha=0.5, label='Model 1', s=2)
    plt.scatter(theta_test[:, 0], ensemble_mle_predictions[1, :, 0], alpha=0.5, label='Model 2', s=2)
    plt.xlabel(r'$\theta_0$ (true)')
    plt.ylabel(r'$\hat{\theta}_0$ (predicted)')
    plt.legend(framealpha=0.0)

    plt.subplot(122)
    plt.scatter(theta_test[:, 1], ensemble_mle_predictions[0, :, 1], alpha=0.5, label='Model 1', s=2)
    plt.scatter(theta_test[:, 1], ensemble_mle_predictions[1, :, 1], alpha=0.5, label='Model 2', s=2)
    plt.xlabel(r'$\theta_1$ (true)')
    plt.ylabel(r'$\hat{\theta}_1$ (predicted)')
    plt.legend(framealpha=0.0)
    
    plt.suptitle("MLE Predictions")
    plt.tight_layout()
    plt.savefig("mle_predictions.png", dpi=400, bbox_inches='tight')
    print("Saved MLE predictions plot to: mle_predictions.png")
    plt.show()



def weighted_std(values, weights, axis=0):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = jnp.average(values, weights=weights, axis=axis)
    # Fast and numerically precise:
    variance = jnp.average((values-average)**2, weights=weights, axis=axis)
    return jnp.sqrt(variance)


def normalize_to_correlation(F):
    """
    Convert Fisher matrix to correlation-like form.
    Normalizes by diagonal elements so diagonal = 1.
    Off-diagonals become relative ratios.
    """
    diag_sqrt = jnp.sqrt(jnp.diagonal(F, axis1=-2, axis2=-1))
    diag_sqrt_inv = 1.0 / diag_sqrt
    return F * diag_sqrt_inv[..., :, None] * diag_sqrt_inv[..., None, :]


def normalize_by_trace(F):
    """
    Normalize Fisher matrix by its trace.
    Preserves all relative ratios while removing overall scale.
    """
    return F / jnp.trace(F, axis1=-2, axis2=-1)[..., None, None]


def predictive_histogram(F_avg, F_std, true_Fishers, 
                        title="Fisher Matrix",
                        filename="fisher_components_hist",
                        matrix_label="F",
                        abs_offdiag=False):
    """
    Create a 3-panel histogram plot comparing predicted vs true Fisher matrix components.
    
    Parameters:
        F_avg: Predicted mean Fisher matrices [n_samples, n_params, n_params]
        F_std: Predicted std Fisher matrices [n_samples, n_params, n_params]
        true_Fishers: True Fisher matrices [n_samples, n_params, n_params]
        title: Title for the plot
        filename: Filename to save the plot
        matrix_label: Label for matrix components (e.g., 'F' or 'L')
        abs_offdiag: Whether to take absolute value of off-diagonal elements
    """
    
    plt.figure(figsize=(7, 3))
    
    # Component [0,0]
    plt.subplot(131)
    mus_pred = F_avg[:, 0, 0]
    mus_sig = F_std[:, 0, 0]
    mus_true = true_Fishers[..., 0, 0]
    trace_to_plot = (mus_pred - mus_true) / mus_sig
    
    plt.hist(np.random.normal(0.0, scale=1.0, size=(10000,)),
             bins=np.linspace(-10, 10, num=50), density=True, label="normal")
    plt.hist(trace_to_plot, alpha=0.45, density=True,
             bins=np.linspace(-10, 10, num=50), label="data")
    plt.xlabel(rf"$(\hat{{{matrix_label}}}_{{{0}{0}}} - {matrix_label}_{{{0}{0}}}) / \delta\hat{{{matrix_label}}}_{{{0}{0}}}$")
    plt.legend(framealpha=0.0)
    
    # Component [0,1]
    plt.subplot(132)
    mus_pred = jnp.abs(F_avg[:, 0, 1]) if abs_offdiag else F_avg[:, 0, 1]
    mus_sig = F_std[:, 0, 1]
    mus_true = jnp.abs(true_Fishers[..., 0, 1]) if abs_offdiag else true_Fishers[..., 0, 1]
    trace_to_plot = (mus_pred - mus_true) / mus_sig
    
    plt.hist(np.random.normal(0.0, scale=1.0, size=(10000,)),
             bins=np.linspace(-10, 10, num=50), density=True, label="normal")
    plt.hist(trace_to_plot, alpha=0.45, density=True,
             bins=np.linspace(-10, 10, num=50), label="data")
    plt.xlabel(rf"$(\hat{{{matrix_label}}}_{{{0}{1}}} - {matrix_label}_{{{0}{1}}}) / \delta\hat{{{matrix_label}}}_{{{0}{1}}}$")
    plt.legend(framealpha=0.0)
    
    # Component [1,1]
    plt.subplot(133)
    mus_pred = F_avg[:, 1, 1]
    mus_sig = F_std[:, 1, 1]
    mus_true = true_Fishers[..., 1, 1]
    trace_to_plot = (mus_pred - mus_true) / mus_sig
    
    plt.hist(np.random.normal(0.0, scale=1.0, size=(10000,)),
             bins=np.linspace(-10, 10, num=50), density=True, label="normal")
    plt.hist(trace_to_plot, alpha=0.45, density=True,
             bins=np.linspace(-10, 10, num=50), label="data")
    plt.xlabel(rf"$(\hat{{{matrix_label}}}_{{{1}{1}}} - {matrix_label}_{{{1}{1}}}) / \delta\hat{{{matrix_label}}}_{{{1}{1}}}$")
    plt.legend(framealpha=0.0)
    
    plt.suptitle(title)
    plt.tight_layout()
    if not filename.endswith(('.png', '.pdf', '.jpg', '.svg')):
        filename = filename + '.png'
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    print(f"Saved histogram plot to: {filename}")
    plt.show()


def predictive_scatter(F_avg, true_Fishers,
                      title="Fisher Matrix predictive",
                      filename="fisher_components",
                      matrix_label="F",
                      abs_offdiag=False,
                      do_regression=False):
    """
    Create a 3-panel scatter plot comparing predicted vs true Fisher matrix components.
    
    Parameters:
        F_avg: Predicted mean Fisher matrices [n_samples, n_params, n_params]
        true_Fishers: True Fisher matrices [n_samples, n_params, n_params]
        title: Title for the plot
        filename: Filename to save the plot
        matrix_label: Label for matrix components (e.g., 'F' or 'L')
        abs_offdiag: Whether to take absolute value of off-diagonal elements
        do_regression: Whether to compute and print linear regression results
    """
    
    plt.figure(figsize=(7, 3))
    
    # Component [0,0]
    plt.subplot(131)
    mus_pred = F_avg[:, 0, 0]
    mus_true = true_Fishers[..., 0, 0]
    
    plt.scatter(mus_true, mus_pred, alpha=0.45, label="data", s=2)
    plt.plot(mus_true, mus_true, ls=':', c='k')
    plt.xlabel(rf"${matrix_label}_{{{0}{0}}}$")
    plt.ylabel(rf"$\hat{{{matrix_label}}}_{{{0}{0}}}$")
    plt.legend(framealpha=0.0)
    
    if do_regression:
        result = linregress(mus_true, mus_pred)
        print(f"{matrix_label}_00 regression: slope={result.slope:.4f}, intercept={result.intercept:.4f}")
    
    # Component [0,1]
    plt.subplot(132)
    mus_pred = jnp.abs(F_avg[:, 0, 1]) if abs_offdiag else F_avg[:, 0, 1]
    mus_true = jnp.abs(true_Fishers[..., 0, 1]) if abs_offdiag else true_Fishers[..., 0, 1]
    
    plt.scatter(mus_true, mus_pred, alpha=0.45, label="data", s=2)
    plt.plot(mus_true, mus_true, ls=':', c='k')
    plt.xlabel(rf"${matrix_label}_{{{0}{1}}}$")
    plt.ylabel(rf"$\hat{{{matrix_label}}}_{{{0}{1}}}$")
    plt.legend(framealpha=0.0)
    
    if do_regression:
        result = linregress(mus_true, mus_pred)
        print(f"{matrix_label}_01 regression: slope={result.slope:.4f}, intercept={result.intercept:.4f}")
    
    # Component [1,1]
    plt.subplot(133)
    mus_pred = F_avg[:, 1, 1]
    mus_true = true_Fishers[..., 1, 1]
    
    plt.scatter(mus_true, mus_pred, alpha=0.45, label="data", s=2)
    plt.plot(mus_true, mus_true, ls=':', c='k')
    plt.xlabel(rf"${matrix_label}_{{{1}{1}}}$")
    plt.ylabel(rf"$\hat{{{matrix_label}}}_{{{1}{1}}}$")
    plt.legend(framealpha=0.0)
    
    if do_regression:
        result = linregress(mus_true, mus_pred)
        print(f"{matrix_label}_11 regression: slope={result.slope:.4f}, intercept={result.intercept:.4f}")
    
    plt.suptitle(title)
    plt.tight_layout()
    if not filename.endswith(('.png', '.pdf', '.jpg', '.svg')):
        filename = filename + '.png'
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    print(f"Saved scatter plot to: {filename}")
    plt.show()

    # Fisher Matrix components
    F_ensemble = ((ensemble_F_predictions)) 
    true_Fishers = ((F_true_out))

    weights = ((ensemble_weights))

    # compute average
    F_avg = (jnp.average(F_ensemble, axis=0, weights=weights))
    F_std = weighted_std(F_ensemble, weights=weights, axis=0)

    # Fisher Matrix plots
    predictive_histogram(F_avg, F_std, true_Fishers,
                        title="Fisher Matrix",
                        filename="fisher_components_hist",
                        matrix_label="F",
                        abs_offdiag=True)
    
    predictive_scatter(F_avg, true_Fishers,
                      title="Fisher Matrix predictive",
                      filename="fisher_components",
                      matrix_label="F",
                      abs_offdiag=True,
                      do_regression=True)




    # Cholesky Factors
    F_ensemble = jnp.linalg.cholesky(ensemble_F_predictions, upper=True)
    true_Fishers = jnp.linalg.cholesky(F_true_out, upper=True)

    weights = ((ensemble_weights))

    F_avg = (jnp.average(F_ensemble, axis=0, weights=weights))
    F_std = weighted_std(F_ensemble, weights=weights, axis=0)

    # Cholesky Factors plots
    predictive_histogram(F_avg, F_std, true_Fishers,
                        title="Cholesky Factors",
                        filename="cholesky_factors_hist",
                        matrix_label="L",
                        abs_offdiag=True)
    
    predictive_scatter(F_avg, true_Fishers,
                      title="Cholesky Factors predictive",
                      filename="cholesky_components",
                      matrix_label="L",
                      abs_offdiag=True,
                      do_regression=False)



    # Shape comparison
    F_ensemble = ensemble_F_predictions
    true_Fishers = F_true_out

    # Apply shape normalization based on command line argument
    if args.shape_norm == "correlation":
        F_ensemble = jax.vmap(jax.vmap(normalize_to_correlation))(F_ensemble)
        true_Fishers = jax.vmap(normalize_to_correlation)(true_Fishers)
        title_suffix = r"(correlation-normalized: $D^{-1/2} F D^{-1/2}$)"
        title_suffix_short = "(correlation-normalized)"
    elif args.shape_norm == "trace":
        F_ensemble = jax.vmap(jax.vmap(normalize_by_trace))(F_ensemble)
        true_Fishers = jax.vmap(normalize_by_trace)(true_Fishers)
        title_suffix = r"(trace-normalized: $F / \mathrm{tr}(F)$)"
        title_suffix_short = "(trace-normalized)"

    weights = ((ensemble_weights))

    F_avg = (jnp.average(F_ensemble, axis=0, weights=weights))
    F_std = weighted_std(F_ensemble, weights=weights, axis=0)

    # Shape comparison plots
    predictive_histogram(F_avg, F_std, true_Fishers,
                        title=f"Shape comparison {title_suffix}",
                        filename="shape_components_hist",
                        matrix_label="F",
                        abs_offdiag=False)
    
    predictive_scatter(F_avg, true_Fishers,
                      title=f"Shape predictive {title_suffix_short}",
                      filename="shapes",
                      matrix_label="F",
                      abs_offdiag=False,
                      do_regression=False)





if __name__ == "__main__":
    main()