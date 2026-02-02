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

from sklearn.preprocessing import MinMaxScaler

#import tensorflow_probability.substrates.jax as tfp

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fishnets import *

# Function to load yaml configuration file
def load_config(config_name, config_path="./"):
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)

    return config



# fixed covariance
#Sigma = jnp.diag(jnp.array([3.0, 2.0])**2)
#Sigma = jnp.diag(jnp.array([1.0, 2.0])**2) <-- default
Sigma = jnp.diag(jnp.array([4.0, 1.0])**2) # <-- stress test



def mu_x_from_y(y):
  x0 = y[0]
  x1 = y[1] - x0**2

  return jnp.array([x0, x1])

def mu_y_from_x(x):
  return jnp.array([x[0], x[1] + x[0]**2])


#@jax.jit
def Fisher2(θ, nd):
    #cov = jnp.eye(dim)
    cov = Sigma
    invC = jnp.linalg.inv(cov)

    dμ_dθ = jax.jacrev(mu_x_from_y)(θ)
    # do the simple case first
    return nd * jnp.einsum("ij,ik,kl->jl", dμ_dθ, invC, dμ_dθ)


#@jax.jit
def Fisher(θ, nd):
    #cov = jnp.eye(dim)
    print("Sigma", Sigma)
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
    ax1.set_xlabel('$\mu_1$')
    ax2.set_ylabel('$\mu_2$')
    ax2.set_title(r'$ \frac{1}{2} \ln \det \langle F_{\rm NN}(\theta) \rangle $')
    plt.tight_layout()
    plt.savefig(filename, dpi=400)

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
    ax1.set_xlabel('$\mu_1$')
    ax2.set_ylabel('$\mu_2$')
    ax2.set_title(r'$ \frac{1}{2} \ln \det \langle F_{\rm NN}(\theta) \rangle $')
    plt.tight_layout()
    plt.savefig(filename, dpi=400)

    plt.close()








def predicted_fishers(model, w, data):
    ensemble_predictions = []
    def _getf(d):
        return model.apply(w, d)[1]
    F_network_out = jax.vmap(_getf)(data)
    return F_network_out

def predicted_mle(model,w,data):
    def _getmle(d):
        return model.apply(w, d)[0]
    mle_out = jax.vmap(_getmle)(data)
    return mle_out

# plot all the other models as well
# look at each model separately
# for i in range(num_models):

#     if i < 14:
#         make_fisher_plot_twopanel(all_fisher_preds[i], "saturation_test_model_%d"%(i+1))





def main():
    # -------------- DEFINE SIMULATOR AND PARAMS --------------
    #config = load_config('test_config.yaml')

    do_grid_plot = bool(int(sys.argv[1]))


    dim = 2 # dimension of the multivariate normal distribution
    n_d = 50
    n_params = 2 #config["n_params"]
    data_shape = n_d * dim
    input_shape = (data_shape,)
    nsims = 5000

    print("running with n_d=%d"%(n_d))
    print("generating %d train simulations"%(nsims))
    print("Sigma = ", Sigma)

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


    # rescale data for network
    data_scaler = MinMaxScaler(feature_range=(0, 1))
    data = data_scaler.fit_transform(data.reshape(-1, data_shape)).reshape(data.shape)
    data_test = data_scaler.transform(data_test.reshape(-1, data_shape)).reshape(data_test.shape)


    print("data_test", data_test.shape)
    print("theta_test", theta_test.shape)


    # ---------------------------------------------------------------------------------------------


    # -------------- INITIALISE MODELS --------------
    key = jr.PRNGKey(201)


    mish = lambda x: x * nn.tanh(nn.softplus(x))


    # acts = [
    #         nn.relu,
    #         nn.relu,
    #         nn.relu,
    #         nn.leaky_relu,
    #         nn.leaky_relu,
    #         nn.leaky_relu,
    #         smooth_leaky,
    #         nn.swish,
    #         nn.swish,
    #         nn.elu,
    #         nn.elu,
    #         nn.elu,
    #         mish,
    #         mish,
    #         shifted_softplus,
    #         shifted_softplus,
    #         nn.softmax,
    #         nn.softmax,
    #         nn.gelu,
    #         nn.gelu,
    #         nn.gelu,
    #         ]

    acts = [nn.relu, 
            nn.relu,
            nn.relu,
            nn.leaky_relu,
            nn.leaky_relu,
            nn.leaky_relu,
            #nn.elu,   # elus fail the saturation test
            #nn.elu,  
            nn.leaky_relu,
            nn.swish,
            nn.swish, # not as good
            nn.swish, # not as good
            mish,
            mish,
            nn.gelu,
            nn.gelu,
            nn.gelu,
            nn.gelu,
            nn.gelu,
            nn.gelu,
            nn.gelu,
            nn.gelu,
            ]

    #acts = acts[:15]

    # initialise several models
    num_models = len(acts)

    # hids = jnp.array([10, 20, 32, 50, 50, 64, 64, 100, 100, 100, 128])
    # hids = jnp.array([10, 20, 50, 100, 128])
    #hids = np.arange(10, 100)
    #hids = np.arange(20, 200)

    hids = np.arange(100, 300)

    all_n_hidden = []

    # EDIT: IMPOSE A BOTTLENECK FOR EMBEDDING NET

    for n in range(num_models):
        key, rng = jr.split(key)
        hidden = int(jr.choice(key, hids, replace=True))
        print(hidden)

        all_n_hidden.append([hidden,hidden,hidden]) # could add in (n_params + 1)

    models = [nn.Sequential([
                MLP(all_n_hidden[i],
                    act=acts[i]),
                Fishnet_from_embedding(
                            n_p = n_params,
                            act=acts[i],
                            hidden=all_n_hidden[i][0]
                )]
            )
            for i in range(num_models)]

    data = jnp.squeeze(data)
    keys = jr.split(key, num=num_models)
    ws = [m.init(keys[i], data[0]) for i,m in enumerate(models)]


    batch_size = 200 # change to like 10 or 20
    patience = 200 ## set to 20 ?
    epochs = 4000
    min_epochs = 100
    key = jr.PRNGKey(999)

    def training_loop(key, model, w, data,
                        theta,
                        data_val,
                        theta_val,
                        patience=patience,
                        epochs=epochs,
                    min_epochs=min_epochs):

        @jax.jit
        def kl_loss(w, x_batched, theta_batched):

            def fn(x, theta):
                mle,F = model.apply(w, x)
                return mle, F

            mle, F = jax.vmap(fn)(x_batched, theta_batched)

            res = (theta_batched - mle) #- xmin # subtract minimum
            #res += 1.0

            return 0.5*jnp.mean(
                        (jnp.einsum('ij,ij->i', res, jnp.einsum('ijk,ik->ij', F, res)) \
                                                        - jnp.log(jnp.linalg.det(F))), 
                        axis=0)

        #scheduler = optax.warmup_cosine_decay_schedule(
        #  init_value=1e-5, peak_value=1e-3,
        #  warmup_steps=200, decay_steps=400, end_value=5e-6
        #)

        #tx = optax.adam(learning_rate=5e-5) # scheduler
        tx = optax.adam(learning_rate=5e-5)
        opt_state = tx.init(w)
        loss_grad_fn = jax.value_and_grad(kl_loss)

        def body_fun(i, inputs):
            w,loss_val, opt_state, _data, _theta = inputs
            x_samples = _data[i]
            y_samples = _theta[i]

            loss, grads = loss_grad_fn(w, x_samples, y_samples)
            updates, opt_state = tx.update(grads, opt_state, w)
            w = optax.apply_updates(w, updates)

            # keep running average
            loss_val += loss

            return w, loss_val, opt_state, _data, _theta


        losses = jnp.zeros(epochs)
        val_losses = jnp.zeros(epochs)

        loss_val = 0.
        n_train = nsims
        lower = 0
        upper = n_train // batch_size

        counter = 0
        patience_counter = 0
        best_loss = jnp.inf
        best_w = w

        pbar = tqdm(range(epochs), leave=True, position=0)

        for j in pbar:
            key,rng = jr.split(key)

            # reset loss value every epoch
            loss_val = 0.0

            data_shape = n_d*dim
            # shuffle data every epoch
            randidx = jr.permutation(key, jnp.arange(theta.reshape(-1, n_params).shape[0]), independent=True)
            _data = data.reshape(-1, data_shape)[randidx].reshape(-1, batch_size, data_shape)
            _theta = theta.reshape(-1, n_params)[randidx].reshape(-1, batch_size, n_params)

            inits = (w, loss_val, opt_state, _data, _theta)

            w, loss_val, opt_state, _data, _theta = jax.lax.fori_loop(lower, upper, body_fun, inits)
            loss_val /= _data.shape[0] # average over all batches

            losses = losses.at[j].set(loss_val)


                # pass over validation data
            val_loss, _ = loss_grad_fn(w, data_val, \
                                        theta_val)

            val_losses = val_losses.at[j].set(val_loss)


            pbar.set_description('epoch %d loss: %.5f, val_loss: %.5f'%(j, loss_val, val_loss))


            # use last 10 epochs as running average
            #if j+1 > 10:
            #  loss_val = jnp.mean(losses[j-10:])

            counter += 1

            # patience criterion
            if val_loss < best_loss:
                best_loss = val_loss
                best_w = w

            else:
                patience_counter += 1

            if (patience_counter - min_epochs > patience) and (j + 1 > min_epochs):
                print("\n patience count exceeded: loss stopped decreasing \n")
                break


        return losses[:counter], val_losses[:counter], best_loss, best_w


    print("STARTING TRAINING LOOP")

    all_losses = []
    all_val_losses = []
    best_val_losses = []
    trained_weights = []

    keys = jr.split(key, num_models)

    for i,w in enumerate(ws):
        print("\n training model %d of %d \n"%(i+1, num_models))
        loss, val_loss, best_val_loss, wtrained = training_loop(keys[i], models[i], w, data, theta,
                                        data_test.squeeze(),
                                        theta_test.squeeze())
        all_losses.append(loss)
        all_val_losses.append(val_loss)
        best_val_losses.append(best_val_loss)
        ws[i] = wtrained

    # EDIT: TAKE THE BEST VALIDATION LOSS BEFORE EARLY STOPPING FOR ENSEMBLE WEIGHTS
    ensemble_weights = jnp.array([1./jnp.exp(best_val_losses[i]) for i in range(num_models)])
    print("ensemble weights", ensemble_weights)


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
                sims = data_scaler.transform(sims.reshape(-1, n_d)).reshape(sims.shape)


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

    _f = lambda t: Fisher2(t, nd=n_d)

    F_true_out = jax.vmap(_f)(theta_test)


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

    plt.subplot(121)
    plt.scatter(theta_test[:, 0], ensemble_mle_predictions[0, :, 0])
    plt.scatter(theta_test[:, 0], ensemble_mle_predictions[1, :, 0])

    plt.subplot(122)
    plt.scatter(theta_test[:, 1], ensemble_mle_predictions[0, :, 1])
    plt.scatter(theta_test[:, 1], ensemble_mle_predictions[1, :, 1])

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


    from scipy.stats import linregress

    F_ensemble = ((ensemble_F_predictions)) 
    true_Fishers = ((F_true_out))

    weights = ((ensemble_weights))

    # compute average
    F_avg = (jnp.average(F_ensemble, axis=0, weights=weights))
    F_std = weighted_std(F_ensemble, weights=weights, axis=0)



    plt.figure(figsize=(7, 3))

    plt.subplot(131)

    mus_pred = F_avg[:, 0,0] #/ F_avg[:, 0,1]
    mus_sig = F_std[:, 0,0] #/ F_std[:, 0,1]

    # trace of true fishers
    mus_true = (((true_Fishers)))[..., 0, 0] #/ true_Fishers[:, 0,1]

    trace_to_plot = (mus_pred - mus_true) / mus_sig

    plt.hist(np.random.normal(0.0, scale=1.0, size=(10000,)),
        bins=np.linspace(-10, 10, num=50), density=True, label="normal")

    plt.hist(trace_to_plot, alpha=0.45, density=True,
        bins=np.linspace(-10, 10, num=50), label="data")
    plt.xlabel(r"$(\hat{F}_{00} - F_{00}) / \delta\hat{F}_{00}$")
    plt.legend(framealpha=0.0)


    plt.subplot(132)

    mus_pred = jnp.abs(F_avg[:, 0,1]) #/ F_avg[:, 0,1]
    mus_sig = F_std[:, 0,1] #/ F_std[:, 0,1]

    # trace of true fishers
    mus_true = jnp.abs(((true_Fishers)))[..., 0, 1] #/ true_Fishers[:, 0,1]

    trace_to_plot = (mus_pred - mus_true) / mus_sig

    plt.hist(np.random.normal(0.0, scale=1.0, size=(10000,)),
        bins=np.linspace(-10, 10, num=50), density=True, label="normal")

    plt.hist(trace_to_plot, alpha=0.45, density=True,
        bins=np.linspace(-10, 10, num=50), label="data")
    plt.xlabel(r"$(\hat{F}_{01} - F_{01}) / \delta\hat{F}_{01}$")
    plt.legend(framealpha=0.0)



    plt.subplot(133)

    mus_pred = F_avg[:, 1,1] #/ F_avg[:, 0,1]
    mus_sig = F_std[:, 1,1] #/ F_std[:, 0,1]

    # trace of true fishers
    mus_true = (((true_Fishers)))[..., 1, 1] #/ true_Fishers[:, 0,1]
    trace_to_plot = (mus_pred - mus_true) / mus_sig

    plt.hist(np.random.normal(0.0, scale=1.0, size=(10000,)),
        bins=np.linspace(-10, 10, num=50), density=True, label="normal")

    plt.hist(trace_to_plot, alpha=0.45, density=True,
        bins=np.linspace(-10, 10, num=50), label="data")
    plt.xlabel(r"$(\hat{F}_{11} - F_{11}) / \delta\hat{F}_{11}$")
    plt.legend(framealpha=0.0)

    plt.suptitle("Fisher Matrix")
    plt.tight_layout()
    plt.savefig("fisher_components_hist", dpi=400)

    plt.show()


    # PREDICTIVE PLOT --------------------------------------------------------------------------------------

    # make a predictive plot
    plt.figure(figsize=(7, 3))

    plt.subplot(131)

    mus_pred = F_avg[:, 0,0] #/ F_avg[:, 0,1]
    mus_sig = F_std[:, 0,0] #/ F_std[:, 0,1]

    # trace of true fishers
    mus_true = (((true_Fishers)))[..., 0, 0] #/ true_Fishers[:, 0,1]

    plt.scatter(mus_true, mus_pred, alpha=0.45, label="data", s=2)
    plt.plot(mus_true, mus_true, ls=':', c='k')

    plt.xlabel(r"$F_{00}$")
    plt.ylabel(r"$\hat{F}_{00}$")
    plt.legend(framealpha=0.0)

    result = linregress(mus_true, mus_pred)
    print("F_00 regression", "slope", result.intercept, "intercept", result.slope)


    plt.subplot(132)

    mus_pred = jnp.abs(F_avg[:, 0,1]) #/ F_avg[:, 0,1]
    mus_sig = F_std[:, 0,1] #/ F_std[:, 0,1]

    # trace of true fishers
    mus_true = jnp.abs(((true_Fishers)))[..., 0, 1] #/ true_Fishers[:, 0,1]

    plt.scatter(mus_true, mus_pred, alpha=0.45, label="data", s=2)
    plt.plot(mus_true, mus_true, ls=':', c='k')

    plt.xlabel(r"$F_{01}$")
    plt.ylabel(r"$\hat{F}_{01}$")
    plt.legend(framealpha=0.0)

    result = linregress(mus_true, mus_pred)
    print("F_01 regression", "slope", result.intercept, "intercept", result.slope)

    plt.subplot(133)

    mus_pred = F_avg[:, 1,1] #/ F_avg[:, 0,1]
    mus_sig = F_std[:, 1,1] #/ F_std[:, 0,1]

    # trace of true fishers
    mus_true = (((true_Fishers)))[..., 1, 1] #/ true_Fishers[:, 0,1]


    plt.scatter(mus_true, mus_pred, alpha=0.45, label="data", s=2)
    plt.plot(mus_true, mus_true, ls=':', c='k')
    plt.xlabel(r"$F_{11}$")
    plt.ylabel(r"$\hat{F}_{11}$")
    plt.legend(framealpha=0.0)

    #result = linregress(mus_true, mus_pred)
    #print("F_11 regression", "slope", result.intercept, "intercept", result.slope)


    plt.suptitle("Fisher Matrix predictive")
    plt.tight_layout()

    plt.savefig("fisher_components", dpi=400)
    plt.show()




    F_ensemble = jnp.linalg.cholesky(ensemble_F_predictions, upper=True)
    true_Fishers = jnp.linalg.cholesky(F_true_out, upper=True)


    #F_ensemble /=  jnp.abs(jnp.triu(F_ensemble, k=-1) + 1.0).reshape(num_models, -1, n_params*n_params).sum(-1)[..., jnp.newaxis, jnp.newaxis]
    #true_Fishers /= jnp.abs(jnp.triu(true_Fishers, k=-1) + 1.0).reshape(-1, n_params*n_params).sum(-1)[..., jnp.newaxis, jnp.newaxis]

    weights = ((ensemble_weights))

    F_avg = (jnp.average(F_ensemble, axis=0, weights=weights))
    F_std = weighted_std(F_ensemble, weights=weights, axis=0)


    plt.figure(figsize=(7, 3))

    plt.subplot(131)

    mus_pred = F_avg[:, 0,0] #/ F_avg[:, 0,1]
    mus_sig = F_std[:, 0,0] #/ F_std[:, 0,1]

    # trace of true fishers
    mus_true = (((true_Fishers)))[..., 0, 0] #/ true_Fishers[:, 0,1]

    trace_to_plot = (mus_pred - mus_true) / mus_sig

    plt.hist(np.random.normal(0.0, scale=1.0, size=(10000,)),
        bins=np.linspace(-10, 10, num=50), density=True, label="normal")

    plt.hist(trace_to_plot, alpha=0.45, density=True,
        bins=np.linspace(-10, 10, num=50), label="data")
    plt.xlabel(r"$(\hat{L}_{00} - L_{00}) / \delta\hat{L}_{00}$")
    plt.legend(framealpha=0.0)


    plt.subplot(132)

    mus_pred = jnp.abs(F_avg[:, 0,1]) #/ F_avg[:, 0,1]
    mus_sig = F_std[:, 0,1] #/ F_std[:, 0,1]

    # trace of true fishers
    mus_true = jnp.abs(((true_Fishers)))[..., 0, 1] #/ true_Fishers[:, 0,1]

    trace_to_plot = (mus_pred - mus_true) / mus_sig

    plt.hist(np.random.normal(0.0, scale=1.0, size=(10000,)),
        bins=np.linspace(-10, 10, num=50), density=True, label="normal")

    plt.hist(trace_to_plot, alpha=0.45, density=True,
        bins=np.linspace(-10, 10, num=50), label="data")
    plt.xlabel(r"$(\hat{L}_{01} - L_{01}) / \delta\hat{L}_{01}$")
    plt.legend(framealpha=0.0)



    plt.subplot(133)

    mus_pred = F_avg[:, 1,1] #/ F_avg[:, 0,1]
    mus_sig = F_std[:, 1,1] #/ F_std[:, 0,1]

    # trace of true fishers
    mus_true = (((true_Fishers)))[..., 1, 1] #/ true_Fishers[:, 0,1]
    trace_to_plot = (mus_pred - mus_true) / mus_sig

    plt.hist(np.random.normal(0.0, scale=1.0, size=(10000,)),
        bins=np.linspace(-10, 10, num=50), density=True, label="normal")

    plt.hist(trace_to_plot, alpha=0.45, density=True,
        bins=np.linspace(-10, 10, num=50), label="data")
    plt.xlabel(r"$(\hat{F}_{11} - F_{11}) / \delta\hat{F}_{11}$")
    plt.legend(framealpha=0.0)


    plt.suptitle("Cholesky Factors")
    plt.tight_layout()
    plt.savefig("cholesky_factors_hist", dpi=400)

    plt.show()



    # PREDICTIVE PLOT --------------------------------------------------------------------------------------

    # make a predictive plot
    plt.figure(figsize=(7, 3))

    plt.subplot(131)

    mus_pred = F_avg[:, 0,0] #/ F_avg[:, 0,1]
    mus_sig = F_std[:, 0,0] #/ F_std[:, 0,1]

    # trace of true fishers
    mus_true = (((true_Fishers)))[..., 0, 0] #/ true_Fishers[:, 0,1]

    plt.scatter(mus_true, mus_pred, alpha=0.45, label="data", s=2)
    plt.plot(mus_true, mus_true, ls=':', c='k')

    plt.xlabel(r"$F_{00}$")
    plt.ylabel(r"$\hat{F}_{00}$")
    plt.legend(framealpha=0.0)


    plt.subplot(132)

    mus_pred = jnp.abs(F_avg[:, 0,1]) #/ F_avg[:, 0,1]
    mus_sig = F_std[:, 0,1] #/ F_std[:, 0,1]

    # trace of true fishers
    mus_true = jnp.abs(((true_Fishers)))[..., 0, 1] #/ true_Fishers[:, 0,1]

    plt.scatter(mus_true, mus_pred, alpha=0.45, label="data", s=2)
    plt.plot(mus_true, mus_true, ls=':', c='k')

    plt.xlabel(r"$F_{01}$")
    plt.ylabel(r"$\hat{F}_{01}$")
    plt.legend(framealpha=0.0)



    plt.subplot(133)

    mus_pred = F_avg[:, 1,1] #/ F_avg[:, 0,1]
    mus_sig = F_std[:, 1,1] #/ F_std[:, 0,1]

    # trace of true fishers
    mus_true = (((true_Fishers)))[..., 1, 1] #/ true_Fishers[:, 0,1]


    plt.scatter(mus_true, mus_pred, alpha=0.45, label="data", s=2)
    plt.plot(mus_true, mus_true, ls=':', c='k')
    plt.xlabel(r"$F_{11}$")
    plt.ylabel(r"$\hat{F}_{11}$")
    plt.legend(framealpha=0.0)

    plt.suptitle("Cholesky Factors predictive")
    plt.tight_layout()
    plt.savefig("cholesky_components", dpi=400)
    plt.show()



    F_ensemble = ensemble_F_predictions
    true_Fishers = F_true_out


    F_ensemble /=  jnp.abs(jnp.triu(F_ensemble, k=-1) + 1.0).reshape(num_models, -1, n_params*n_params).sum(-1)[..., jnp.newaxis, jnp.newaxis]
    true_Fishers /= jnp.abs(jnp.triu(true_Fishers, k=-1) + 1.0).reshape(-1, n_params*n_params).sum(-1)[..., jnp.newaxis, jnp.newaxis]

    weights = ((ensemble_weights))

    F_avg = (jnp.average(F_ensemble, axis=0, weights=weights))
    F_std = weighted_std(F_ensemble, weights=weights, axis=0)


    plt.figure(figsize=(7, 3))

    plt.subplot(131)

    mus_pred = F_avg[:, 0,0] #/ F_avg[:, 0,1]
    mus_sig = F_std[:, 0,0] #/ F_std[:, 0,1]

    # trace of true fishers
    mus_true = (((true_Fishers)))[..., 0, 0] #/ true_Fishers[:, 0,1]

    trace_to_plot = (mus_pred - mus_true) / mus_sig

    plt.hist(np.random.normal(0.0, scale=1.0, size=(10000,)),
        bins=np.linspace(-10, 10, num=50), density=True, label="normal")

    plt.hist(trace_to_plot, alpha=0.45, density=True,
        bins=np.linspace(-10, 10, num=50), label="data")
    plt.xlabel(r"$(\hat{F}_{00} - F_{00}) / \delta\hat{F}_{00}$")
    plt.legend(framealpha=0.0)


    plt.subplot(132)

    mus_pred = F_avg[:, 0,1] #/ F_avg[:, 0,1]
    mus_sig = F_std[:, 0,1] #/ F_std[:, 0,1]

    # trace of true fishers
    mus_true = (((true_Fishers)))[..., 0, 1] #/ true_Fishers[:, 0,1]

    trace_to_plot = (mus_pred - mus_true) / mus_sig

    plt.hist(np.random.normal(0.0, scale=1.0, size=(10000,)),
        bins=np.linspace(-10, 10, num=50), density=True, label="normal")
    plt.hist(trace_to_plot, alpha=0.45, density=True,
        bins=np.linspace(-10, 10, num=50), label="data")
    plt.xlabel(r"$(\hat{F}_{01} - F_{01}) / \delta\hat{F}_{01}$")
    plt.legend(framealpha=0.0)



    plt.subplot(133)

    mus_pred = F_avg[:, 1,1] #/ F_avg[:, 0,1]
    mus_sig = F_std[:, 1,1] #/ F_std[:, 0,1]

    # trace of true fishers
    mus_true = (((true_Fishers)))[..., 1, 1] #/ true_Fishers[:, 0,1]
    trace_to_plot = (mus_pred - mus_true) / mus_sig

    plt.hist(np.random.normal(0.0, scale=1.0, size=(10000,)),
        bins=np.linspace(-10, 10, num=50), density=True, label="normal")

    plt.hist(trace_to_plot, alpha=0.45, density=True,
        bins=np.linspace(-10, 10, num=50), label="data")
    plt.xlabel(r"$(\hat{F}_{11} - F_{11}) / \delta\hat{F}_{11}$")
    plt.legend(framealpha=0.0)
    plt.suptitle(r"relative shape $\frac{F_{ij}}{\sum_{i,j < i}|F_{ij}|}$")
    plt.tight_layout()
    plt.savefig("shape_components_hist", dpi=400)

    plt.show()




    # PREDICTIVE PLOT --------------------------------------------------------------------------------------

    # make a predictive plot
    plt.figure(figsize=(7, 3))

    plt.subplot(131)

    mus_pred = F_avg[:, 0,0] #/ F_avg[:, 0,1]
    mus_sig = F_std[:, 0,0] #/ F_std[:, 0,1]
    # trace of true fishers
    mus_true = (((true_Fishers)))[..., 0, 0] #/ true_Fishers[:, 0,1]

    plt.scatter(mus_true, mus_pred, alpha=0.45, label="data", s=2)
    plt.plot(mus_true, mus_true, ls=':', c='k')
    plt.xlabel(r"$F_{00}$")
    plt.ylabel(r"$\hat{F}_{00}$")
    plt.legend(framealpha=0.0)



    plt.subplot(132)
    mus_pred = jnp.abs(F_avg[:, 0,1]) #/ F_avg[:, 0,1]
    mus_sig = F_std[:, 0,1] #/ F_std[:, 0,1]
    # trace of true fishers
    mus_true = jnp.abs(((true_Fishers)))[..., 0, 1] #/ true_Fishers[:, 0,1]

    plt.scatter(mus_true, mus_pred, alpha=0.45, label="data", s=2)
    plt.plot(mus_true, mus_true, ls=':', c='k')
    plt.xlabel(r"$F_{01}$")
    plt.ylabel(r"$\hat{F}_{01}$")
    plt.legend(framealpha=0.0)



    plt.subplot(133)
    mus_pred = F_avg[:, 1,1] #/ F_avg[:, 0,1]
    mus_sig = F_std[:, 1,1] #/ F_std[:, 0,1]

    # trace of true fishers
    mus_true = (((true_Fishers)))[..., 1, 1] #/ true_Fishers[:, 0,1]

    plt.scatter(mus_true, mus_pred, alpha=0.45, label="data", s=2)
    plt.plot(mus_true, mus_true, ls=':', c='k')
    plt.xlabel(r"$F_{11}$")
    plt.ylabel(r"$\hat{F}_{11}$")
    plt.legend(framealpha=0.0)

    plt.suptitle("Relative shape predictive")
    plt.tight_layout()
    plt.savefig("shapes", dpi=400)
    plt.show()





if __name__ == "__main__":
    main()