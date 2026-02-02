import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
import optax

import numpy as np
import math
from typing import Sequence
from tqdm import tqdm
import yaml,os

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

# -------------- DEFINE SIMULATOR AND PARAMS --------------
config = load_config('test_config.yaml')

dim = 2 # dimension of the multivariate normal distribution
n_d = 10 * dim
n_params = config["n_params"]
input_shape = (n_d // dim,)

MAX_VAR = config["MAX_VAR"]
MIN_VAR = config["MIN_VAR"]

MAX_MU = 0 #config["MAX_MU"]
MIN_MU = 1 #config["MIN_MU"]
SCALE_THETA = False #bool(config["SCALE_THETA"])
scale_theta_to = config["scale_theta_to"]

xmin = jnp.array([MIN_MU]*dim)
xmax = jnp.array([MAX_MU]*dim)


def minmax(x, 
           xmin=xmin,
           xmax=xmax,
           feature_range=(0,1)):
    minval, maxval = feature_range
    xstd = (x - xmin) / (xmax - xmin)
    return xstd * (maxval - minval) + minval

def minmax_inv(x,
               xmin=xmin,
               xmax=xmax,
               feature_range=(0,1)):
               
    minval, maxval = feature_range
    x -= minval
    x /= (maxval - minval)
    x *= (xmax - xmin)
    return x + xmin


def transform_mean(θ):
    # inverse-transform for simulator
    θ = minmax_inv(θ)
    return jax.nn.sigmoid(θ) #jnp.tanh(θ)

@jax.jit
def Fisher(θ, n_d=n_d / dim):
    cov = jnp.eye(dim)
    invC = jnp.linalg.inv(cov)
    
    dμ_dθ = jax.jacrev(transform_mean)(θ)
    # do the simple case first
    return n_d * jnp.einsum("ij,ik,kl->jl", dμ_dθ, invC, dμ_dθ)

@jax.jit
def simulator(key, θ):
    cov = jnp.eye(dim)
    # could include some invertible transformation to the mean
    mean = transform_mean(θ)
    return jr.multivariate_normal(key, mean=mean, cov=cov, shape=input_shape).reshape(-1)





# -------------- MAKE SOME DATA --------------
key = jr.PRNGKey(0)
key1,key2 = jr.split(key)

# simulate in (0,1) space --> transformation happens in simulator

mu_= jr.uniform(key1, shape=(10000,dim), minval=0, maxval=1)
theta_ = mu_

# make test set
key1,key2 = jr.split(key1)
mu_test = jr.uniform(key1, shape=(5000,dim), minval=0, maxval=1)

theta_test = mu_test #jnp.stack([mu_test, sigma_test], axis=-1)

# create data
keys = jr.split(key, num=10000)
data = jax.vmap(simulator)(keys, theta_)



keys = jr.split(key2, num=5000)
data_test = jax.vmap(simulator)(keys, theta_test)
theta = theta_.copy()

# rescale data for network
data_scaler = MinMaxScaler(feature_range=(0, 1))
#datascale = 10.0
data = data_scaler.fit_transform(data.reshape(-1, n_d)).reshape(data.shape)
data_test = data_scaler.transform(data_test.reshape(-1, n_d)).reshape(data_test.shape)
#data /= datascale
#data_test /= datascale

print("data_test", data_test.shape)
print("theta_test", theta_test.shape)



# if SCALE_THETA:
#   print("scaling theta")
#   scaler = MinMaxScaler(feature_range=(0,1))
#   theta = scaler.fit_transform(theta)
#   theta_test = scaler.transform(theta_test)
#theta = minmax(theta)
#theta_test = minmax(theta)

# -------------- INITIALISE MODELS --------------
key = jr.PRNGKey(201)


mish = lambda x: x * nn.tanh(nn.softplus(x))

acts = [
        # nn.relu, 
        # nn.relu,
        # nn.relu,
        # nn.leaky_relu,
        # nn.leaky_relu,
        # nn.leaky_relu,
        # #nn.elu,   # elus fail the saturation test
        # #nn.elu,  
        # nn.leaky_relu,
        # nn.swish,
        # nn.swish, # not as good
        # nn.swish, # not as good
        # mish,
        # mish,
        # nn.gelu,
        # nn.gelu,
        # nn.gelu,
        # nn.gelu,
        # nn.gelu,
        # nn.gelu,
        # nn.gelu,
        nn.gelu,
        ]

# initialise several models
num_models = len(acts)

n_hiddens = [[256,256,256]]*num_models

models = [nn.Sequential([
            MLP(n_hiddens[i], 
                act=acts[i]),
            Fishnet_from_embedding(
                          n_p = n_params,
                          act=acts[i],
                          hidden=256
            )]
        )
        for i in range(num_models)]

data = jnp.squeeze(data)
keys = jr.split(key, num=num_models)
ws = [m.init(keys[i], data[0]) for i,m in enumerate(models)]


batch_size = 100
patience = 200
epochs = 4000
key = jr.PRNGKey(999)

def training_loop(key, model, w, data, 
                    theta, 
                    data_val,
                    theta_val,
                    patience=patience,
                    epochs=epochs, min_epochs=400):

    @jax.jit
    def kl_loss(w, x_batched, theta_batched):

        def fn(x, theta):
          mle,F = model.apply(w, x)
          return mle, F
        
        mle, F = jax.vmap(fn)(x_batched, theta_batched)
        return -jnp.mean(-0.5 * jnp.einsum('ij,ij->i', (theta_batched - mle), \
                                                jnp.einsum('ijk,ik->ij', F, (theta_batched - mle))) \
                                                      + 0.5*jnp.log(jnp.linalg.det(F)), axis=0)
    
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
    n_train = 10000
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
          
          # shuffle data every epoch
          randidx = jr.permutation(key, jnp.arange(theta.reshape(-1, 2).shape[0]), independent=True)
          _data = data.reshape(-1, n_d, 1)[randidx].reshape(batch_size, -1, n_d)
          _theta = theta.reshape(-1, 2)[randidx].reshape(batch_size, -1, n_params)

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



# LOOK AT GRID OF FISHERS

num = 10

xs1 = jnp.linspace(0, 1, num) # MEAN
ys1 = jnp.linspace(0, 1, num) #jnp.logspace(-1.0, 0.0, num) # VARIANCE


xs,ys = jnp.meshgrid(xs1, ys1)

fishers = []

for _mu,_sigma in zip(xs.ravel(), ys.ravel()):
  fishers.append(Fisher(jnp.array([_mu,_sigma]), n_d=n_d))


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

all_fisher_preds = [predicted_fisher_grid(key, models[i], ws[i]) for i in range(num_models)]
all_fisher_preds = jnp.array(all_fisher_preds)
avg_fisher_preds = jnp.average(all_fisher_preds, axis=0, weights=ensemble_weights)
std_fisher_preds = all_fisher_preds.std(0)



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
   
  fig = plt.figure(figsize=(16, 12))

  # exact fishers
  dets1 =  jax.vmap(jnp.linalg.det)(jnp.array(fishers)).reshape(xs.shape)
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



# MAKE THE PLOT FOR THE AVERAGE FISHERS
make_fisher_plot_twopanel(avg_fisher_preds, filename="saturation_test")




# plot all the other models as well
# look at each model separately
for i in range(num_models):

  make_fisher_plot_twopanel(all_fisher_preds[i], "saturation_test_model_%d"%(i+1))
  # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,4), sharex="col", sharey=True)
  # (ax1, ax2) = axs

  # dets1 =  jax.vmap(jnp.linalg.det)(jnp.array(fishers)).reshape(xs.shape)
  # dets1 = 0.5*np.log(dets1)
  # levels = np.linspace(dets1.min(), dets1.max(), 10) #[1.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
  # ax1.contourf(xs, ys, dets1, cmap='viridis', levels=levels)
  # dets2 = jax.vmap(jnp.linalg.det)(all_fisher_preds[i]).reshape(xs.shape) 

  # if SCALE_THETA:
  #   levels=10
  # else:
  #   levels=levels

  # cs2 = ax2.contourf(xs, ys, 0.5*np.log(dets2), cmap='viridis', levels=levels)
  # plt.colorbar(cs2)

  # ax1.set_title(r'$ \frac{1}{2} \ln \det F_{\rm true}(\theta)$')
  # ax1.set_ylabel('$\Sigma$')
  # ax1.set_xlabel('$\mu$')
  # ax2.set_xlabel('$\mu$')
  # ax2.set_title(r'$ \frac{1}{2} \ln \det \langle F_{\rm NN}(\theta) \rangle $')
  # plt.tight_layout()
  # plt.savefig("saturation_test_model_%d"%(i+1), dpi=400)
  # plt.close()


# EXPORT ENSEMBLE ESTIMATES FOR FISHER

key = jr.PRNGKey(10000)

key1,key2 = jr.split(key)

mu_ = jr.uniform(key1, shape=(10000,), minval=0, maxval=1)
sigma_ = jr.uniform(key2, shape=(10000,), minval=0, maxval=1)

theta_test = jnp.stack([mu_, sigma_], axis=-1)

keys = jr.split(key, num=10000)
data_test = jax.vmap(simulator)(keys, theta_test) #[:, :, jnp.newaxis]
data_test = data_scaler.transform(data_test.reshape(-1, n_d)).reshape(data_test.shape)

#data_test /= datascale



# scale theta
if SCALE_THETA:
  theta_test = scaler.transform(theta_test)

# calculate network fisher over all the data
def predicted_fishers(model, w, data):

  ensemble_predictions = []

  def _getf(d):
      return model.apply(w, d)[1]
  
  F_network_out = jax.vmap(_getf)(data)
  #ensemble_predictions.append(F_network_out)


  return F_network_out


ensemble_F_predictions = jnp.array([predicted_fishers(models[i], ws[i], data_test) for i in range(num_models)])
# calculate true fisher at the same theta
F_true_out = jax.vmap(Fisher)(theta_test)

# save everything
outname = "simple_test_regression_outputs"
if SCALE_THETA:
    outname += "_scaled"


np.savez(outname,
         #data=data_test,
         theta=theta_test,
         F_network_ensemble=ensemble_F_predictions,
         ensemble_weights=ensemble_weights,
         F_true=F_true_out
         )

# save models