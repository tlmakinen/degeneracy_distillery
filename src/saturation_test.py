import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
import optax

import numpy as np
import math
from typing import Sequence
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler

#import tensorflow_probability.substrates.jax as tfp

import matplotlib.pyplot as plt
from fishnets import *


# -------------- DEFINE SIMULATOR AND PARAMS --------------
n_d = 100
input_shape = (n_d,)

MAX_VAR = 15.0
MIN_VAR = 0.15

MAX_MU = 5.0
MIN_MU = -5.0

xmin = jnp.array([MIN_MU, MIN_VAR])
xmax = jnp.array([MAX_MU, MAX_VAR])
scale_theta_to = (0.5, 1.5)

SCALE_THETA = True


def minmax(x, 
           xmin=xmin,
           xmax=xmax,
           feature_range=scale_theta_to):
    minval, maxval = feature_range
    xstd = (x - xmin) / (xmax - xmin)
    return xstd * (maxval - minval) + minval

def minmax_inv(x,
               xmin=xmin,
               xmax=xmax,
               feature_range=scale_theta_to):
               
    minval, maxval = feature_range
    x -= minval
    x /= (maxval - minval)
    x *= (xmax - xmin)
    return x + xmin

@jax.jit
def Fisher(θ, n_d=n_d):
    Σ = θ[1]
    return jnp.array([[n_d / Σ, 0.], [0., n_d / (2. * Σ**2.)]])

@jax.jit
def simulator(key, θ):
    return θ[0] + jr.normal(key, shape=input_shape) * jnp.sqrt(θ[1])


# -------------- DEFINE ID SET-BASED MODEL --------------

class SetEmbedding(nn.Module):
  n_hidden: Sequence[int]
  n_hidden_globals: Sequence[int]
  n_inputs: int=1

  def setup(self):

        self.model_score = MLP(self.n_hidden)
        self.model_fisher = MLP(self.n_hidden)
        self.model_globals = MLP(self.n_hidden_globals)

  def __call__(self, x):

        t = self.model_score(x)
        fisher_cholesky = self.model_fisher(x) 
        t = jnp.mean(t, axis=0)
        fisher_cholesky = jnp.mean(fisher_cholesky, axis=0)
        outputs = self.model_globals(jnp.concatenate([t, fisher_cholesky], axis=-1))
        
        return outputs



# -------------- MAKE SOME DATA --------------
key = jr.PRNGKey(0)
key1,key2 = jr.split(key)

mu_ = jr.uniform(key1, shape=(10000,), minval=MIN_MU, maxval=MAX_MU)
sigma_ = jr.uniform(key2, shape=(10000,), minval=MIN_VAR, maxval=MAX_VAR)
theta_ = jnp.stack([mu_, sigma_], axis=-1)

# make test set
key1,key2 = jr.split(key1)

mu_test = jr.uniform(key1, shape=(5000,), minval=MIN_MU, maxval=MAX_MU)
sigma_test = jr.uniform(key2, shape=(5000,), minval=MIN_VAR, maxval=MAX_VAR)
theta_test = jnp.stack([mu_test, sigma_test], axis=-1)

# create data
keys = jr.split(key, num=10000)
data = jax.vmap(simulator)(keys, theta_)[:, :, jnp.newaxis]

keys = jr.split(key2, num=5000)
data_test = jax.vmap(simulator)(keys, theta_test)[:, :, jnp.newaxis]

theta = theta_.copy()

# rescale data for network
datascale = 10.0
data /= datascale
data_test /= datascale

if SCALE_THETA:
  print("scaling theta")
  scaler = MinMaxScaler(feature_range=scale_theta_to)
  theta = scaler.fit_transform(theta)
#theta = minmax(theta)
#theta_test = minmax(theta)

# -------------- INITIALISE MODELS --------------
key = jr.PRNGKey(0)

# initialise several models
num_models = 3

n_hiddens = [
    [128, 128],
    [128,128,128],
    [256,256,256]
]

models = [nn.Sequential([
            SetEmbedding(n_hiddens[i], 
                          [50,50]),
            Fishnet_from_embedding(
                          n_p = 2
                                      
            )]
        )
        for i in range(num_models)]

keys = jr.split(key, num=num_models)
ws = [m.init(keys[i], data[0]) for i,m in enumerate(models)]


batch_size = 100
epochs = 400
key = jr.PRNGKey(999)

def training_loop(key, model, w, data, theta, patience=20,
                    epochs=epochs, min_epochs=100):

    @jax.jit
    def kl_loss(w, x_batched, theta_batched):

        def fn(x, theta):
          mle,F = model.apply(w, x)
          return mle, F
        
        mle, F = jax.vmap(fn)(x_batched, theta_batched)
        return -jnp.mean(-0.5 * jnp.einsum('ij,ij->i', (theta_batched - mle), \
                                                jnp.einsum('ijk,ik->ij', F, (theta_batched - mle))) \
                                                      + 0.5*jnp.log(jnp.linalg.det(F)), axis=0)
    
    steps = epochs*(theta.shape[0]//batch_size) + epochs
    #scheduler = optax.cosine_onecycle_schedule(steps, 
    #                  peak_value=1e-3, pct_start=0.3, 
    #                  div_factor=25.0, 
    #                  final_div_factor=10000.0)

    tx = optax.adam(learning_rate=5e-5) # scheduler
    opt_state = tx.init(w)
    loss_grad_fn = jax.value_and_grad(kl_loss)

    def body_fun(i, inputs):
        w,loss_val, opt_state, _data, _theta = inputs
        x_samples = _data[i]
        y_samples = _theta[i]

        loss, grads = loss_grad_fn(w, x_samples, y_samples)
        updates, opt_state = tx.update(grads, opt_state)
        w = optax.apply_updates(w, updates)

        # keep running average
        loss_val += loss
        
        return w, loss_val, opt_state, _data, _theta

    
    losses = jnp.zeros(epochs)

    loss_val = 0.
    n_train = 10000
    lower = 0
    upper = n_train // batch_size

    counter = 0
    patience_counter = 0
    best_loss = jnp.inf

    pbar = tqdm(range(epochs), leave=True, position=0)

    for j in pbar:
          key,rng = jr.split(key)

          # reset loss value every epoch
          loss_val = 0.0
          
          # shuffle data every epoch
          randidx = jr.permutation(key, jnp.arange(theta.reshape(-1, 2).shape[0]), independent=True)
          _data = data.reshape(-1, n_d, 1)[randidx].reshape(batch_size, -1, n_d, 1)
          _theta = theta.reshape(-1, 2)[randidx].reshape(batch_size, -1, 2)

          inits = (w, loss_val, opt_state, _data, _theta)

          w, loss_val, opt_state, _data, _theta = jax.lax.fori_loop(lower, upper, body_fun, inits)
          loss_val /= _data.shape[0] # average over all batches

          losses = losses.at[j].set(loss_val)
          pbar.set_description('epoch %d loss: %.5f'%(j, loss_val))

          # use last 10 epochs as running average
          #if j+1 > 10:
          #  loss_val = jnp.mean(losses[j-10:])

          counter += 1

          # patience criterion
          if loss_val < best_loss:
              best_loss = loss_val

          else:
              patience_counter += 1

          if (patience_counter - min_epochs > patience) and (j + 1 > min_epochs):
              print("\n patience count exceeded: loss stopped decreasing \n")
              break
          

    return losses[:counter], w


print("STARTING TRAINING LOOP")

all_losses = []
trained_weights = []

keys = jr.split(key, num_models)

for i,w in enumerate(ws):
  print("\n training model %d of %d \n"%(i+1, num_models))
  loss, wtrained = training_loop(keys[i], models[i], w, data, theta)
  all_losses.append(loss)
  ws[i] = wtrained




# LOOK AT GRID OF FISHERS
lo = [-1.0, 0.5]
hi = [1.0, 3.0]

num = 10

xs1 = jnp.linspace(MIN_MU, MAX_MU, num) # MEAN
ys1 = jnp.linspace(MIN_VAR, MAX_VAR, num) #jnp.logspace(-1.0, 0.0, num) # VARIANCE


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

      sims = jax.vmap(simulator)(keys, jnp.tile(jnp.array([[_mu], [_sigma]]), num_sims_avg).T)[:, :, jnp.newaxis]
      sims /= datascale

      fpreds = jax.vmap(_getf)(sims)

      fishers_pred.append(jnp.mean(fpreds, axis=0))

  return jnp.array(fishers_pred)

key = jr.PRNGKey(4) # keep key the same across sims

all_fisher_preds = [predicted_fisher_grid(key, models[i], ws[i]) for i in range(num_models)]
all_fisher_preds = jnp.array(all_fisher_preds)
avg_fisher_preds = all_fisher_preds.mean(0)
std_fisher_preds = all_fisher_preds.std(0)



# -------------- CREATE PLOT --------------
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,4), sharex="col", sharey=True)
(ax1, ax2) = axs

dets1 =  jax.vmap(jnp.linalg.det)(jnp.array(fishers)).reshape(xs.shape)

dets1 = 0.5*np.log(dets1)
levels = np.linspace(dets1.min(), dets1.max(), 10) #[1.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


ax1.contourf(xs, ys, dets1, cmap='viridis', levels=levels)
dets2 = jax.vmap(jnp.linalg.det)(avg_fisher_preds).reshape(xs.shape) 
cs2 = ax2.contourf(xs, ys, 0.5*np.log(dets2), cmap='viridis', levels=levels)
plt.colorbar(cs2)

ax1.set_title(r'$ \frac{1}{2} \ln \det F_{\rm true}(\theta)$')
ax1.set_ylabel('$\Sigma$')
ax1.set_xlabel('$\mu$')
ax2.set_xlabel('$\mu$')
ax2.set_title(r'$ \frac{1}{2} \ln \det \langle F_{\rm NN}(\theta) \rangle $')
plt.tight_layout()
plt.savefig("saturation_test", dpi=400)

plt.close()


# plot all the other models as well
# look at each model separately
for i in range(num_models):

  fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,4), sharex="col", sharey=True)
  (ax1, ax2) = axs

  dets1 =  jax.vmap(jnp.linalg.det)(jnp.array(fishers)).reshape(xs.shape)
  dets1 = 0.5*np.log(dets1)
  levels = np.linspace(dets1.min(), dets1.max(), 10) #[1.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
  ax1.contourf(xs, ys, dets1, cmap='viridis', levels=levels)
  dets2 = jax.vmap(jnp.linalg.det)(all_fisher_preds[i]).reshape(xs.shape) 

  if SCALE_THETA:
    levels=10
  else:
    levels=levels

  cs2 = ax2.contourf(xs, ys, 0.5*np.log(dets2), cmap='viridis', levels=levels)
  plt.colorbar(cs2)

  ax1.set_title(r'$ \frac{1}{2} \ln \det F_{\rm true}(\theta)$')
  ax1.set_ylabel('$\Sigma$')
  ax1.set_xlabel('$\mu$')
  ax2.set_xlabel('$\mu$')
  ax2.set_title(r'$ \frac{1}{2} \ln \det \langle F_{\rm NN}(\theta) \rangle $')
  plt.tight_layout()
  plt.close()



# EXPORT ENSEMBLE ESTIMATES FOR FISHER


key = jr.PRNGKey(10000)

key1,key2 = jr.split(key)

mu_ = jr.uniform(key1, shape=(10000,), minval=MIN_MU, maxval=MAX_MU)
sigma_ = jr.uniform(key2, shape=(10000,), minval=MIN_VAR, maxval=MAX_VAR)

theta_test = jnp.stack([mu_, sigma_], axis=-1)

keys = jr.split(key, num=10000)
data_test = jax.vmap(simulator)(keys, theta_test)[:, :, jnp.newaxis]
data_test /= datascale

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
outname = "toy_problem_regression_outputs"
if SCALE_THETA:
    outname += "_scaled"


np.savez(outname,
         #data=data_test,
         theta=theta_test,
         F_network_ensemble=ensemble_F_predictions,
         F_true=F_true_out
         )

# save models