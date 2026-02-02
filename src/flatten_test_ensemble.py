import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np


import matplotlib.pyplot as plt

from typing import Sequence, Any
from tqdm import tqdm
import optax

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

import flax.linen as nn
import scipy

from fishnets import *
from flatten_net import *


Array = Any

def minmax(x, 
           xmin,
           xmax,
           feature_range):
    minval, maxval = feature_range
    xstd = (x - xmin) / (xmax - xmin)
    return xstd * (maxval - minval) + minval

def minmax_inv(x,
               xmin,
               xmax,
               feature_range):
               
    minval, maxval = feature_range
    x -= minval
    x /= (maxval - minval)
    x *= (xmax - xmin)
    return x + xmin


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

# -------------- DEFINE SIMULATOR AND PARAMS --------------
n_d = 100
input_shape = (n_d,)

MAX_VAR = 15.0
MIN_VAR = 0.1

MAX_MU = 5.0
MIN_MU = -5.0

SCALE_THETA = False


n_params = 2


key = jr.PRNGKey(0)
n_outputs = int(n_params + int(n_params * (n_params + 1)) // 2)
hidden_size = 200

# get scaling from data
fname = "toy_problem_regression_outputs"
if SCALE_THETA:
  fname += "_scaled"

θs = jnp.array(np.load(fname + ".npz")["theta"])
ensemble_weights = np.load(fname + ".npz")["ensemble_weights"]
Fs = jnp.average(jnp.array(np.load(fname + ".npz")["F_network_ensemble"]), 
              axis=0, weights=ensemble_weights)



max_x = θs.max(0)
min_x = θs.min(0)

# maybe we can have an automatic flag for the flattening network if we don't get
# to detQ < 1.1 or something that asks the user to deepen or widen their flattening net

model = custom_MLP([hidden_size, 
                    hidden_size, 
                    hidden_size,
                    hidden_size,
                    n_params],
                  max_x = max_x, #jnp.array([MAX_MU, MAX_VAR]),
                  min_x = min_x, #jnp.array([MIN_MU, MIN_VAR]))
                  act = smooth_leaky
)

num = 10000


# learn η(θ; w) function where η is a neural network

@jax.jit
def norm(A):
    return jnp.sqrt(jnp.einsum('ij,ij->', A, A))

def get_α(λ=10., ϵ=0.1):
    return - jnp.log(ϵ * (λ - 1.) + ϵ ** 2. / (1 + ϵ)) / ϵ

@jax.jit # 0.01
def l1_reg(x, alpha=0.01):
    return alpha * (jnp.abs(x)).mean()

    
@jax.jit
def info_loss(w, theta_batched, F_batched):
    λ=10. 
    ϵ=0.000001
    α = get_α(λ, ϵ)
    def fn(theta, F):
                        
        mymodel = lambda d: model.apply(w, d)
        J_eta = jax.jacrev(mymodel)(theta)
        Jeta_inv = jnp.linalg.inv(J_eta)
        Q = Jeta_inv @ F @ Jeta_inv.T
        
        loss = norm((Q - jnp.eye(n_params))) + norm((jnp.linalg.inv(Q) - jnp.eye(n_params)))
        
        # add L1 regularization for jacobian
        loss += l1_reg(J_eta.reshape(-1))
        # hack from Tom to improve Frob norm flattening
        r =  λ * loss / (loss + jnp.exp(-1.0*α*loss))
        loss *= r

        return loss, jnp.linalg.det(Q)
    
    loss,Q = jax.vmap(fn)(theta_batched, F_batched)

    return jnp.mean(loss), jnp.mean(Q)



# TRAINING LOOP STUFF


batch_size = 250
epochs = 5500
min_epochs = 1000
patience = 300
w = model.init(key, jnp.ones((n_params,)))

noise = 0 # 1e-7
theta_true = θs.reshape(-1, batch_size, n_params)
F_fishnets = Fs.reshape(-1, batch_size, n_params, n_params)


def training_loop(key, w, 
                  theta_true,
                  F_fishnets,
                  val_size=2, # in batches
                  lr=5e-4,
                  batch_size=batch_size, 
                  patience=patience,
                  epochs=epochs, 
                  min_epochs=min_epochs):
    
    # start optimiser
    tx = optax.adam(learning_rate=lr)
    opt_state = tx.init(w)
    loss_grad_fn = jax.value_and_grad(info_loss, has_aux=True)  

    # speed up the for loop
    def body_fun(i, inputs):
        w,loss_val, opt_state, detFeta, key, theta_true, F_fishnets = inputs
        theta_samples = theta_true[i]
        F_samples = F_fishnets[i]
        
        # add some noise to Fisher
        #F_samples += jr.normal(key, shape=F_samples.shape)*noise
        #theta_samples += jr.normal(key, shape=theta_samples.shape)*noise
            
        (loss_val, detFeta), grads = loss_grad_fn(w, theta_samples, F_samples)
        updates, opt_state = tx.update(grads, opt_state)
        w = optax.apply_updates(w, updates)
        
        return w, loss_val, opt_state, detFeta, key, theta_true, F_fishnets


  
    # train-val split
    mask = jr.uniform(key, shape=(theta_true.shape[0],)) < 0.9
    F_train = F_fishnets[:-val_size]
    F_val = F_fishnets[-val_size:].reshape(-1, n_params, n_params)

    theta_train = theta_true[:-val_size]
    theta_val = theta_true[-val_size:].reshape(-1, n_params)

    losses = jnp.zeros(epochs)
    detFetas = jnp.zeros(epochs)

    val_losses = jnp.zeros(epochs)
    val_detFetas = jnp.zeros(epochs)

    loss = 0.
    detFeta = 0.
    best_detFeta = np.inf

    num_sims = theta_train.reshape(-1, n_params).shape[0]
    lower = 0
    upper = theta_train.shape[0] #num_sims // batch_size

    pbar = tqdm(range(epochs), leave=True, position=0)
    counter = 0

    for j in pbar:
      
      if (counter > patience) and (j + 1 > min_epochs):
            print("\n patience reached. stopping training.")
            losses = losses[:j]
            detFetas = detFetas[:j]
            val_losses = val_losses[:j]
            val_detFetas = val_detFetas[:j]
            break
            
      else:
        
          key,rng = jr.split(key)

          # shuffle data every epoch
          randidx = jr.permutation(key, jnp.arange(num_sims), independent=True)
          F_train = F_train.reshape(-1, n_params, n_params)[randidx].reshape(-1, batch_size, n_params, n_params)
          theta_train = theta_train.reshape(-1, n_params)[randidx].reshape(-1, batch_size, n_params)

          inits = (w, loss, opt_state, detFeta, key, theta_train, F_train)

          w, loss, opt_state, detFeta, key, theta_train, F_train = jax.lax.fori_loop(lower, upper, body_fun, inits)

          # pass over validation data
          (val_loss, val_detFeta), _ = loss_grad_fn(w, theta_val, F_val)

          losses = losses.at[j].set(loss)
          detFetas = detFetas.at[j].set(detFeta)

          val_losses = val_losses.at[j].set(val_loss)
          val_detFetas = val_detFetas.at[j].set(val_detFeta)


          if np.abs(val_detFeta - 1.0) < np.abs(best_detFeta - 1.0):
            best_detFeta = val_detFeta
            counter = 0
          else:
            counter += 1 
        
      pbar.set_description('epoch %d loss: %.4f, detFeta: %.4f, val_detFeta: %.4f'%(j, loss, detFeta, val_detFeta))

    
    return w, (losses, val_losses), (detFetas, val_detFetas)



# RUN LOOP
print("TRAINING FLATTENER NET")
key,rng = jr.split(key)
w, all_loss, all_dets = training_loop(key, w, theta_true, F_fishnets, lr=1e-4)




# continue training for each ensemble member separately
F_ensemble = jnp.load(fname + ".npz")["F_network_ensemble"]


theta_true = θs.reshape(-1, batch_size, n_params)
F_fishnets_ensemble = [f.reshape(-1, batch_size, n_params, n_params) for f in F_ensemble]


key,rng = jr.split(key)
ensemble_ws = []

for k,f in enumerate(F_fishnets_ensemble):
  print("fine-tuning for ensemble member %d"%(k))
  _w, all_loss, all_dets = training_loop(key, w, theta_true, f, lr=3e-5, epochs=200)
  ensemble_ws.append(_w)




def fit_kabsh(θ, η, theta_star):
    theta_star = jnp.array([0.0, 4.0]) # X.mean(0) #
    # find theta closest to central value
    argstar = jnp.argmin(np.sum((θ - theta_star)** 2, -1))
    theta_star = θ[argstar] # X.mean(0)
    print("thetastar", theta_star)
    eta_star = η[argstar] # y.mean(0)
    # find optimal rotation matrix from eta -> theta
    rotmat = kabsch(jnp.array([eta_star]), jnp.array([theta_star]))
    # to apply: np.dot(y - eta_star, rotmat) + eta_star

    return rotmat, eta_star

def get_jacobian(θ, w=w):
    mymodel = lambda d: model.apply(w, d)

    return jax.jacobian(mymodel)(θ)


def get_etas_rotated(θ, rotmat, eta_star, w=w):
    mymodel = lambda d: model.apply(w, d)
    fn = lambda d: jnp.dot(mymodel(d) - eta_star, rotmat) + eta_star
    return fn(θ)

def get_jacobian_rotated(θ, rotmat, eta_star, w=w):
    mymodel = lambda d: model.apply(w, d)
    fn = lambda d: jnp.dot(mymodel(d) - eta_star, rotmat) + eta_star

    return jax.jacobian(fn)(θ)




# then apply model to obtain all ηs for ensemble
η_ensemble = []
Jbar_ensemble = []

for k,_w in enumerate(ensemble_ws):
    
    ηs = model.apply(_w, θs)

    #rotmat, eta_star = fit_kabsh(θs, ηs, theta_star=jnp.array([0.0, 2.0]))
    # getjac = lambda d: get_jacobian_rotated(d, rotmat, eta_star, w=_w)
    
    getjac = lambda d: get_jacobian(d, w=_w)
    

    print("applying model to ensemble member %d"%(k))
    #η_ensemble.append(get_etas_rotated(θs, rotmat, eta_star, w=_w))
    η_ensemble.append(ηs)
    Jbar_ensemble.append(jnp.concatenate(jnp.array([jax.vmap(getjac)(t) for t in θs.reshape(-1, 100, 2)])))
  


# GET ERROR ON NETWORK JACOBIANS FOR MEAN JACOBIAN


ηs = model.apply(w, θs)
Jbar = jnp.concatenate(jnp.array([jax.vmap(get_jacobian)(t) for t in θs.reshape(-1, 100, 2)]))

allFs = jnp.array(np.squeeze(np.load(fname + ".npz")["F_network_ensemble"]))
ensemble_weights = np.load(fname + ".npz")["ensemble_weights"]

print("getting weighted dFs")
Fs = jnp.average(jnp.array(np.load(fname + ".npz")["F_network_ensemble"]), 
              axis=0, weights=ensemble_weights)

dFs = weighted_std(allFs, ensemble_weights, axis=0) #jnp.std(allFs, 0) 


# now set up the solver for δJ:
# for now let's do this all in numpy and vanilla scipy

def get_δJ(F, δF, Jbar):
    """
    Propagate the error on a neural Fisher matrix estimate in θ to the
    Jacobian for a flattened coordinate system η.
    """

    # invert Jbar = <dη/dθ> here to obtain J=Jbar^-1=<dθ/dη>
    J = np.linalg.inv(Jbar)

    # we've obtained J^T F J = I
    # now Q = - J δF  J^T = δJ X^T - X δJ^T ; with X = JF
    # imposing our L2 constraint on our original eq for Q we arrive at
    # Q = XX^TS + SXX^T which is in sylvester form !

    Q = - np.einsum("bik,bkj,blj->bil", J, δF, J) # Q = - J δF J^T
    X = J @ F
    A = np.einsum("bij,bkj->bik", X, X) # A = X X^T
    
    # loop this calculation over batched index of array
    S = jnp.array([scipy.linalg.solve_sylvester(a=A[i], b=A[i], q=Q[i]) for i in range(Q.shape[0])])

    # then we know that δJ = SX
    δJ = S @ X

    # but then finally we want to go back to Jbar = <dη/dθ> coordinates
    # (J + δJ) = (Jbar + δJbar)^-1 where we now know the LHS
    # => (J + δJ)^-1 = Jbar + δJbar
    # => δJbar = (J + δJ)^-1 - Jbar

    return np.linalg.inv(J + δJ) - Jbar, δJ


print("CALCULATING JACOBIAN ERROR")

δJs, δinvJ  = get_δJ(allFs.mean(0), dFs, Jbar)

print("SAVING EVERYTHING")
# save all outputs
outname = "flattened_coords_sr"
if SCALE_THETA:
  outname += "_scaled"
  
np.savez(outname,
         theta=θs,
         eta=ηs,
         Jacobians=Jbar,
         deltaJ=δJs,
         delta_invJ=δinvJ,
         meanF=Fs,
         dFs=dFs,
         F_ensemble=allFs,
         ensemble_weights=ensemble_weights,
         eta_ensemble=jnp.array(η_ensemble),
         Jbar_ensemble=jnp.array(Jbar_ensemble)
)






lo = [MIN_MU, MIN_VAR]
hi = [MAX_MU, MAX_VAR]


num = 30

xs = jnp.linspace(MIN_MU, MAX_MU, num) # MEAN
ys = jnp.linspace(MIN_VAR, MAX_VAR, num) # VARIANCE
xs,ys = jnp.meshgrid(xs, ys)

X = jnp.stack([xs.flatten(), (ys.flatten())], axis=-1)
etas = model.apply(w, X)


# MAKE NICE PICTURE
plt.figure(figsize=(10, 3))

plt.subplot(121)

data = etas[:, 0].reshape(xs.shape)

im = plt.contourf(xs, ys, (data), cmap='viridis', levels=20)
plt.colorbar(im)
#plt.yscale('log')
plt.ylabel('$\Sigma$')
plt.xlabel('$\mu$')
plt.title(r'$ \eta_1$')
plt.legend(framealpha=0., loc='lower left')

plt.subplot(122)
data = etas[:, 1].reshape(xs.shape)

im = plt.contourf(xs, ys, (data), cmap='viridis', levels=20)
plt.colorbar(im)
#plt.yscale('log')
plt.ylabel('$\Sigma$')
plt.xlabel('$\mu$')
plt.title(r'$ \eta_2$')
plt.legend(framealpha=0., loc='lower left')

plt.savefig("coordinate_visualisation.png")

