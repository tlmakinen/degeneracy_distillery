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
import yaml

from fishnets import *
from flatten_test import *


# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


# -------------- DEFINE SIMULATOR AND PARAMS --------------

config = load_config('test_config.yaml')

n_d = config["n_d"]
input_shape = (n_d,)

MAX_VAR = config["MAX_VAR"]
MIN_VAR = config["MIN_VAR"]

MAX_MU = config["MAX_MU"]
MIN_MU = config["MIN_MU"]
SCALE_THETA = bool(config["SCALE_THETA"])
scale_theta_to = config["scale_theta_to"]

xmin = jnp.array([MIN_MU, MIN_VAR])
xmax = jnp.array([MAX_MU, MAX_VAR])


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


