import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from pyoperon.sklearn import SymbolicRegressor
import multiprocessing
import csv
from sklearn.metrics import r2_score
import string
import sys
import sympy
import scipy
import esr.generation.generator

import jax
import jax.numpy as jnp

from tqdm import tqdm as tq

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

