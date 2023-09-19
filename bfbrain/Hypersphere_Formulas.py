"""This module contains code for some basic manipulations to translate between n-dimensional polar and Cartesian coordinates.
"""
import numpy as np
import jax.numpy as jnp
import jax
import sympy as sym

from sympy import derive_by_array

def convert_from_polar(v):
    """Converts an array of inputs in the range [0,pi) into an array of inputs in a Cartesian form

    Parameters
    ----------
    v : np.array(np.float32, np.float32)
        A 2-D NumPy array of points in polar coordinates

    Returns
    -------
    np.array(np.float32, np.float32)
        A 2-D NumPy array of points in Cartesian coordinates on the surface of the unit hypersphere.
    """
    v_new = np.copy(v)
    v_len = len(v_new)
    sv = np.concatenate((np.ones((v_len, 1)), np.cumprod(np.sin(v_new), axis=1)), axis=1)
    cv = np.cos(np.concatenate((v_new, np.zeros((v_len, 1))), axis=1))
    return sv*cv

def convert_to_polar(v):
    """Converts an array of inputs in Cartesian form into a lower-dimensional angular parameterization.

    Parameters
    ----------
    v : np.array(np.float32, np.float32)
        A 2-D NumPy array of points in Cartesian coordinates

    Returns
    -------
    np.array(np.float32, np.float32)
        A 2-D NumPy array of points in polar coordinates.
    """
    v_cum_norm = np.flip(np.cumsum(np.flip(v**2, axis=1), axis=1), axis=1)
    with np.errstate(divide='ignore'):
        v_out = np.where(v_cum_norm[:,1:-1] > 0., np.arctan(np.sqrt(v_cum_norm[:,1:-1])/(v[:,:-2])), np.where(v[:,:-2] >= 0., 0., np.pi))
        v_out = np.concatenate([v_out, np.expand_dims(np.where(~(v[:,-1]==0.), np.arctan(v[:,-1]/(v[:,-2] + np.sqrt(v_cum_norm[:,-2]))), 0.), axis=1)], axis=1)
    v_out = np.where(v_out > 0., v_out, v_out + np.pi)
    v_out[:,-1] = 2*v_out[:,-1]
    return v_out

def jax_convert_to_polar(v):
    """Converts an array of inputs in Cartesian form into a lower-dimensional angular parameterization.

    Parameters
    ----------
    v : jnp.array(jnp.float32)
        A 1-D Jax NumPy array representing a point in Cartesian coordinates.

    Returns
    -------
    jnp.array(jnp.float32, jnp.float32)
        A 2-D Jax NumPy array of points in polar coordinates.
    """
    v_cum_norm = jnp.flip(jnp.cumsum(jnp.flip(v**2)))
    v_out = jnp.where(v_cum_norm[1:-1] > 0., jnp.arctan(jnp.sqrt(v_cum_norm[1:-1])/(v[:-2])), jnp.where(v[:-2] >= 0., 0., jnp.pi))
    v_out = jnp.concatenate([v_out, jnp.expand_dims(jnp.where(~(v[-1]==0.), jnp.arctan(v[-1]/(v[-2] + jnp.sqrt(v_cum_norm[-2]))), 0.), axis = 0)])
    v_out = jnp.where(v_out > 0., v_out, v_out + jnp.pi)
    v_out = v_out.at[-1].multiply(2.)
    return v_out

def jax_convert_from_polar(v):
    """Converts an array of inputs in the range [0,pi) into an array of inputs in a Cartesian form

    Parameters
    ----------
    v : jnp.array(jnp.float32)
        A 1-D Jax NumPy array represenging a point in polar coordinates.

    Returns
    -------
    jnp.array(jnp.float32, jnp.float32)
        A 2-D Jax NumPy array of points in Cartesian coordinates on the surface of the unit hypersphere.
    """
    sv = jnp.concatenate((jnp.array([1.]), jnp.cumprod(jnp.sin(v))))
    cv = jnp.cos(jnp.concatenate((v, jnp.array([0.]))))
    return sv*cv

def rand_nsphere(n_points, n_dims, rng):
    """Randomly generates points sampled uniformly from the surface of a unit hypersphere of specified dimension.

    Parameters
    ----------
    n_points : int
        The number of points the method should generate.

    n_dims : int
        The dimensionality of the hypersphere that the method should sample on the surface of.

    rng : np.random.Generator

    Returns
    -------
    np.array(np.float32, np.float32)
        A 2-D NumPy array representing sets of points uniformly sampled from the surface of the n_dims-dimensional unit hypersphere.
    """
    rands = (rng.normal(size=(n_points, n_dims)))
    return rands / (np.linalg.norm(rands, axis=1, keepdims=True))


def cumprod_sym(x, j):
    """Computes the cumulative product of some subset of elements of a 1-D SymPy array.

    Parameters
    ----------
    x : sympy.Array
        A 1-D SymPy array of symbols.

    j : int
        The cumulative product will be computed by taking the product of the 0th through jth element of the array.

    Returns
    -------
    sympy.symbol
    """
    out = 1
    for i in range(j + 1):
        out *= x[i]
    return out

def convert_from_polar_sym(v):
    """Converts symbolic polar coordinates into symbolic Cartesian coordinates.

    Parameters
    ----------
    v : sympy.Array
        A 1-D SymPy array of symbols representing a set of polar coordinates.
        
    Returns
    -------
    sympy.Array
    """
    sinv = v.applyfunc(lambda x: sym.sin(x))
    cosv = v.applyfunc(lambda x: sym.cos(x))
    sv = sym.Matrix([1] + sinv.tolist())
    sv = sym.Matrix([cumprod_sym(sv, i) for i in range(v.shape[0]+1)])
    cv = sym.Matrix(cosv.tolist() + [1])
    return (sym.Array([sv[i]*cv[i] for i in range(v.shape[0]+1)]))