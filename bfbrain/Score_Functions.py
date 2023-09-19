"""This module contains different methods to extract uncertainty estimates 
from a trained neural network. It also contains two useful analysis 
functions, MC_call_fast and MC_call_full, which execute Monte Carlo 
dropout with the neural networks that BFBrain produces.
"""

import tensorflow as tf
import numpy as np

@tf.function(jit_compile = True)
def MC_call_full(model, lams, n_trials):
    """Perform predictions on a given input using Monte Carlo dropout. 
    Evaluates the output of model on the input repeatedly, with random 
    dropout applied, and returns all results.

    Parameters
    ----------
    model : tf.keras.Model

    lams : tf.constant(tf.float32, tf.float32)
        A 2-D Tensorflow tensor representing sets of quartic potential 
        coefficients.

    n_trials : int
        Specifies the number of forward passes through the neural network 
        to perform when doing Monte Carlo dropout.

    Returns
    -------
    tf.constant(tf.float32, tf.float32)
        A 2-D Tensorflow tensor of scores for each set of quartic 
        potential coefficients in lams. Each entry along the zero axis 
        represents the results of a different forward pass of the 
        same inputs.
    """
    out_array = tf.TensorArray(dtype = tf.float32, size = n_trials)
    for i in tf.range(n_trials):
        out_array = out_array.write(i, tf.reshape(model(lams, training = True), shape=[-1]))
    out_array = out_array.stack()
    return out_array

@tf.function(jit_compile = True)
def MC_call_fast(model, lams, n_trials):
    """perform predictions on a given input using Monte Carlo dropout 
    when only the average output is required. Evaluates the output of 
    model on the input lams n_trials times and takes the average output 
    for each point. Schematically equivalent to 
    tf.reduce_mean(MC_call_full(model, lams, n_trials), axis = 0), 
    but faster.

    Parameters
    ----------
    model : tf.keras.Model

    lams : tf.constant(tf.float32, tf.float32)
        A 2-D Tensorflow tensor representing sets of quartic potential 
        coefficients.

    n_trials : int
        Specifies the number of forward passes through the neural network 
        to perform when doing Monte Carlo dropout.

    Returns
    -------
    tf.constant(tf.float32) 
        A 1-D Tensorflow tensor of scores for each set of quartic 
        potential coefficients in lams.
    """
    result = tf.reshape(model(lams, training = True), shape=[-1])/tf.cast(tf.constant(n_trials), tf.float32)
    for _ in tf.range(tf.constant(1), tf.constant(n_trials)):
        result = result + tf.reshape(model(lams, training = True), shape=[-1])/ tf.cast(tf.constant(n_trials), tf.float32)
    return result


def QBDC(model, lams, n_trials = 100):
    """Score an ensemble of possible additional training points by 
    "query by dropout committee". Averages the result of many evaluations 
    with dropout enabled in the network and gives the highest scores to 
    points which are closest to the decision boundary. This should 
    estimate total predictive uncertainty, that is both aleatoric 
    (from ambiguity of the underlying input) and epistemic (from lack 
    of training data in the vicinity of the input) uncertainties.

    Parameters
    ----------
    model : tf.keras.Model

    lams : tf.constant(tf.float32, tf.float32)
        A 2-D Tensorflow tensor representing sets of quartic potential 
        coefficients.

    n_trials : int, default=100
        Specifies the number of forward passes through the neural 
        network to perform when doing Monte Carlo dropout.

    Returns
    -------
    tf.constant(tf.float32) 
        A 1-D Tensorflow tensor of scores for each set of quartic 
        potential coefficients in lams.
    """
    result = MC_call_fast(model, lams, n_trials)
    return result*(tf.constant(1.)-result)

def Max_Entropy(model, lams, n_trials = 100):
    """Score an ensemble of possible additional training points by 
    Shannon entropy. Averages the result of many evaluations with dropout 
    enabled in the network and gives the highest scores to points which 
    have the largest entropy. This should estimate total predictive 
    uncertainty, that is both aleatoric (from ambiguity of the underlying 
    input) and epistemic (from lack of training data in the vicinity of 
    the input) uncertainties.

    Parameters
    ----------
    model : tf.keras.Model

    lams : tf.constant(tf.float32, tf.float32)
        A 2-D Tensorflow tensor representing sets of quartic potential 
        coefficients.

    n_trials : int, default=100
        Specifies the number of forward passes through the neural network 
        to perform when doing Monte Carlo dropout.

    Returns
    -------
    tf.constant(tf.float32) 
        A 1-D Tensorflow tensor of scores for each set of quartic 
        potential coefficients in lams.
    """
    eps = 1e-7
    result = MC_call_fast(model, lams, n_trials)
    return -result*tf.math.log(result + eps) - (1. - result)*tf.math.log(1.-result + eps)

def BALD(model, lams, n_trials = 1000):
    """Score an ensemble of possible additional training points by 
    mutual information (as is done in Bayesian Active Learning by 
    Disagreement, or BALD). This should estimate solely epistemic 
    (stemming from lack of training data in the vicinity of the input) 
    uncertainty.

    Parameters
    ----------
    model : tf.keras.Model

    lams : tf.constant(tf.float32, tf.float32)
        A 2-D Tensorflow tensor representing sets of quartic potential 
        coefficients.

    n_trials : int, default=1000
        Specifies the number of forward passes through the neural 
        network to perform when doing Monte Carlo dropout.

    Returns
    -------
    tf.constant(tf.float32) 
        A 1-D Tensorflow tensor of scores for each set of quartic 
        potential coefficients in lams.
    """
    eps = 1e-7
    out_array = MC_call_full(model, lams, n_trials)
    mean_out = tf.reduce_mean(out_array, axis = 0)
    return -mean_out*tf.math.log(mean_out+eps) - (1. - mean_out)*tf.math.log(1.-mean_out + eps) + tf.reduce_mean(out_array*tf.math.log(out_array + eps) + (1.-out_array)*tf.math.log(1.-out_array + eps), axis = 0)

def Predictive_Variance(model, lams, n_trials = 1000):
    """Score an ensemble of possible additional training points by 
    variance of the predicted score. This should estimate solely 
    epistemic (stemming from lack of training data in the vicinity 
    of the input) uncertainty.

    Parameters
    ----------
    model : tf.keras.Model

    lams : tf.constant(tf.float32, tf.float32)
        A 2-D Tensorflow tensor representing sets of quartic potential 
        coefficients.

    n_trials : int, default=1000
        Specifies the number of forward passes through the neural 
        network to perform when doing Monte Carlo dropout.

    Returns
    -------
    tf.constant(tf.float32) 
        A 1-D Tensorflow tensor of scores for each set of quartic 
        potential coefficients in lams.
    """
    out_array = MC_call_full(model, lams, n_trials)
    return tf.math.reduce_std(out_array, axis = 0)

def Variation_Ratios(model, lams, n_trials = 1000):
    """Score an ensemble of possible additional training points by 
    variation ratios (the fraction of evaluations which give the opposite 
    classification to the mode). This should be sensitive to total 
    predictive uncertainty, that is both aleatoric (from ambiguity
    of the underlying input) and epistemic (from lack of training data 
    in the vicinity of the input) uncertainties

    Parameters
    ----------
    model : tf.keras.Model

    lams : tf.constant(tf.float32, tf.float32)
        A 2-D Tensorflow tensor representing sets of quartic potential 
        coefficients.

    n_trials : int, default=1000
        Specifies the number of forward passes through the neural 
        network to perform when doing Monte Carlo dropout.

    Returns
    -------
    tf.constant(tf.float32) 
        A 1-D Tensorflow tensor of scores for each set of quartic 
        potential coefficients in lams.
    """
    out_array = MC_call_full(model, lams, n_trials)
    out_sum = tf.math.reduce_sum(tf.where(out_array > 0.5, 1., 0.), axis = 0) / n_trials

    return tf.math.minimum(out_sum, 1. - out_sum)

def Random_AL(model, lams):
    """Score an ensemble of possible additional training points randomly.
    This can act as a control to confirm that active learning strategies
    outperform a randomly generated ensemble of training points.

    Parameters
    ----------
    model : tf.keras.Model

    lams : tf.constant(tf.float32, tf.float32)
        A 2-D Tensorflow tensor representing sets of quartic potential 
        coefficients.
        
    Returns
    -------
    tf.constant(tf.float32) 
        A 1-D Tensorflow tensor of scores for each set of quartic potential 
        coefficients in lams.
    """
    return tf.random.uniform(shape=(tf.shape(lams)[0],))