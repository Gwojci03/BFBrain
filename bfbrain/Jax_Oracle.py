"""The code in this module contains BFBrain's default algorithm for 
labelling if a point is bounded from below. It is designed to work with 
the methods in Active_Learning.py. The fundamental strategy of this 
labelling method is to take a quartic part of the potential function with 
fixed quartic coefficients, and attempt many consecutive local 
optimizations with respect to the scalar vev with random initial starting 
points. If it can find a point where the potential is negative, it will 
label the set of quartic coefficients as not bounded from below, while 
if after some user-specified number of local minimization iterations the 
found minima are always positive, the set will be labelled as bounded 
from below.
"""

import jax
from jax import random as jrandom
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial 
from bfbrain.Hypersphere_Formulas import jax_convert_to_polar
import numpy as np


import jaxopt
import os


#This line forces Jax to only grab memory from the GPU as it needs it-- 
#otherwise it will by default reserve 90% of the available GPU memory and leave Tensorflow to live on scraps. 
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

def take_step(x0, key, polar):
    """Return a tuple of phi, key (in that order) after randomly 
    generating phi as a vev point on the unit hypersphere.

    Parameters
    ----------
    x0 : jnp.array(jnp.float32)
        A Jax NumPy array that tells the function what shape the vev input 
        arrays should be.

    key : Jax PRNGGKey
        A key object used to generate random numbers in Jax. Will be 
        consumed over the running of the function and a new key will be 
        returned.

    polar : bool
        A flag denoting whether the labeller should use Cartesian 
        coordinates for the vev (False) or convert them into polar 
        coordinates on the unit hypersphere (True).

    Returns
    -------
    rand_phi : jnp.array(jnp.float32).
        A 1-D Jax Numpy array representing a single point in vev space.

    new_key : Jax PRNGKey
        A key object used to generate random numbers in Jax. Supplied to 
        replace the consumed key.
    """
    # Generate two new keys from key: subkey to be consumed to generate a 
    # random vev configuration and new_key to pass back for future random 
    # number generation.
    new_key, subkey = jrandom.split(key)
    # Generate a random vev configuration on the phi_len-dimensional unit hypersphere.
    rand_phi = (jrandom.normal(key = subkey, shape = jnp.shape(x0)))
    rand_phi = rand_phi / jnp.linalg.norm(rand_phi)
    # If the minimizer is using polar coordinates, convert rand_phi 
    # to polar coordinates. Otherwise simply return rand_phi (and new_key).
    if(polar):
        return jax_convert_to_polar(rand_phi), new_key
    else:
        return rand_phi, new_key

def run_one_step(key, minimizer, stepper, lam, polar):
    """Takes a single local minimization step. This consists of randomly 
    generating a starting point, running the local minimizer from that 
    starting point, and extracting the potential value after minimization.

    Parameters
    ----------
    key : Jax PRNGKey
        A key object used to generate random numbers in Jax. Will be 
        consumed over the running of the function and a new key will be 
        returned.

    minimizer : JaxOpt Projected Gradient optimizer object.

    stepper : callable
        The function take_step, wrapped to require only a PRNGKey object 
        as input.

    lam : jnp.array(jnp.float32)
        A 1-D Jax Numpy array representing a set of quartic coefficients 
        in the scalar potential.

    Returns
    -------
    energy_after_quench : jnp.float32
        The minimum value of the potential found by minimizer after 
        starting from a random starting position generated with key.

    new_key : Jax PRNGKey
        A key object used to generate random numbers in Jax. Supplied to 
        replace the consumed key.
    """
    # Generate an initial position for the minimizer to use from the 
    # function stepper. This will also consume the PRNG key and 
    # necessitate acquiring a new one.
    x_after_step, new_key = stepper(key = key)
    # Run the minimizer from the point x_after_step. The arguments 
    # for the bounds on variables in the minimizer depend on whether 
    # the minimizer is using polar coordinates or not.
    if(polar):
        ub = jnp.pi*jnp.ones(jnp.shape(x_after_step)[0])
        ub = ub.at[-1].multiply(2.)
        minres = minimizer.run(x_after_step, hyperparams_proj = (jnp.zeros(jnp.shape(x_after_step)[0]), ub), lam = lam)
    else:
        minres = minimizer.run(x_after_step, hyperparams_proj = 1., lam = lam)
    # Compute the value of the quartic part of the potential at the 
    # minimum.
    energy_after_quench = minimizer.fun(minres[0], lam)[0]
    return energy_after_quench, new_key


@partial(jax.jit, static_argnums=[3,5])
def jax_basinhopping(lam, rng_key, x0, minimizer, niter, polar):
    """Iterate over run_one_step a large number of times.

    Parameters
    ----------
    lam : jnp.array(jnp.float32)
        A 1-D Jax Numpy array representing a set of quartic coefficients 
        in the scalar potential.
        
    rng_key : Jax PRNGKey
        Used to generate random numbers.

    x0 : jnp.array(jnp.float32)
        A Jax NumPy array that tells the function what shape the vev input 
        arrays should be.

    minimizer : JaxOpt Projected Gradient optimizer object.

    niter : int
        The maximum number of run_one_step iterations to perform in an 
        attempt to find a negative local minimum of the potential.

    polar : bool
        A flag denoting whether the labeller should use Cartesian 
        coordinates for the vev (False) or convert them into polar 
        coordinates on the phi_len-dimensional unit hypersphere (True).

    Returns
    -------
    new_min_energy: jnp.float32
        The smallest local minimum that the optimizer found after its 
        iterations.
    """
    # Define the run_step method from run_one_step, specifying various arguments.
    run_step = partial(run_one_step, minimizer=minimizer, stepper=partial(take_step, x0 = x0, polar = polar), lam=lam, polar = polar)
    # Define a Jax while loop. This loop performs run_one_step repeatedly 
    # and keeps track of the minimum potential value it's found as min_energy. 
    # If niter iterations have been performed or min_energy is
    # ever negative, break out of the loop.
    def body(carry):
        nstep, min_energy, key = carry
        trial_energy, new_key = run_step(key = key)
        new_nstep = nstep + jnp.array(1)
        min_energy = jax.lax.select(trial_energy > min_energy, min_energy, trial_energy)
        return new_nstep, min_energy, new_key
    def cond_fun(carry):
        nstep, min_energy, _ = carry
        break_cond_negative = min_energy < jnp.array(0.)
        return (~break_cond_negative) & (nstep < niter)
    _, new_min_energy, _ = jax.lax.while_loop(cond_fun, body, init_val=(jnp.array(0), jnp.array(1.), rng_key))
    # Return the minimum energy found from the loop.
    return new_min_energy

@partial(jax.jit, static_argnums=[0,6])
@partial(jax.vmap, in_axes=(None, 0, 0, None, None, None, None))
def vectorized_minTest(func, lam, key, x0, niter, tol, polar):
    """A vectorized version of jax_basinhopping.

    Parameters
    ----------
    func : callable
        A Jax Numpy function that returns the quartic potential and its 
        gradient with respect to the vev parameters. This function will 
        be generated by a BFBrain.DataManager object.

    lam : jnp.array(jnp.float32, jnp.float32)
        A 2-D Jax NumPy array representing multiple sets of quartic 
        coefficients for the potential.

    key : Jax PRNGKey

    x0 : jnp.array(jnp.float32)
        A Jax NumPy array that informs the function about the shape of 
        the vev input.

    niter : int
        The maximum number of local minimization iterations the minimizer 
        should perform for each set of quartic coefficients in lam, 
        searching for a negative minimum potential value.

    tol : jnp.float32
        The tolerance for the local minimizer to stop.

    polar : bool
        A flag denoting whether the labeller should use Cartesian 
        coordinates for the vev (False) or convert them into polar 
        coordinates on the phi_len-dimensional unit hypersphere (True).

    Returns
    -------
    jnp.array(jnp.float32)
        An array of minimum energy values found for the potentials 
        specified by the elements of lam.
    """
    # Print statement to alert user to retracing. Retracing is expensive 
    # and should be minimized.
    print('recompiling vectorized_minTest...')
    # Set up a local minimizer based on projected gradient descent, 
    # with slightly different construction depending on whether or not 
    # the solver is using polar coordinates.
    if(polar):
        minimizer = jaxopt.ProjectedGradient(func, jaxopt.projection.projection_box, tol=tol, value_and_grad = True)
    else:
        minimizer = jaxopt.ProjectedGradient(func, jaxopt.projection.projection_l2_sphere, tol=tol, value_and_grad = True)

    # Run jax_basinhopping. The jax.vmap decorator vectorizes this code.
    return jax_basinhopping(lam, key, x0, minimizer = minimizer, niter = niter, polar = polar) > 0


def label_func(func, phi_len, polar, rng, lam, niter = 100, tol = 0.001, cutoff = 150000):
    """The function which interfaces directly with BFBrain.DataManager's 
    methods for handling oracles.

    Parameters
    ----------
    func : callable
        The Jax Numpy function that returns the quartic part of the 
        potential and the gradient.

    phi_len : int
        The number of real parameters necessary to uniquely specify a vev.

    polar : bool
        A flag denoting whether the labeller should use Cartesian 
        coordinates for the vev (False) or convert them into polar 
        coordinates on the phi_len-dimensional unit hypersphere (True).

    rng : np.random.Generator
        The NumPy random number generator which will generate the initial 
        PRNGKey used by this oracle.

    lam : np.array(np.float32,np.float32)
        A 2-D Numpy array of quartic potential coefficients.

    niter : int, default=100
        The number of local minimizations to perform on the potential 
        before declaring it to be bounded-from-below.

    tol : float, default=0.001
        The tolerance for the local minimizer.

    cutoff : int, default=150000
        The maximum size of a batch of coefficient values to pass to the 
        GPU at one time. If lam consists of more sets of coefficient 
        values than this, the method will split it into digestible batches.

    Returns
    -------
    np.array(bool)
        a 1-D NumPy array of labels for each set of quartic coefficients 
        in lam. Labels False for points where the labeller found a 
        negative local minimum and True otherwise.
    """
    # If the size of lam is below cutoff, transfer the entirety of lam to 
    # the GPU at once and label it.
    if(len(lam) <= cutoff):
        return label_func_do_batch(func, phi_len, polar, rng, lam, niter, tol)
    # Otherwise, split lam into smaller chunks that are guaranteed to have 
    # length smaller than cutoff.
    else:
        lam_arr = np.array_split(lam, len(lam)//cutoff + 1)
        out_arr = []
        for in_lam in lam_arr:
            out_arr.append(label_func_do_batch(func, phi_len, polar, rng, in_lam, niter, tol))
        return np.concatenate(out_arr)


def label_func_do_batch(func, phi_len, polar, rng, lam, niter, tol):
    """The method usedy by label_func to transfer the coefficient data to 
    the GPU and perform the jit-compiled analysis with Jax.

    Parameters
    ----------
    func : callable
        The Jax Numpy function that returns the quartic part of the 
        potential and the gradient.

    phi_len : int
        The number of real parameters necessary to uniquely specify a vev.

    polar : bool
        A flag denoting whether the labeller should use Cartesian 
        coordinates for the vev (False) or convert them into polar 
        coordinates on the phi_len-dimensional unit hypersphere (True).

    rng : np.random.Generator
        The NumPy random number generator which will generate the initial 
        PRNGKey used by this oracle.

    lam : np.array(np.float32,np.float32)
        A 2-D Numpy array of quartic potential coefficients.

    niter : int
        The number of local minimizations to perform on the potential 
        before declaring it to be bounded-from-below.

    tol : float
        The tolerance for the local minimizer.

    Returns
    -------
    np.array(bool)
        a 1-D NumPy array of labels for each set of quartic coefficients 
        in lam. Labels False for points where the labeller found a 
        negative local minimum and True otherwise.
    """
    # Create a Jax random number generation key by randomly generating a 64-bit integer.
    key = jrandom.PRNGKey(rng.integers((2**63) - 1))
    # Split the key into a vector of random keys in order to use each with one point's minimization problem.
    keys = jrandom.split(key,len(lam))
    # Copy lam into a Jax Numpy array on the GPU.
    jnp_lam = jnp.array(lam)
    x0 = jnp.zeros(phi_len)
    # Use vectorized_minTest in order to label each state.
    return np.array(vectorized_minTest(func, jnp_lam, keys, x0, jnp.array(niter), jnp.array(tol), polar))

def test_labeller(func, phi_len, polar, rng, lam, niter = 100, tol = 0.001, cutoff = 150000, niter_step = 50, count_success = 5, max_iter = 20, verbose = False):
    """A method to test the accuracy of the oracle. Will perform 
    label_func repeatedly for the same 2-D NumPy array of quartic 
    coefficients, but with niter increased each time, until the same 
    labels are returned for for a specified consecutive number of 
    iterations, or some maximum number of labellings has been completed 
    without finding consistent results.

    Parameters
    ----------
    func : callable
        The Jax Numpy function that returns the quartic part of the 
        potential and the gradient.

    phi_len : int
        The number of real parameters necessary to uniquely specify a vev.

    polar : bool
        A flag denoting whether the labeller should use Cartesian 
        coordinates for the vev (False) or convert them into polar 
        coordinates on the phi_len-dimensional unit hypersphere (True).

    rng : np.random.Generator
        The NumPy random number generator which will generate the initial 
        PRNGKey used by this oracle.

    lam : np.array(np.float32,np.float32)
        A 2-D Numpy array of quartic potential coefficients.

    niter : int, default=100
        The initial number of local minimizations to perform on the 
        potential before declaring it to be bounded-from-below-- this 
        value will be incremented over the running of the method.

    tol : float, default=0.001
        The tolerance for the local minimizer.
    cutoff : int, default=150000
        The maximum size of a batch of coefficient values to pass to the 
        GPU at one time. If lam consists of more sets of coefficient 
        values than this, the method will split it into digestible batches.

    niter_step : int, default=50
        The amount to increment the niter parameter of label_func with 
        each successive attempt at labelling lam.

    count_success : int, default=5
        The number of consecutive labelling attempts that must yield 
        identical labels for the function to declare that increasing 
        niter is no longer affecting the results of label_func.

    max_iter : int, default=20
        The maximum number of labelling attempts that the method will 
        make. If no consistent results are found before that time, the 
        test ends in failure.

    verbose : bool, default=False
        If True, print out statements informing the user of the progress 
        of the method.
        
    Returns
    -------
    int
        The minimum niter parameter such that count_success consecutive 
        attempts to label lam with increasing niter yielded the same 
        label. If max_iter attempts are made without running into 
        count_success consecutive identical label results, -1 is returned.
    """
    # An integer to keep track of the number of consecutive iterations with 
    # identical labels.
    count = 0
    # A NumPy array to keep track of the latest iteration's labels.
    current_res = np.zeros(shape=len(lam), dtype=bool)
    # The index of the earliest iteration to give the same labels as the 
    # current iteration.
    current_ind = 0
    for i in range(max_iter):
        # In the loop, repeatedly call label_func. With each loop 
        # iteration, increment niter by niter_step. 
        if(verbose):
            print('doing round ' + str(i))
        new_res = label_func(func, phi_len, polar, rng, lam, niter + i*niter_step, tol, cutoff)
        if(verbose):
            print('done!')
        # If the labels produced from this iteration differ at all from 
        # those produced by the previous iteration, reset count and 
        # update current_res, and current_ind.
        is_update = np.any(current_res != new_res)
        if(is_update):
            if(verbose):
                print('updating the res array from ' + str(len(current_res[current_res])) + ' positives to ' + str(len(new_res[new_res])) + ' positives...')
            count = 0
            current_res = new_res
            current_ind = i
        # If the labels are the same as the previous iteration, 
        # increment count.
        else:
            count += 1
        # If count_success consecutive iterations have given the same 
        # labels, or it is now impossible for that to occur before 
        # max_iter iterations are completed, break out of the loop.
        if(count >= count_success or count + max_iter - i - 1 < count_success):
            break
    # If the function was successful in finding consistent labels, 
    # return the smallest niter value such that consecutive iterations 
    # yielded the same result. If the function was unsuccessful, return -1.
    if(count < count_success):
        if(verbose):
            print('failed to find consistent results after ' + str(max_iter) + ' iterations. Recommend decreasing tol or starting again with niter = ' + str(niter + niter_step*(max_iter - 1)))
        return -1
    else:
        if(verbose):
            print('Found consistent results for niter >= ' + str(niter + current_ind*niter_step))
        return niter + current_ind*niter_step