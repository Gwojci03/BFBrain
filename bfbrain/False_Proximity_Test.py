"""These are a series of functions used to score models based on how 
distant their false positives and false negatives over some data set are 
from the model's decision boundary. The only function used externally is 
combined_false_score, which discusses how the scoring is done. All other 
functions in this file are only used internally by combined_false_score 
and the functions that it calls.
"""

import tensorflow as tf
import numpy as np
    
def combined_false_score(model, ds, dist = tf.constant(0.05)):
    """A method to evaluate how "wrong" the model's predictions of the 
    validation set actually are, based on how far false positives and 
    false negatives are from the decision boundary. Returns two sets 
    of information for the false positives and the false negatives.
    For false positives (negatives), uses _find_accurate_points to find 
    nearby sets of coefficients that are correctly classified as 
    negative (positive). The angular distance between these new points
    and the corresponding false positive (negative) points is then 
    computed in radians. The function returns the mean, 
    standard deviation, and maximum of these distances for both 
    false positives and negatives, as well as the number of points for 
    which the angular distance exceeds a specified angle in radians.

    Parameters
    ----------
    model : tf.keras.Model

    ds : tf.data.Dataset
        A labelled Tensorflow dataset of sets of quartic potential 
        coefficients.

    dist : tf.float32
        A maximum angular distance in radians between an incorrectly 
        classified point and the classifier's decision boundary that the 
        user deems acceptable. For small values of dist, this corresponds 
        to the maximum difference in a (normalized) quartic potential 
        coefficient between an incorrectly classified point and a 
        correctly classified one.

    Returns
    -------
    tuple of lists of floats.
        Each list contains the mean, standard deviation, and maximum of 
        the angular distance between incorrectly classified points in ds 
        and correctly classified ones generated with 
        _find_accurate_points. The final element of each list gives the 
        number of incorrectly classified points that are greater than 
        dist radians away from a correctly classified one. 
"""
    #Find the false positives and false negatives from the dataset.
    false_positives, false_negatives = get_false_pos_and_neg_tf(model, ds)

    # Create tf.Variable objects representing the false positives and 
    # false negatives, so that we can deform them to create new instances 
    # which cross the decision boundary.
    neighbors_fp = tf.Variable(tf.identity(false_positives))
    neighbors_fn = tf.Variable(tf.identity(false_negatives))
    # Also create two more identical tf.Variable tensors. These will keep 
    # track of the initial states of the false positives and 
    # false negatives and be used as contrapoints when finding the 
    # decision boundary of the neural network.
    init_fp = tf.Variable(tf.identity(false_positives))
    init_fn = tf.Variable(tf.identity(false_negatives))

    #Now generate points near these points which the neural network 
    # correctly classifies.
    _find_accurate_points(model, neighbors_fp, init_fp, false_pos = tf.constant(True), maxiter = tf.constant(10000), init_rot = dist/50.)
    _find_accurate_points(model, neighbors_fn, init_fn, false_pos = tf.constant(False), maxiter = tf.constant(10000), init_rot = dist/50.)

    # Compute the angular distance between the false positives 
    # and the correctly labelled points.
    results_fp = tf.acos(tf.clip_by_value(tf.math.reduce_sum(false_positives*neighbors_fp, axis = 1), -1., 1.))
    results_fn = tf.acos(tf.clip_by_value(tf.math.reduce_sum(false_negatives*neighbors_fn, axis = 1), -1., 1.))
    #Return information about the minimum distances between falsely 
    # classified points and correctly classified bounded-from-below points.
    return [tf.math.reduce_mean(results_fp).numpy(), tf.math.reduce_std(results_fp).numpy(), tf.math.reduce_max(results_fp).numpy(), tf.shape(tf.gather(results_fp, tf.where(tf.math.greater(results_fp, dist))))[0].numpy()], [tf.math.reduce_mean(results_fn).numpy(), tf.math.reduce_std(results_fn).numpy(), tf.math.reduce_max(results_fn).numpy(), tf.shape(tf.gather(results_fn, tf.where(tf.math.greater(results_fn, dist))))[0].numpy()]


def _tf_flatten(t):
    """A method which returns a flattened tensor.
    
    Parameters
    ----------
    t : tf.Tensor

    Returns
    -------
    tf.Tensor
        A tensor with the same content as t, but flattened to be 
        one-dimensional.
    """
    return tf.reshape(t, shape = [-1])

@tf.function(jit_compile = True)
def _model_grad(model, lams_var):
    """A convenience function to compute the gradient of the model 
    prediction as a function of the inputs. Used to help validate 
    the accuracy of the model.

    Parameters
    ----------
    lams_var : tf.Variable
        A tf.Variable object which holds a 2-D tensor representing 
        sets of quartic potential coefficients.
    
    Returns
    -------
    tf.tensor(tf.float32, tf.float32), tf.tensor(tf.float32, tf.float32)
        Two Tensorflow tensors: A tensor representing the model output 
        evaluated at lams_var, and a 2-D Tensorflow tensor representing 
        the gradient of the model prediction on a batch of inputs.
    """
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(lams_var)
        y = model(lams_var)
    return y, tape.gradient(y, lams_var)

def _rand_rotate(lams, rot_dist):
    """A function which randomly rotates a given set of quartic potential 
    coefficients by a specified angular distance on the unit hypersphere.

    Parameters
    ----------
    lams : tf.Variable
        A 2-D tf.Variable which represents sets of quartic potential 
        coefficients in the vicinity of points that the neural network
          labels as either false positives or false negatives.

    rot_dist : float
        A float that determines how far (in radians) each quartic 
        coefficient should be rotated.

    Returns
    -------
    tf.Tensor(tf.float32, tf.float32)
        A tf.Tensor obtained by rotating each set of quartic coefficients 
        in lams by rot_dist in random directions.
    """
    #Generate an ensemble of random unit vectors that are orthogonal to lams
    orth_rands = tf.random.normal(tf.shape(lams))
    orth_rands = orth_rands /(tf.norm(orth_rands, axis = 1, keepdims = True))
    orth_rands = (orth_rands - tf.math.reduce_sum(orth_rands * lams, axis=1)[:,np.newaxis]*lams)
    orth_rands = orth_rands /(tf.norm(orth_rands, axis = 1, keepdims = True))
    
    #Rotate lams in the direction of orth_rands by the angle rot_dist
    return orth_rands*(tf.math.sin(rot_dist)) + lams*(tf.math.cos(rot_dist))

def _one_step_validation(lams_var, grad_array, stepsize, false_pos, mask):
    """A convenience function to help in finding sets of quartic 
    coefficients that are correctly labelled in the vicinity of points 
    which are incorrectly labelled. Given a tf.Variable lams_var, follows 
    the direction of steepest descent (ascent) in the model prediction 
    for false positives (negatives) in order to locate a point very close 
    by that is correctly labelled by the neural network.

    Parameters
    ----------
    lams_var : tf.Variable
        A tf.Variable object which holds a 2-D tensor representing 
        sets of quartic potential coefficients.
    
    grad_array : tf.tensor(tf.float32, tf.float32)
        A Tensorflow tensor that holds the gradient of the model output 
        with respect to the input quartic coefficients.
    
    stepsize : tf.Tensor(tf.float32)
        Controls how rapidly to follow the direction of steepest descent 
        or ascent for each point. Automatically computed by 
        _estimate_step_size.
    
    false_pos : bool
        If True, the function will assume that lams_var are points in the 
        vicinity of false positives, and so should be looking for points 
        that the network classifies as negative. Otherwise, the function 
        will assume that lams_var are points in the vicinity of 
        false negatives, and so should be looking for points that the 
        network classifies as positive.

    mask : tf.Tensor(bool)
        A 1-dimensional tensor of booleans. The function will only 
        update indices of lams_var where mask is True.
    """
    # Follow the gradient to produce nearby points which the network classifies as more negative (positive) for false positives (negatives).
    if(false_pos):
        lams_var.scatter_nd_sub(tf.where(mask), tf.gather(stepsize*(grad_array),_tf_flatten(tf.where(mask))))
    else:
        lams_var.scatter_nd_add(tf.where(mask), tf.gather(stepsize*(grad_array),_tf_flatten(tf.where(mask))))
    # Now project the result back onto the unit hypersphere.
    lams_var.assign((lams_var.value())/(tf.norm(lams_var.value(), axis = 1, keepdims = True)))

def _find_prediction_boundary(model, lams_var, false_pos, converged, maxiter, init_rot):
    """A convenience function for finding how far misclassified points 
    are from the decision boundary. Uses gradient ascent/descent to modify 
    quartic coefficients of false positives (negatives) until the neural 
    network classifies them as negative (positive). Updates the values 
    in a tf.Variable object representing sets of quartic coupling 
    coefficients, and returns an array that describes whether or not the 
    attempt to locate all decision boundaries was successful.

    Parameters
    ----------
    lams_var : tf.Variable
        A tf.Variable  which represents sets of quartic potential 
        coefficients in the vicinity of points that the neural network 
        labels as either false positives or false negatives.
    
    stepsize : float
        Governs how rapidly the function follows the gradient of the 
        model prediction.

    false_pos : bool
        If True, the function assumes that lams_var are false positives. 
        If False, assumes they are false negatives.

    converged : tf.Tensor(bool)
        A Tensorflow tensor of Boolean values that determine whether or 
        not a given set of quartic coefficients in lams_var has been 
        sufficiently changed to cross the decision boundary.
        When this function is called, every element should be False.

    maxiter : int
        The maximum number of gradient ascent/descent iterations to perform.

    init_rot : float
        A parameter for randomly rotating coefficients that have zero 
        gradient, to avoid encountering local extrema that aren't past 
        the decision boundary.

    Returns
    -------
    tf.Tensor(bool)
        A Tensorflow tensor that labels whether the quartic coefficients 
        in a given index of lams_var have been sufficiently changed to 
        cross the decision boundary. If the function is successful in
        locating all decision boundaries, every element should be True.
    """
    i = tf.constant(0)
    while (tf.logical_and(tf.reduce_any(~converged), tf.less(i, maxiter))):
        # If any points have zero gradient and aren't past the decision boundary, fix them by randomly perturbing until they have nonzero gradients.
        _fix_zero_gradients(model, lams_var, init_rot, ~converged)
        # Now compute the model value and the gradient at lams_var.
        model_val, grad_array = _model_grad(model, lams_var)
        # Determine if any new points have converged.
        if(false_pos):
            converged = _tf_flatten(tf.math.less(model_val, 0.5))
        else:
            converged = _tf_flatten(tf.math.greater(model_val, 0.5))
        # Estimate the step size for the optimization step using backtracking
        stepsize = _estimate_step_size(model, lams_var, model_val, grad_array, false_pos, ~converged)
        _one_step_validation(lams_var, grad_array, stepsize, false_pos, ~converged)
        i = i + tf.constant(1)
    return converged

def _tf_scatter_nd_mask(tensor, mask, update):
    """Updates a tensor's values according to an update tensor, only for 
    indices where a a mask is not true.

    Parameters
    ----------
    tensor : tf.Tensor
        Some input tensor

    mask : tf.Tensor(bool)
        A tensor of booleans which should be "True" for indices where the 
        values of tensor should be replaced with update.

    update : tf.Tensor
        A tensor of the same shape and dtype as tensor.

    Returns
    -------
    tf.Tensor
        A tf.Tensor which has the values of update where mask is True, 
        and tensor where mask is False.
    """
    indices = tf.where(mask)
    return tf.tensor_scatter_nd_update(tensor, indices, tf.gather(update, _tf_flatten(indices)))

def _estimate_step_size(model, lams_var, model_val, grad_array, false_pos, mask):
    """Estimate the optimal size for each gradient descent step using 
    backtracking. Note that we don't use the backtracking for projected 
    gradient descent here, since the neural network already automatically 
    projects the input onto the unit hypersphere before evaluation, so the 
    result using the simpler unconstrained backtracking strategy will be 
    the same as the projected gradient result.

    Parameters
    ----------
    model : tf.keras.Model

    lams_var : tf.Variable
        A tf.Variable  which represents sets of quartic potential 
        coefficients in the vicinity of points that the neural network 
        labels as either false positives or false negatives.

    model_val : tf.Tensor(tf.float32)
        The model predictions on the initial value of lams_var.

    grad_array : tf.Tensor(tf.float32, tf.float32)
        A Tensorflow tensor that holds the gradient of the model output 
        with respect to the initial value of lams_var
    
    false_pos : bool
        If True, the points in lams_var denote false positives. 
        Otherwise, they denote false negatives.

    mask : tf.Tensor(bool)
        A boolean mask. The method will only increment the step size 
        estimate for indices where mask is True.

    Returns
    -------
    tf.Tensor(tf.float32)
        A tf.Tensor of step sizes.
    """
    stepsize = tf.ones(shape=(tf.shape(lams_var)[0],1), dtype = tf.float32)
    squared_grads = tf.math.square(tf.linalg.norm(grad_array, axis = 1, keepdims = True))
    if(false_pos):
        converged = tf.logical_or(~mask, _tf_flatten(tf.math.less_equal(model(lams_var - stepsize*grad_array)-model_val, - 0.5*stepsize*squared_grads)))
    else:
        converged = tf.logical_or(~mask, _tf_flatten(tf.math.greater_equal(model(lams_var + stepsize*grad_array)-model_val, 0.5*stepsize*squared_grads)))
    i = 0
    while tf.logical_and(tf.reduce_any(~converged), i < 25):
        stepsize = _tf_scatter_nd_mask(stepsize, ~converged, 0.5*stepsize)
        if(false_pos):
            converged = tf.logical_or(~mask, _tf_flatten(tf.math.less_equal(model(lams_var - stepsize*grad_array)-model_val, - 0.5*stepsize*squared_grads)))
        else:
            converged = tf.logical_or(~mask, _tf_flatten(tf.math.greater_equal(model(lams_var + stepsize*grad_array)-model_val, 0.5*stepsize*squared_grads)))
        i += 1
    return tf.clip_by_value(2.*stepsize, 1e-8, 0.01)

    
def _fix_zero_gradients(model, lams_var, init_rot, mask):
    """A method to deal with the possibility of points in the model that 
    have a gradient of exactly zero (so that it isn't suitable to use 
    gradient descent/ascent to find nearby points with the appropriate 
    label). This method finds such points and randomly rotates them a 
    small angle until the gradient is nonzero.

    Parameters
    ----------
    model : tf.keras.Model

    lams_var : tf.Variable
        Represents sets of quartic potential coefficients in the vicinity 
        of points that the neural network labels as either false positives 
        or false negatives.

    init_rot : float
        The initial angular distance which variables with zero gradient 
        should be rotated to find points with nonzero gradient. If this 
        angular distance fails to find points with nonzero gradients, 
        progressively larger rotations will be attempted until a nonzero 
        gradient is found.
    
    mask : tf.Tensor(bool)
        A mask denoting which zero gradients should be "fixed". Used to 
        avoid randomly rotating points that have already been deformed 
        past the decision boundary.
    """
    init_lams = tf.identity(lams_var.value())
    _, init_grads = _model_grad(model, lams_var)
    zero_grad = tf.logical_and(tf.math.less(tf.norm(init_grads, axis = 1), 1e-7), mask)
    rot_dist = init_rot
    i = 0
    while tf.math.reduce_any(zero_grad):
        # increase rot_dist if too many iterations have gone by without fixing all the zero-gradient points.
        if(tf.logical_and(tf.math.greater(i, 0), tf.math.equal(tf.math.floormod(i, 1000), tf.constant(0)))):
            rot_dist = tf.constant(2.)*rot_dist
        # Update the lams_var to randomly rotate those points at which the model has zero gradient.
        lams_var.scatter_nd_update(tf.where(zero_grad), _rand_rotate(tf.gather(lams_var, _tf_flatten(tf.where(zero_grad))), rot_dist))
        # Check to see whether the updated points still have zero gradient.
        _, init_grads = _model_grad(model, lams_var)
        zero_grad = tf.logical_and(tf.norm(init_grads, axis = 1) == 0., mask)
        # If there are still points with zero gradient, reset these points to their initial values so we can rotate them randomly again in the next iteration.
        lams_var.scatter_nd_update(tf.where(zero_grad), tf.gather(init_lams, _tf_flatten(tf.where(zero_grad))))
        # Increment a counter of how many loops have been performed.
        i += 1

def _random_rot_search(model, lams_var, false_pos, init_rot):
    """A method to deal with points that still haven't been deformed past 
    the decision boundary by the gradient descent/ascent strategy.
    This function randomly rotates these points until points that are past 
    the decision boundary are found.

    Parameters
    ----------
    model : tf.keras.Model

    lams_var : tf.Variable
        A tf.Variable  which represents sets of quartic potential 
        coefficients in the vicinity of points that the neural network 
        labels as either false positives or false negatives.

    false_pos : bool
        If True, the function will assume that lams_var are points in the 
        vicinity of false positives, and so should be looking for points 
        that the network classifies as negative. Otherwise, the function 
        will assume that lams_var are points in the vicinity of 
        false negatives, and so should be looking for points that the 
        network classifies as positive.

    init_rot : float
        The initial angular distance which variables should be rotated to 
        find points across the decision boundary. If this angular distance 
        fails to find valid points, progressively larger rotations will be 
        attempted until a point across the decision boundary is found.
    """
    # Keep track of the positions of the variables before rotation.
    init_lams = tf.identity(lams_var.value())

    # Determine which points are not yet deformed across the 
    # decision boundary.
    if(false_pos):
        converged = tf.less(_tf_flatten(model(lams_var)), 0.5)
    else:
        converged = tf.greater(_tf_flatten(model(lams_var)), 0.5)
    rot_dist = init_rot
    i = 0
    while tf.math.reduce_any(~converged):
        # increase rot_dist if too many iterations have gone by 
        # without converging for all points.
        if(tf.logical_and(tf.math.greater(i, 0), tf.math.equal(tf.math.floormod(i, 1000), tf.constant(0)))):
            rot_dist = tf.constant(2.)*rot_dist
        # Update the lams_var to randomly rotate those points 
        # at which the model has zero gradient.
        lams_var.scatter_nd_update(tf.where(~converged), _rand_rotate(tf.gather(lams_var, _tf_flatten(tf.where(~converged))), rot_dist))
        # Check to see whether the updated points are now 
        # across the decision boundary.
        if(false_pos):
            converged = tf.less(_tf_flatten(model(lams_var)), 0.5)
        else:
            converged = tf.greater(_tf_flatten(model(lams_var)), 0.5)
        # If there are still points that aren't past the decision boundary, 
        # reset these points to their initial values so we can rotate them 
        # randomly again in the next iteration.
        lams_var.scatter_nd_update(tf.where(~converged), tf.gather(init_lams, _tf_flatten(tf.where(~converged))))
        # Increment a counter of how many loops have been performed.
        i += 1
    
@tf.function
def _find_accurate_points(model, lams_var, lams_init, false_pos, maxiter = tf.constant(10000), init_rot = tf.constant(1e-3)):
    """A function which, given tf.Variables of points in the quartic 
    potential which the neural network classifies incorrectly as 
    false positives (negatives), deforms them into nearby points across 
    the neural network's decision boundary. Used to validate the neural 
    network by determining how far away an incorrectly classified point 
    is from the decision boundary. The strategy employed here is to 
    follow the direction of steepest descent (ascent) of the neural 
    networks' prediction function with respect to the quartic coefficients 
    to find points across the decision boundary, and then use bisection 
    root-finding methods to deformed points to be as near to the decision 
    boundary as possible.

    Parameters
    ----------
    lams_var : tf.Variable
        A tf.Variable  which represents sets of quartic potential 
        coefficients in the vicinity of points that the neural network 
        labels as either false positives or false negatives.

    lams_init : tf.Variable
        Another tf.Variable that initially carries identical values to 
        lams_var. This will be updated as part of the bisection search 
        for the decision boundary.

    false_pos : bool
        If True, the function will assume that lams_var are 
        false positives, and so should be looking for points that 
        the network classifies as negative. Otherwise, the function will 
        assume that lams are false negatives, and so should be looking for 
        points that the network classifies as positive.

    maxiter : int
        An integer which controls how long to continue iterating in order 
        to find correctly labelled points in the vicinity of 
        false positives or negatives. This value should be large, and the 
        number of iterations should generally never approach it, but 
        this ensures that the program will not run indefinitely.

    init_rot : float
        A parameter that governs how far to rotate points which have 
        exactly zero gradient, as in _fix_zero_gradients, as well 
        as for the search based on random rotations for points for which 
        the gradient ascent/descent-based strategy fails.
    """
    # If any points have zero gradient, randomly rotate them by small 
    # angles until they don't.
    _fix_zero_gradients(model, lams_var, init_rot, tf.ones(tf.shape(lams_var)[0], dtype = bool))
    # Consider points as "converged" if they are on the other 
    # side of the decision boundary from the initial point.
    if(false_pos):
        converged = _tf_flatten(tf.math.less(model(lams_var), 0.5))
    else:
        converged = _tf_flatten(tf.math.greater(model(lams_var), 0.5))
    # Perform the gradient descent/ascent search for points past 
    # the decision boundary. The array converged keeps track of 
    # whether this attempt was successful.
    converged = _find_prediction_boundary(model, lams_var, false_pos, converged, maxiter, init_rot)

    # If the gradient descent strategy was unsuccessful, randomly rotate 
    # the unconverged points until they are past the decision boundary.
    if(tf.reduce_any(~converged)):
        _random_rot_search(model, lams_var, false_pos, init_rot)
    # Finally, refine the deformed points in lams_var to as close to 
    # the decision boundary as possible using bisection. The parameter 
    # init_rot, which by default will already be at least an order of 
    # magnitude lower than the accuracy tolerance of the active learning 
    # program, will also serve as the tolerance for the bisection root 
    # finding algorithm.
    _bisection_method(model, lams_var, lams_init, false_pos, init_rot/10.)

def _bisection_method(model, lams_var, lams_init, false_pos, tol):
    """A method that, given sets of points lams_var and lams_init that 
    are on either side of the decision boundary, updates all elements 
    of lams_var to have its model prediction be within tol of 0.5, 
    the decision boundary. It accomplishes this by repeatedly iterating 
    bisections (projected onto the unit hypersphere) between lams_var 
    and lams_init.

    Parameters
    ----------
    model : tf.keras.Model
    
    lams_var : tf.Variable
        A tf.Variable which represents sets of quartic potential 
        coefficients in the vicinity of points that the neural network 
        labels as either false positives or false negatives.

    lams_init : tf.Variable
        A tf.Variable which represents sets of quartic potential 
        coefficients that ARE false positives or false negatives.

    false_pos : bool
        If True, the function will assume that lams_init are 
        false positives, and so lams_var should be points 
        which are classified as negative. Otherwise, the function 
        will assume that lams_init are false negatives, and so 
        lams_var should be points which are classified as positive.

    tol : float
        The level of closeness to the decision boundary that lams_var 
        should be deformed to reach, without crossing.
    """
    # Assess which points are not yet converged to values within 
    # tol from the decision boundary, but across it.
    converged = tf.math.less(tf.math.abs(_tf_flatten(model(lams_var)) - 0.5), tol)
    # As long as some lams_var elements are more than tol away 
    # from the decision boundary, perform bisection root finding iterations.
    while tf.reduce_any(~converged):
        # Locate the midpoints between each lams_var and lams_init, 
        # and project it onto the unit hypersphere.
        bisection = lams_var + ((lams_init - lams_var)/2.)
        bisection = bisection / tf.norm(bisection, axis = 1, keepdims = True)
        # Update each element of lams_var and lams_init to the corresponding 
        # value of the midpoint, depending on which side of the decision boundary 
        # the midpoint is on.
        if(false_pos):
            init_update = tf.where(tf.logical_and(tf.greater(_tf_flatten(model(bisection)), 0.5), ~converged))
            var_update = tf.where(tf.logical_and(tf.less_equal(_tf_flatten(model(bisection)), 0.5), ~converged))
        else:
            init_update = tf.where(tf.logical_and(tf.less_equal(_tf_flatten(model(bisection)), 0.5), ~converged))
            var_update = tf.where(tf.logical_and(tf.greater(_tf_flatten(model(bisection)), 0.5), ~converged))
        lams_init.scatter_nd_update(init_update, tf.gather(bisection, _tf_flatten(init_update)))
        lams_var.scatter_nd_update(var_update, tf.gather(bisection, _tf_flatten(var_update)))
        # Update converged.
        converged = tf.math.less(tf.math.abs(_tf_flatten(model(lams_var)) - 0.5), tol)


def get_false_pos_and_neg_tf(model, ds):
    """A function which extracts all sets of quartic coefficients in a 
    Tensorflow dataset that the neural network classifies incorrectly, 
    either false positives (points it incorrectly classifies as 
    bounded-from-below) or false negatives (points it incorrectly 
    classifies as NOT bounded-from-below).

    Parameters
    ----------
    model : tf.keras.Model
    
    ds: tf.data.Dataset
        A Tensorflow dataset representing labelled sets of quartic 
        potential coefficients.

    Returns
    -------
    tuple of tf.Tensors
        Two 2-D tensors representing the sets of false positive and false 
        negative quartic coefficients, respectively.
    """
    false_pos = []
    false_neg = []
    for x, y in ds:
        pred = _tf_flatten(model(x, training=False))
        false_pos.append(tf.gather(x, _tf_flatten(tf.where(tf.logical_and(tf.math.greater(pred, tf.constant(0.5)), ~y)))))
        false_neg.append(tf.gather(x, _tf_flatten(tf.where(tf.logical_and(tf.math.less_equal(pred, tf.constant(0.5)), y)))))
    return tf.concat(false_pos, axis = 0), tf.concat(false_neg, axis = 0)