"""This module contains the core of BFBrain's training capabilities, in particular the class BFBLearner,
the object which contains the neural network classifier and methods to execute the active learning training
loop.
"""


import os
import logging
import warnings
# Without disabling logging, notifications of each time a Tensorflow model is saved will be printed to the console.
logging.disable(logging.INFO)

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from functools import partial
import pickle

import tensorflow as tf
import numpy as np
from bfbrain.Custom_Layers import HypersphereProjectionLayer, ConcreteDenseDropout, get_weight_regularizer, get_dropout_regularizer

from bfbrain.Data_Manager import DataManager, np_data
from bfbrain.AL_Metrics import *


import matplotlib.pyplot as plt

from scipy.special import hyp2f1, gamma
from scipy.optimize import root_scalar

def _n_sphere_fraction(sd, n):
    """Given an angle in radians and a number of dimensions, computes the fraction of the surface area of the unit hypersphere that is subtended by that angle in that
    number of dimensions.

    Parameters
    ----------
    sd : float
        An angle (in radians).
    
    n : int
        The number of dimensions of a unit hypersphere.
    
    Returns
    -------
    float
        The fraction of the n-dimensional hypersphere's surface area that is subtended by the angle sd.
    """
    return (gamma(0.5*n) / (gamma(0.5*(n - 1.))*np.sqrt(np.pi)))*(sd**(n-1.))*(hyp2f1(0.5, 0.5*(n - 1.), 0.5*(n + 1.), sd**2)) / (n - 1.)

def _get_hop_dist(data, n):
    """Tool for estimating a reasonable distance scale hyperparameter for generating the pool of candidate points to be added to the training set for active learning
    (see BFBLearner's documentation). Does so by determining the angle delta (in radians) such that on a unit hypersphere of the specified number of dimensions, the
    fraction of the unit hypersphere's surface area subtended by the angle delta will be equal to the fraction of points that are bounded-from-below in a specified set of 
    labelled quartic potential coefficients.

    Parameters
    ----------
    data : np_data
        Labelled sets of quartic potential coefficients, uniformly sampled from the surface of an n-dimensional hypersphere
    
    n : int
        The number of independent real components of each set of quartic coefficients necessary to uniquely specify a potential.
    
    Returns
    -------
    float
        A recommended distance scale hyperparameter for active learning (hop_dist in BFBLearner).
    """
    frac = len(data.pos) / data.n_elements()
    if frac > 0.5:
        return 0.5*np.pi
    def root_goal(sd):
        return _n_sphere_fraction(sd, n) - frac
    return np.arcsin(root_scalar(root_goal, bracket = [0., 1.]).root)


def create_seq_network(lam_len, n_layers, n_neurons, l, N):
    """Create a sequential Bayesian neural network approximated by concrete dropout.

    Parameters
    ----------
    lam_len : int
        The number of quartic coefficients needed to uniquely specify a potential in the model.
    n_layers : int
        The number of hidden layers of neurons to include in the model. Generally recommended to be O(a few)
    n_neurons : int
        The number of neurons in each dense layer. Recommended to be O(100).
    l : float
        The prior length scale parameter of the neural network. Weights have a prior distribution of N(0, 1/l**2).
    N : int
        The number of entries in the training data. Needed to determine the appropriate loss regularization terms in concrete dropout.

    Returns
    -------
    tf.keras.Sequential
        Returns a neural network 
    """
    #An input layer and a normalization layer.
    model = tf.keras.Sequential([tf.keras.Input(shape=(lam_len,)), HypersphereProjectionLayer()])
    wr = get_weight_regularizer(N, l, 1.0)
    dr = get_dropout_regularizer(N, 1.0, cross_entropy_loss = True)
    if n_layers >= 1:
        model.add(tf.keras.layers.Dense(n_neurons, activation = 'relu', kernel_regularizer = tf.keras.regularizers.L2(l2 = wr), use_bias = False))
    # Stack sequential neural network layers of dense neurons.
    for _ in range(1, n_layers):
        model.add(ConcreteDenseDropout(n_neurons, weight_regularizer = wr, dropout_regularizer = dr, activation = 'relu', use_bias = False))
    # Add a sigmoid activation layer at the end of the neural network.
    model.add(ConcreteDenseDropout(1, weight_regularizer = wr, dropout_regularizer = dr, activation = 'sigmoid', use_bias = False))
    return model


# A class which holds cumulative information about the model performance at each epoch. An instance of this class is saved as part of the BFBLearner object.
class AL_history:
    """Holds cumulative information about the model performance at each epoch. An instance of this class is saved as part of the BFBLearner object.

    Attributes
    ----------
    losses : list(float)
        Represents the loss values at each training epoch of the neural network model, calculated over the training set.

    accuracy : list(float)
        Represents the binary accuracy at each training epoch of the neural network model, calculated over the training set.

    val_losses : list(float)
        Represents the loss values at each training epoch of the neural network model, calculated over the validation set. 
        Only tracked if the validation performance is tracked during the call to model.fit during training, which the current BFBrain.BFBLearner.AL_loop method doesn't do.

    val_accuracy : list(float)
        Represents the binary accuracy at each training epoch of the neural network model, calculated over the validation set.
        Only tracked if the validation performance is tracked during the call to model.fit during training, which the current BFBrain.BFBLearner.AL_loop method doesn't do.

    Parameters
    ----------
    None
    """
    def __init__(self):
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.val_accuracy = []

    def append_history(self, history):
        """Merges information from the latest active learning iteration, extracted from the output of the Tensorflow fit method, into the AL_history object.

        Parameters
        ----------
        history : Tensorflow history object
        """
        self.losses = self.losses + history.history["loss"]
        self.accuracy = self.accuracy + history.history["binary_accuracy"]
        try:
            self.val_losses = self.val_losses + history.history["val_loss"]
            self.val_accuracy = self.val_accuracy + history.history["val_binary_accuracy"]
        except:
            self.val_losses = []
            self.val_accuracy = []

    def plot_history(self, filepath = None):
        """Plots the data stored in AL_history.

        Parameters
        ----------
        filepath : str, optional
            If specified, saves the resulting plots as .png images to the directory named in filepath. If not specified, the method just displays them on the console.
        """
        plt.figure()
        plt.plot(self.losses)
        if len(self.val_losses) > 0:
            plt.plot(self.val_losses)
            plt.legend(["train_loss", "val_loss"])
        else:
            plt.legend(["train_loss"])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        if(filepath is None):
            plt.show()
        else:
            plt.savefig(filepath+'/history_loss.png')
        
        plt.close()
        plt.figure()
        plt.plot(self.accuracy)
        if len(self.val_accuracy) > 0:
            plt.plot(self.val_accuracy)
            plt.legend(["train_accuracy", "val_accuracy"])
        else:
            plt.legend(["train_accuracy"])
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        if(filepath is None):
            plt.show()
        else:
            plt.savefig(filepath+'/history_accuracy.png')
        plt.close()

class BFBLearner:
    """Class which controls the active learning loop. Holds the model, the training and validation data, and some information about the training so far, and includes methods which perform the training loop and
    save the model for further training or exporting.

    Attributes
    ----------
    dm : DataManager
        The DataManager object that handles the generation and labelling of new training and validation data.

    model : tf.keras.Sequential
        The sequential neural network that classifies potentials as bounded-from-below.

    data_train : np_data
        The data on which the neural network is trained. Is periodically augmented during the active learning loop.

    data_val : np_data or None
        A separate np_data object on which the neural network performance can be tested. If metrics doesn't include any performance metrics on validation data, this should be None.

    metrics : list(BFBrain.ALMetric)
        A list of objects which inherit from the abstract class ALMetric. These will represent the performance metrics that are tracked over each active learning iteration.

    history : AL_history
        An AL_history object which tracks the training loss and binary accuracy over each epoch.

    learning_rate : float
        The learning rate used for the Adam optimizer during neural network training.

    hop_dist : float
        The distance scale used to generate new training points in the vicinity of existing bounded-from-below points, given as the hop_dist argument to DataManager's generate_L function.

    rand_fraction : float
        The percentage of points generated by DataManager's generate_L function that are randomly sampled instead of sampled in the vicinity of positively labelled points.

    idx : int
        The number of active learning rounds that the model has completed.

    l_constant : float
        The prior length scale for the neural network weights. The neural network is constructed so that the prior distribution on the weights is N(0, 1/l**2).

    Methods
    -------
    init_for_first_run
        The recommended constructor for initializing a BFBLearner object from scratch.

    from_file
        The constructor for loading a saved BFBLearner object.

    save_AL_state
        Used to save the BFBLearner object.

    redefine_model
        Used to replace the neural network with a new one (for example, adding layers, adjusting weight priors, or changing the number of hidden neurons/layer).
        A use case would be generating a single BFBLearner object with a large labelled validation set, saving the object with an untrained neural network, and then
        loading the object in different contexts to experiment with different neural network architectures and hyperparameters.

    set_l_constant
        Sets l_constant to a new value. Same use case as redefine_model.

    add_metrics
        Adds new children of the ALMetric class to the BFBLearner object. Use this method rather than directly appending objects to the metrics attribute, since
        otherwise errors can occur, for example from adding a metric which requires a validation data set when the BFBLearner object had no previous validation data set.

    plot_metrics
        Creates simple plots of all the metrics recorded in metrics. Useful to get a quick visual sense of the performance of the model after active learning finishes.

    AL_loop
        The core active learning function. Executes the active learning loop to train the classifier. Highly customizable.
    """
    def __init__(self, dm, model, data_train, data_val, metrics, history, learning_rate, hop_dist, rand_fraction, idx, l_constant):
        self.dm = dm
        self.data_train = data_train
        self.data_val = data_val
        self.model = model
        self.learning_rate = learning_rate
        self.hop_dist = hop_dist
        self.rand_fraction = rand_fraction
        self.idx = idx
        self.metrics = metrics
        self.history = history
        self.l_constant = l_constant
        self._compile_model()
    
    @classmethod
    def from_file(cls, directory):
        """A constructor which loads an instance of BFBLearner from a folder.

        Parameters
        ----------
        directory : str
            Should denote a directory into which a previous BFBLearner object was saved.
        """
        data_train = np_data.from_file(directory + '/data_train')
        data_val = np_data.from_file(directory + '/data_val')
        with open(directory + '/dm.pickle', mode = 'rb') as f:
            dm = pickle.load(f)
        with open(directory + '/variables.pickle', mode = 'rb') as f:
            kwargs = pickle.load(f)
        with open(directory + '/history.pickle', mode = 'rb') as f:
            history = pickle.load(f)
        with open(directory + '/metrics.pickle', mode = 'rb') as f:
            metrics = pickle.load(f)
        model = tf.keras.models.load_model(directory+'/model')
        return cls(dm, model, data_train, data_val, metrics, history, **kwargs)
    
    @classmethod
    def init_for_first_run(cls, dm, n_layers, n_neurons, metrics, nlams, nlams_val = -1, learning_rate = 0.001, rand_fraction = 0., hop_dist = None, l_constant = 0.1, truth_label_fn = None, use_truth_for_train = False, val_label_kwargs = None, balance_val = False):
        """The recommended constructor when not loading a saved BFBLearner object.

        Parameters
        ----------
        dm : DataManager
            The DataManager object that will handle the generation and processing of training and validation data.

        n_layers : int
            The number of layers of hidden layers in the neural network.

        n_neurons : int
            The number of neurons in each hidden layer.

        metrics : list(BFBrain.ALMetric)
            A list of objects which inherit from the abstract class ALMetric. These will represent the performance metrics that are tracked over each active learning iteration.

        nlams : int
            The number of sets of quartic potential coefficients to generate to produce the initial training data.

        nlams_val : int, optional
            The number of sets of quartic potential coefficients to generate to produce the validation data. If not specified, the value 100*nlams is used.

        learning_rate : float, default=0.001
            The learning rate used for the Adam optimizer during neural network training.

        rand_fraction : float, default=0.
            The percentage of points generated by DataManager's generate_L function that are randomly sampled instead of sampled in the vicinity of positively labelled points.

        hop_dist : float, optional
            The distance scale used to generate new training points in the vicinity of existing bounded-from-below points, given as the hop_dist argument to DataManager's generate_L function.
            If not specified, the value is estimated using BFBrain.Active_Learning._get_hop_dist

        l_constant : float, default=0.1
            The prior length scale for the neural network weights. The neural network is constructed so that the prior distribution on the weights is N(0, 1/l**2).

        truth_label_fn : callable, optional
            Must take a 1-D NumPy array representing a single set of quartic coefficients and return a Boolean True if the potential they describe is bounded from below, False otherwise.
            If this argument is specified, the method will use this callable to label the validation data set instead of DataManager's oracle, and if use_truth_for_train is True, then it will also 
            use this function to label the initial training data. This is used in specific instances when a fast symbolic expression
            for the bounded-from-below constraints is known, and the performance of the classifier training loop can be evaluated in the absence of noise due to the approximate labeller. 
            Obviously the use case of the classifier is for potentials where such a symbolic expression is NOT known, so the real-world model building usefulness of this option is limited.

        use_truth_for_train : bool, default=False
            If True, use truth_label_fn to label the initial training data instead of the DataManager's oracle function. Has no effect unless truth_label_fn is specified.

        val_label_kwargs : dict, optional
            An alternate set of keyword arguments for the oracle that can be specified when labelling the validation data instead of using the settings in the DataManager object. Useful if,
            for example, a less expensive oracle can be used for the validation labelling than for labelling the training data, or if trying to gauge the effect of using a noisier oracle
            during training while still verifying performance with reliably labelled validation data.

        balance_val : bool, default=False
            If True, balance the validation data set between positive and negative examples using DataManager's balance_array method, making binary accuracy of the classifier more informative at the cost of rendering the generating
            distribution for the validation data considerably more opaque. Generally recommended to leave False, since other performance metrics, such as F score, can help gauge the classifier performance
            without needing to rebalance the validation data.
        """
        # Create the initial training data for the model to use.
        print('creating training data...')
        if(use_truth_for_train):
            data_train = dm.create_random_data(nlams, validation = False, truth_label_fn = truth_label_fn)
        else:
            data_train = dm.create_random_data(nlams, validation = False)
        if hop_dist is None:
            hop_dist = _get_hop_dist(data_train, dm.lam_len)
        dm.balance_array(data_train)
        print('done!')
        # If we have any metrics that use validation data, also create a validation set.
        if any([metric.sc_type == 'val' for metric in metrics]):
            print('creating validation data...')
            if(nlams_val==-1):
                data_val = dm.create_random_data(100*nlams, validation = True, truth_label_fn = truth_label_fn, label_kwargs = val_label_kwargs)
            else:
                data_val = dm.create_random_data(nlams_val, validation = True, truth_label_fn = truth_label_fn, label_kwargs = val_label_kwargs)
            if balance_val:
                dm.balance_array(data_val)
            print('done!')
        else:
            data_val = None

        # Create the neural network.
        model = create_seq_network(dm.lam_len, n_layers, n_neurons, l_constant, data_train.n_elements())
        # Set up a dictionary of some keyword arguments needed to initialize the class.
        kwargs = dict(learning_rate = learning_rate, hop_dist = hop_dist, rand_fraction = rand_fraction, idx = 0, l_constant = l_constant)

        return cls(dm, model, data_train, data_val, metrics, AL_history(), **kwargs)

    def _compile_model(self):
        """A helper function that simply recompiles the model.
        """
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),  metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])
    
    def _update_regularizers(self, l, N, reinitialize_weights):
        """A function to update the neural network for a new round of training. Changes the weight decay constants to ensure the prior weight distribution is given by the weight prior length scale
        and, if the flag is set for it, will also randomize the model weights.

        Parameters
        ----------
        l : float
            The new prior length scale.

        N : int
            The number of elements in the training data.

        reinitialize_weights : bool
            If True, randomize model weights after updating the network's hyperparameters. If False, the model weights are maintained, but the weight decay constants are still changed.
        """
        n_layers = self._count_layers()
        new_model = create_seq_network(self.dm.lam_len, n_layers, tf.shape(self.model.layers[1].get_weights()[0][1])[0], l, N)
        if not reinitialize_weights:
            for i, layer in enumerate(new_model.layers):
                layer.set_weights(self.model.layers[i].get_weights())
        self.model = new_model
        tf.keras.backend.clear_session()
        self._compile_model()

    def _count_layers(self):
        """Counts the number of hidden layers in the neural network, by counting the number of ConcreteDenseDropout layers.
        """
        i = 0
        for layer in self.model.layers:
            if hasattr(layer, 'p_logit'):
                i += 1
        return i

    def set_l_constant(self, l_constant):
        """Sets the weight prior length scale l to a new value and updates the network to reflect it.

        Parameters
        ----------
        l_constant : float
            The new value for the class attribute l_constant, the weight prior length scale.
        """
        self.l_constant = l_constant
        self._update_regularizers(l_constant, self.data_train.n_elements(), reinitialize_weights = False)

    def _save_data(self, filepath, save_val = False):
        """Saves the training (and, if the flag is specified, the validation) data of the BFBLearner object for continuity between runs.

        Parameters
        ----------
        filepath : str
            A string which describes the name of the file to which the data should be saved.

        save_val : bool
            If True, save both the training data and the validation data. If False, save only the training data.
        """
        self.data_train.save_data(filepath+'_train')
        if(save_val and (self.data_val is not None)):
            self.data_val.save_data(filepath+'_val')

    def _get_kwargs(self):
        """A helper function which returns a dictionary with all of the variable attributes needed to reconstitute the model.
        """
        return dict(learning_rate = self.learning_rate, hop_dist = self.hop_dist, rand_fraction = self.rand_fraction, idx = self.idx, l_constant = self.l_constant)


    def _save_variables(self, filepath):
        """A method to save the values of some miscellaneous variables in the active learning class.

        Parameters
        ----------
        filepath : str
            A string which describes the name of the file to which the data should be saved.
        """
        kwargs = self._get_kwargs()
        with open(filepath+'.pickle', mode = 'wb') as f:
            pickle.dump(kwargs, f)
    
    def _print_variables(self, filepath = None):
        """A method to print out the miscellaneous variables of the BFBLearner object into the output file (or the console, if no file path is specified).

        Parameters
        ----------
        filepath : str, optional
            If specified, write the variables to a file 'output.txt' in the directory specified by the string. If not specified, the method will print out the variables to the console.
        """
        kwargs = self._get_kwargs()
        if(filepath is None):
            print(kwargs)
        else:
            with open(filepath+'/output.txt', 'a') as f:
                print(kwargs, file=f)

    def save_AL_state(self, directory):
        """A method for saving all the relevant states for the BFBLearner object in a directory for later retrieval.

        Parameters
        ----------
        directory : str
            A string denoting a directory to which the data needed to reconstruct the BFBLearner object will be saved.
        """
        #If the directory doesn't already exist, create it
        if(not os.path.isdir(directory)):
            os.mkdir(directory)
        #save the training and validation data
        self._save_data(directory+'/data', save_val = True)
        #save the DataManager object
        with open(directory + '/dm.pickle', mode = 'wb') as f:
            pickle.dump(self.dm, f)
        #save the model
        self.model.save(directory+'/model')
        #save the history object
        with open(directory+'/history.pickle', mode = 'wb') as f:
            pickle.dump(self.history, f)
        with open(directory+'/metrics.pickle', mode = 'wb') as f:
            pickle.dump(self.metrics, f)
        #Save the other class variables.
        self._save_variables(directory+'/variables')

    def redefine_model(self, n_layers, n_neurons, l, learning_rate = 0.001):
        """A method for redefining the parameters of the neural network. Should ONLY be called if the neural network is entirely untrained.

        Parameters
        ----------
        n_layers : int
            The number of hidden layers of neurons to include in the model. Generally recommended to be O(a few).

        n_neurons : int
            The number of neurons in each hidden layer. Recommended to be O(100).

        l : float
            The prior length scale parameter of the neural network. Weights have a prior distribution of N(0, 1/l**2).

        learning_rate : float, default=0.001
            The learning rate of the Adam optimizer for neural network training.
        """
        if(self.idx > 0):
            print('WARNING: This method will reset the neural network, but NOT the training data or the neural network random seed. Do NOT call with a partially trained network unless you want your results to not be reproducible!')
        tf.keras.backend.clear_session()
        self.model = create_seq_network(self.dm.lam_len, n_layers, n_neurons, l, self.data_train.n_elements())
        self.l_constant = l
        self.learning_rate = learning_rate
        self._compile_model()
        self.idx = 0
        self.history = AL_history()
        for metric in self.metrics:
            metric.reset_data()

    def _get_L_probabilities(self, L, score_fn, batch_size, probability_weighting):
        """A function which, given a 2-D NumPy array of quartic potential coefficients L, will return probability weights based on score_fn for sampling 
        in the vicinity of these points as part of generating new training set points.

        Parameters
        ----------
        L : np.array(np.float32, np.float32)
            A 2-D NumPy array representing sets of quartic coefficients for the potential.

        score_fn : callable
            A callable which takes a model and a 2-D tensor as inputs, and outputs some sort of scalar score for each input.

        batch_size : int
            The size of a batch of L elements that the program will consider simultaneously.

        probability_weighting : bool
            If True, the function will output a 1-D array of probabilities corresponding to each point in L, by weighting each point quartically according to its score_fn score. If False,
            returns None, which instructs the program to use a uniform probability.

        Returns
        -------
        np.array(np.float32) or None
            Return value represents the probability weighting for different points in L to be sampled near when generating candidate pools to be potentially added to the training set. If
            return value is None, then the active learning algorithm assumes uniform probability for all points.
        """
        if probability_weighting:
            L_ds = tf.data.Dataset.from_tensor_slices(L).batch(batch_size)
            out_scores = []
            for x in L_ds:
                out_scores.append(score_fn(self.model, x).numpy())
            out_scores = np.abs(np.concatenate(out_scores, axis = 0))
            if np.sum(out_scores)==0.:
                return None
            return out_scores**4 / np.sum(out_scores**4)
        else:
            return None


    def _generate_K(self, L, numK, score_fn):
        """A function to select new points K to be included in the next training iteration, given a sample pool L.

        Parameters
        ----------
        L : tf.Tensor(tf.float32, tf.float32)
            A 2-D Tensorflow tensor representing sets of quartic coefficients for the potential.

        numK : int
            The number of points to select from L to pass on for labelling and inclusion in the training data.

        score_fn : callable
            A callable which scores points in L. The points with the highest scores will be gathered and returned for labelling. The function must have a signature 
            (tf.keras.model, tf.tensor(tf.float32, tf.float32))-> tf.tensor(tf.float32). By default, the function AL_loop scores points based on mutual information, 
            but the user can specify any other function of the appropriate signature should they so choose.

        Returns
        -------
        tf.Tensor(tf.float32, tf.float32)
            A 2-D Tensorflow tensor representing sets of quartic coefficients for the potential, specifically numK entries in L of which the model is most uncertain.
        """
        result = score_fn(self.model, L)
        # Select the top numK points (ranked by their score given by result), and return them.
        _, indices = tf.math.top_k(result, k = numK, sorted = False)

        # For metrics that require evaluation of the points proposed in the pool of candidates, record these metrics here.
        for metric in self.metrics:
            if metric.sc_type == 'pool' and hasattr(metric, 'score_fn'):
                if metric.score_fn.__name__ == score_fn.__name__:
                    metric.record_batch(result)
                else:
                    metric.record_batch(self.model, L)
            elif metric.sc_type == 'pool':
                metric.record_batch(self.model, L)
        return tf.gather(L, indices, axis=0).numpy()
    
    def _generate_K_batched(self, K_num, K_batches, K_factor, L_input, score_fn, prob_score_fn, truth_label_fn, probability_weighting, verbose):
        """A function to perform the function _generate_K in batches, to save on GPU memory and shorten execution time.
        Then returns an np_data object containing the labelled points K to be added to the training set.

        Parameters
        ----------
        K_num : int
            The number of sets of quartic coefficients each individual run of _generate_K should produce.

        K_batches : int
            The total number of runs of _generate_K that the function should perform.

        K_factor : int 
            To generate each batch of K_batch_size entries, K_factor*K_num candidate points are generated and the top K_num points are selected to be added to K.

        L_input : np.array(np.float32, np.float32)
            A 2-D NumPy array of sets of quartic coefficients of the potential that will be used to generate the pool of points that will be considered for addition to the training set.

        score_fn : callable
            A function that scores a batch of points given an input model, used in _generate_K. See _generate_K for more information.

        prob_score_fn : callable
            A function of the same signature as score_fn (but not necessarily the same function). If the probability_weighting flag is True, this function will be used to evaluate which points in L_input should be 
            sampled around most frequently when constructing the new training data.

        truth_label_fn : callable, optional
            If a callable, must take a 1-D NumPy array representing a single set of quartic coefficients and return a Boolean True if the potential they describe is bounded from below, False otherwise.
            If specified, training data will be labelled using this callable instead of the oracle in the DataManager object.

        probability_weighting : bool
            If True, weight the probability of selecting certain points in L_input to sample around over others, based on their evaluation with prob_score_fn. If False, sample all points in L_input uniformly.

        verbose : bool
            If True, print statements about the progress of the function.

        Returns
        -------
        np_data
            An np_data object holding the labelled sets of quartic coefficients to be added to the training set.
        """
        L_probs = self._get_L_probabilities(L_input, prob_score_fn, batch_size = K_factor*K_num, probability_weighting = probability_weighting)
        # Execute _generate_K K_num times.
        K_points = self._generate_K(tf.constant(self.dm.generate_L(K_factor*K_num, L_input, hop_dist=self.hop_dist, probs = L_probs, rand_fraction=self.rand_fraction), dtype=tf.float32), K_num, score_fn)
        for _ in range(1, K_batches):
            K_points = np.concatenate((K_points, self._generate_K(tf.constant(self.dm.generate_L(K_factor*K_num, L_input, hop_dist=self.hop_dist, probs = L_probs, rand_fraction=self.rand_fraction), dtype=tf.float32), K_num, score_fn)), axis = 0)
        if(verbose):
            print('Generated K, now labelling it...')
        #Label the K points.
        K_data = self.dm.create_data(K_points, truth_label_fn)
        if(verbose):
            print('Successfully created an additional training sample of ' + str(K_num*K_batches) + ' points in parameter space')
        return K_data
    
    def _balance_with_positives(self, K_num, K_factor, score_fn, verbose):
        """A function which will balance the number of negative and positive elements in data_train by exploiting the convexity of the set of all positive points. Can be used periodically if the overwhelming majority of newly
        generated points are negatively labelled.

        Parameters
        ----------
        K_num : int
            The number of sets of quartic coefficients each individual run of _generate_K should produce.

        K_factor : int
            To generate each batch of K_num entries, K_factor*K_num candidate points are generated and the top K_batch_size points are selected to be added to K.

        score_fn : callable
            A function that scores a batch of points given an input model, used in _generate_K. See _generate_K for more information.

        verbose : bool
            If True, print statements about the progress of the function.
        """
        # First, determine how many batches of results need to be added to (approximately) balance the array.
        num_batches = (len(self.data_train.neg) - len(self.data_train.pos)) // K_num
        # If we have more negative points than positives (or the array is already approximately balanced), the rebalancing technique here won't work, and therefore the method will exit without doing anything.
        if num_batches <= 0:
            return
        if(verbose):
            print('generating new positive points to rebalance the training data...')
        K_points = self._generate_K(tf.constant(self.dm._create_new_positives(self.data_train.pos, K_factor*K_num), dtype = tf.float32), K_num, score_fn)
        for _ in range(1, num_batches - 1):
            K_points = np.concatenate((K_points, self._generate_K(tf.constant(self.dm._create_new_positives(self.data_train.pos, K_factor*K_num), dtype = tf.float32), K_num, score_fn)), axis = 0)
        self.data_train.append_data(np_data(K_points, np.array([]))) 
    
    def _create_output_file(self, filepath):
        """A method to create the output file for recording results. Includes information about the architecture of the neural network.

        Parameters
        ----------
        filepath : str
            A string denoting the directory to which the output should be recorded.
        """
        with open(filepath+'/output.txt', 'w') as f:
            self.model.summary(print_fn = partial(print, file=f))
    
    def _print_status(self, ind, filepath = None):
        """A method to print certain status variables to an output file or to the console.

        Parameters
        ----------
        ind : int
            The number of training rounds (full active learning cycles) the neural network has currently undergone.

        filepath : str, optional
            A string specifying where the information should be saved. If not specified, print the data to the console.
        """
        if filepath is None:
            print('Metrics for round {}:'.format(ind))
            for metric in self.metrics:
                metric.print_status()
        else:
            with open(filepath+'/output.txt', 'a') as f:
                print('Metrics for round {}:'.format(ind), file = f)
                for metric in self.metrics:
                    metric.print_status(file = f)

    def _get_status(self, ind, filepath, verbose = False):
        """A method to find the current status variables (for the performance metrics) and print them to a file (and possibly to the console). 

        Parameters
        ----------
        ind : int
            The number of training rounds (full active learning cycles) the neural network has currently undergone.

        filepath : str
            A string specifying where the information should be saved.

        verbose : bool, default=False
            If True, print the performance metric statuses found with this method to the console in addition to saving it.
        """
        if(verbose):
            self._print_status(ind)
        self._print_status(ind, filepath=filepath)
    
    def add_metrics(self, metrics, nlams_val = 100000):
        """add new metrics to the BFBLearner instance after initialization. Appends any metric or list of metrics specified to self.metrics. If an added metric involves
        measuring the classifier performance on a validation set, and the ActiveLearning instance doesn't have a validation set yet, this method will generate one.

        Parameters
        ----------
        metrics : ALMetric or list of ALMetrics
            A metric or list of objects which inherit from the BFBrain.ALMetric abstract class.

        nlams_val : int, default=100000
            The number of points to be created and labelled to produce a validation set, if a new validation set needs to be generated.
        """
        if self.idx > 0:
            warnings.warn('Warning: Adding a metric after already performing some active learning iterations will result in the metric not recording its values for the iterations already completed.')
        if isinstance(metrics, list):
            self.metrics = self.metrics + metrics
        else:
            self.metrics = self.metrics + [metrics]
        self._check_for_val_metrics(nlams_val)
    
    def _check_for_val_metrics(self, nlams_val = 100000):
        """A helper function which determines if a validation set needs to be created, given the metrics in self.metrics, and creates that set if it doesn't exist.

        Parameters
        ----------
        nlams_val : int, default=100000
            The number of points to be created and labelled to produce a validation set, if a validation set needs to be generated.
        """
        for metric in self.metrics:
            if metric.sc_type == 'val' and (self.data_val is None):
                self.data_val = self.dm.create_random_data(nlams_val, validation = True)
                self.dm.balance_array(self.data_val)

    def plot_metrics(self, filepath = None):
        """A method to plot all metrics being recorded in the BFBLearner's metrics object.

        Parameter
        ---------
        filepath : str, optional
            If specified, the method will save the plots to the folder specified in filepath (provided that folder exists). If not specified, simply prints the plots to the console.
        """
        for metric in self.metrics:
            metric.plot_metric(filepath)

    def get_calibration_uncertainties(self, score_fn, nlams = 1000000, n_trials = None, batch_size = 2000000):
        """A function which gets a sense of the range of uncertainty values (and what constitutes a highly uncertain point).
        It does this by creating a large sample of unlabelled data points by uniformly sampling the unit hypersphere,
        and getting the outputs of a specified uncertainty score over the points in the distribution which the model
        predicts to be positive.

        Parameters
        ----------
        score_fn : {'BALD', 'MaxEntropy', 'variation_ratios', 'random', 'uncertainty', 'predictive_variance'} or callable
            Either a string which denotes which of the implemented strategies for scoring points to add to the training set to employ,
            or a callable representing a custom function to perform this role. Any custom functions must have a signature 
            (tf.keras.model, tf.tensor(tf.float32, tf.float32))-> tf.tensor(tf.float32) or (tf.keras.model, tf.tensor(tf.float32, tf.float32), int)-> tf.tensor(tf.float32), depending on 
            if n_trials is specified or not. The string inputs correspond to scoring based on, in order: Mutual information, Shannon entropy, variation ratios, a random score, expected gradient length,
            query by dropout committee (QBDC), and directly computed variance of the neural network predictions.
        
        nlams : int, default=1000000
            The number of sets of quartic potential coefficients that the method should generate

        n_trials : int, optional
            Many of the implemented score_fn methods ('BALD', 'MaxEntropy', 'variation_ratios', 'uncertainty', and 'predictive_variance') have an optional argument controlling the number of forward passes 
            through the network to compute their values. This parameter allows this value to be specified for score_fn's evaluations.

        batch_size : int, default=200000
            The size of batches of the elements of lams that are transferred to the GPU simultaneously. If OOM errors are encountered, it is recommended to reduce batch_size.
        """
        # Determine how to select additional points to add to the training set based on the input of score_fn.
        if not callable(score_fn):
            try:
                score_fn = scoring_funcs[score_fn]
            except KeyError:
                raise Exception('score_fn must be a callable or one of the strings {}.'.format(list(scoring_funcs.keys())))

        if n_trials is not None:
            score_fn = tf.function(partial(score_fn, n_trials = n_trials), jit_compile = True)
            MC_call = partial(MC_call_fast, n_trials = n_trials)
        else:
            score_fn = tf.function(score_fn, jit_compile = True)
            MC_call = partial(MC_call_fast, n_trials = 100)

        # Create a new set of nlams random quartic coefficients.
        lams = self.dm.create_random_lambdas(nlams, validation = True)
        ds = tf.data.Dataset.from_tensor_slices(lams).batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE).cache()

        # Find the uncertainty scores on lams
        out_score = []
        out_pred = []
        for x in ds:
            out_score.append(score_fn(self.model, x).numpy())
            out_pred.append(MC_call(self.model, x).numpy())

        out_score = np.concatenate(out_score, axis = 0)
        out_pred = np.concatenate(out_pred, axis = 0)

        return out_score[out_pred >= 0.5]


        

    
    def AL_loop(self, filepath = 'saved_AL', K_batch_size = 500, K_batch_num = 10, K_factor = 100, score_fn = 'BALD', nstop = 20, full_train_interval = 1, epoch_patience=100, epoch_limit=20000, batch_size = 200000, val_batch_size = -1,
               save_interval = 5, stopping_cond = None, prob_score_fn = None, truth_label_fn = None, score_ntrials = None, prob_score_ntrials = None, reinitialize_weights = True,
                 rebalance_train_data = False, L_probability_weighting = False, plot_metrics = False, verbose = False):
        """The core function for the active learning loop. Performs active learning for a specified number of rounds with a customizable query strategy and a number of options exposed to the user.
        BFBrain analysis principally consist of execution(s) of this method to train the BFBLearner's classifier.

        Parameters
        ----------
        filepath : str, default='saved_AL'
            A string describing the name of a directory (which will be created, if it doesn't exist) into which the Active_Learning object will be saved after training.

        K_batch_size : int, default=500
            Additional training data, called K, is generated in batches of K_batch_size entries. After each cycle of training,
            K_batch_num batches, each with K_batch_size entries, are generated and added to the training data.

        K_batch_num : int, default 10
            The number of batches, of batch_size K_batch_size, of additional training data that is generated after each cycle of training.

        K_factor : int, default 100
            To generate each batch of K_batch_size entries, K_factor*K_batch_size candidate points are generated and the top K_batch_size points are selected to be added to K.

        score_fn : {'BALD', 'MaxEntropy', 'variation_ratios', 'random', 'uncertainty', 'predictive_variance'} or callable
            Either a string which denotes which of the implemented strategies for scoring points to add to the training set to employ,
            or a callable representing a custom function to perform this role. Any custom functions must have a signature 
            (tf.keras.model, tf.tensor(tf.float32, tf.float32))-> tf.tensor(tf.float32) or (tf.keras.model, tf.tensor(tf.float32, tf.float32), int)-> tf.tensor(tf.float32), depending on 
            if score_ntrials is specified or not. The string inputs correspond to scoring based on, in order: Mutual information, Shannon entropy, variation ratios, a random score, expected gradient length,
            query by dropout committee (QBDC), and directly computed variance of the neural network predictions.

        nstop : int, default=20
            The number of total cycles (generating new training data, fitting the neural network to the training data, and recording metrics) to perform.

        full_train_interval : int, default=1
            The active learning function will train over the entire training data set every full_train_interval cycles. 
            Otherwise, it will only train the network on the most recently generated set of new training data, for a much smaller number of epochs. Recommended setting is 1,
            so that dropout approximation to a Bayesian neural network holds rigorously.

        epoch_patience : int, default=100
            A parameter for the training in individual cycles. The neural network will stop training early in each cycle if 
            the network's performance on the validation data has not improved for epoch_patience epochs.

        epoch_limit : int, default=20000
            The maximum number of epochs for the neural network to train each cycle.

        batch_size : int, default=200000
            The size of batches of training data that are transferred to the GPU simultaneously. If OOM errors are encountered, it is recommended to reduce batch_size.
            Some discussion of batch size is in order. We find that for greatest training stability, it is best to have a batch_size larger than the maximum number of training examples
            that will be included in an active learning iteration-- so for starting at 1000 initial points, with 5000 points added at each iteration and 20 iterations executed, a batch_size > 101000 may be
            recommended. This will maximize training stability. However, if instead more training examples must be considered than can realistically fit into GPU memory, we recommend batch_size parameters of
            approximately the number of points added during each active learning iteration, to avoid different active learning iterations suddenly exhibiting radically different performance as the number of batches
            abruptly changes for the first time late into active learning. Experimentation regarding small batch size for different potentials is recommended.

        val_batch_size : int, optional
            The size of batches of validation data that are transferred to the GPU simultaneously. May be useful if small batches are used for training, but much larger batches
            can be accomodated during validation. If not specified, batch_size will be used to batch the validation data.

        save_interval : int, default=5
            Every save_interval active learning iterations, the function will save the BFBLearner object to the directory specified in filepath.

        stopping_cond : BFBrain.StoppingCondition object or subclass
            This object allows a user to specify conditions under which active learning should terminate before nstop iterations have been performed.
            See the BFBrain.StoppingCondition documentation for more details.

        prob_score_fn : {'BALD', 'MaxEntropy', 'variation_ratios', 'random', 'uncertainty', 'predictive_variance'} or callable
            Either a string which denotes which of the implemented strategies for scoring points to add to the training set,
            or a callable representing a custom function to perform this role. Any custom functions must have a signature 
            (tf.keras.model, tf.tensor(tf.float32, tf.float32))-> tf.tensor(tf.float32) or (tf.keras.model, tf.tensor(tf.float32, tf.float32), int)-> tf.tensor(tf.float32), depending on 
            if score_ntrials is specified or not. Unlike score_fn, this function is employed when weighting points to sample around to generate new training data, so that an algorithm might preferentially sample
            around points with high mutual information (with prob_score_fn = 'BALD'), but decide which points to add to the training set based on Shannon entropy (score_fn = 'MaxEntropy'), for example.
            Has no effect if L_probability_weighting is False.

        truth_label_fn : callable, optional
            If a callable, must take a 1-D NumPy array representing a single set of quartic coefficients and return a Boolean True if the potential they describe is bounded from below, False otherwise.
            If specified, training data will be labelled using this callable instead of the oracle in the DataManager object.

        score_ntrials : int, optional
            Many of the implemented score_fn methods ('BALD', 'MaxEntropy', 'variation_ratios', 'uncertainty', and 'predictive_variance') have an optional argument controlling the number of forward passes 
            through the network to compute their values. This parameter allows this value to be specified for score_fn's evaluations.

        prob_score_ntrials : int, optional
            Same as score_ntrials, but for prob_score_fn. Does not have any effect if prob_score_fn is not specified.

        reinitialize_weights : bool, default=True
            If True, randomize the neural network weights before each new round of active learning (unless the algorithm is not training on the full training data set-- see full_train_interval).
            The recommended setting is True, to prevent potential overfitting on the points sampled earlier and maintain the dropout approximation of the Bayesian neural network rigorously.

        rebalance_train_data : bool, default=False
            If True, the training set will be rebalanced every iteration by adding new bounded from below points to correct an overabundance of points which are not bounded from below.

        L_probability_weighting : bool, default=False
            If True, the selection of existing bounded-from-below training points to sample around for generating new points will be weighted based on prob_score_fn.

        plot_metrics : bool, default=False
            If True, all metrics will be plotted in the command line as well as saved to the output filepath when training completes.

        verbose : bool, default=False
            If True, print out information about the progress and performance of the active learning loop throughout the run to the console. 
            Otherwise data on the neural network performance after each iteration is still saved to a file 'output.txt' in the filepath directory.
        """
        # Determine how to select additional points to add to the training set based on the input of score_fn.
        if not callable(score_fn):
            try:
                score_fn = scoring_funcs[score_fn]
            except KeyError:
                raise Exception('score_fn must be a callable or one of the strings {}.'.format(list(scoring_funcs.keys())))

        if score_ntrials is not None:
            score_fn = tf.function(partial(score_fn, n_trials = score_ntrials), jit_compile = True)
        else:
            score_fn = tf.function(score_fn, jit_compile = True)
        if prob_score_fn is None:
            prob_score_fn = score_fn
        else:
            if not callable(prob_score_fn):
                try:
                    prob_score_fn = scoring_funcs[prob_score_fn]
                except KeyError:
                    raise Exception('score_fn must be a callable or one of the strings {}.'.format(list(scoring_funcs.keys())))
            if prob_score_ntrials is not None:
                prob_score_fn = tf.function(partial(prob_score_fn, n_trials = prob_score_ntrials), jit_compile = True)
            else:
                prob_score_fn = tf.function(prob_score_fn, jit_compile = True)

        # Set up a dictionary of the metrics so that the stopping condition can more easily find the one it's supposed to monitor.
        metrics_dict = {metric.name: metric for metric in self.metrics}
        # Set up a boolean flag which monitors if the active learning loop should stop early.
        should_stop = False

        if (stopping_cond is not None) and (not isinstance(stopping_cond, StoppingCondition)):
            raise Exception('stopping_cond must be either None, StoppingCondition, or a subclass of StoppingCondition.')

        #Set up Tensorflow datasets to greatly increase training speed.
        train_data = self.dm.create_dataset(self.data_train)
        train_data = train_data.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE).cache()

        if self.data_val is not None:
            val_data = self.dm.create_dataset(self.data_val, validation = True)
            if val_batch_size==-1:
                val_data = val_data.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE).cache()
            else:
                val_data = val_data.batch(val_batch_size, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE).cache()

        #Create a new directory in which to save the model checkpoint (and later the model) if one does not exist already.
        if(not os.path.isdir(filepath)):
            os.mkdir(filepath)

        #Create the output summary file here and save the model summary, unless the model has been loaded from a previous file.
        if(self.idx == 0 or (not os.path.isfile(filepath+'/output.txt'))):
            self._create_output_file(filepath)
            self._print_variables(filepath)

        #Set up a callback functions so that the neural network will stop after epoch_patience epochs without improvement on the validation set accuracy.
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=epoch_patience, monitor = 'loss', restore_best_weights = True, verbose=0)
        
        # If the neural network is totally untrained, perform the first training outside of the loop.
        if(self.idx == 0):
            # Take any metrics which should be recorded before any training.
            for metric in self.metrics:
                if metric.sc_type == 'model':
                    metric.record_score(self.model)
                if metric.sc_type == 'train':
                    if metric.check_initial == True:
                        metric.record_score(self.model, train_data)
            if(verbose):
                print('training the first epoch...')
            # Fit the model to the initial training data.
            history = self.model.fit(train_data, epochs=epoch_limit, callbacks=[early_stopping], verbose = 0)
            # history = self.model.fit(train_data, epochs=100*epoch_patience, verbose = 0)
            self.idx += 1
            if(verbose):
                print('trained!')
            # Record the performance of the classifier during its training loop and record any metrics based on performance over validation data.
            self.history.append_history(history)
            for metric in self.metrics:
                if metric.sc_type == 'val':
                    metric.record_score(self.model, val_data)
                if metric.sc_type == 'model':
                    metric.record_score(self.model)

        for ind in range(1, nstop+int(self.idx > 0)):
            #Start the active learning loop by generating a new set K of additional points to add to the model.
            if(verbose):
                print('Generating additional training set K...')
            # Generate the new labelled training data.

            K_data = self._generate_K_batched(K_batch_size, K_batch_num, K_factor, self.data_train.pos, score_fn, prob_score_fn, truth_label_fn, L_probability_weighting, verbose)
            #Add these points and their labels to the training sample
            self.data_train.append_data(K_data)
            # In the likely event that more negative points than positives have been selected in the active learning iteration, generate new positive points using the convexity of the bounded-from-below space
            # and append these points to the training data, if rebalance_training_data is True. A much larger set of candidate training points will be generated here, but as in the first acquisition iteration,
            # only the points score_fn evaluates as most interesting will be retained and added to the training set.
            if(rebalance_train_data):
                self._balance_with_positives(K_batch_size, K_factor, score_fn, verbose)

            # Record metrics that are based on new training data and those based on the pool of proposed candidate points.
            new_tensor = tf.constant(np.concatenate((K_data.pos, K_data.neg), axis = 0))
            new_labels = tf.concat((tf.ones(shape=(len(K_data.pos),)), tf.zeros(shape=(len(K_data.neg),))), axis = 0)
            for metric in self.metrics:
                if metric.sc_type == 'train':
                    metric.record_score(self.model, new_tensor, new_labels)
                if metric.sc_type == 'pool':
                    metric.record_score()
                # Check whether the stopping condition has been met
                if stopping_cond is not None:
                    if metric.name == stopping_cond.metric_name:
                        if stopping_cond(metrics_dict):
                            should_stop = True
            # If the stopping condition has been met, terminate the active learning loop.
            if should_stop:
                if verbose:
                    print('stopping condition satisfied. Exiting training loop early.')
                break

            #Every full_train_interval training steps, train on the full data set. Currently set by default to full_train_interval = 1.
            if(ind % full_train_interval == 0):
                self._update_regularizers(self.l_constant, self.data_train.n_elements(), reinitialize_weights)
                train_data = self.dm.create_dataset(self.data_train)
                # Determine a batch size that's as close to an even divisor of the number of training set elements as possible. 
                # This prevents a sudden severe degredation in classifier performance if one batch happens to be a highly nonrepresentative small sample of the total training data.
                if self.data_train.n_elements() > batch_size:
                    new_batch_size = self.data_train.n_elements() // ((self.data_train.n_elements()//batch_size) + 1)
                    train_data = train_data.batch(new_batch_size, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE).cache()
                else:
                    train_data = train_data.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE).cache()
                if(verbose):
                    print('training round ' + str(self.idx) + '...')
                history = self.model.fit(train_data, epochs=200*epoch_limit, callbacks=[early_stopping], verbose=0)
                # history = self.model.fit(train_data, epochs=100*epoch_patience, verbose=0)
            #Otherwise train on just the new data
            else:
                self._update_regularizers(self.l_constant, (K_data.n_elements()), False)
                train_data = self.dm.create_dataset(K_data)
                train_data = train_data.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE).cache()
                if(verbose):
                    print('training round ' + str(self.idx) + '...')
                # Train on only a small number of epochs here, to prevent overfitting to highly uncertain new data.
                history = self.model.fit(train_data, epochs=10, callbacks=[early_stopping], verbose=0)
            
            # Record the training history.
            self.history.append_history(history)

            # Record the validation and model-based metrics.
            if(verbose):
                print('validating performance of the model...')
            for metric in self.metrics:
                if metric.sc_type == 'val':
                    metric.record_score(self.model, val_data)
                if metric.sc_type == 'model':
                    metric.record_score(self.model)
                if stopping_cond is not None:
                    if metric.name == stopping_cond.metric_name:
                        if stopping_cond(metrics_dict):
                            should_stop = True
            if should_stop:
                if verbose:
                    print('stopping condition satisfied. Exiting training loop early.')
                break
                    
            # Print out the full set of metrics for this round to the output text file, plus the console if verbose is True.
            self._get_status(self.idx, filepath, verbose=verbose)

            # Every save_interval iterations, save the active learning state.
            if(self.idx % save_interval == 0):
                #Every save_interval iterations, Save the critical data for the project (the training data, testing data, model, and rng states) so that training can be resumed later.
                self.save_AL_state(filepath)

            self.idx += 1

        # If the loop terminated early due to a stopping condition, make sure that the status_history was recorded in output.txt here.
        if should_stop:
            self._get_status(self.idx, filepath, verbose = verbose)

        #Save the critical data for the project (the training data, testing data, model, and rng states) so that training can be resumed later or the model can be exported.
        self.save_AL_state(filepath)

        # Plot the results-- in verbose mode also draw these plots in the console, if possible.
        self.history.plot_history(filepath)
        self.plot_metrics(filepath)
        if verbose and plot_metrics:
            self.history.plot_history()
            self.plot_metrics()

        return
