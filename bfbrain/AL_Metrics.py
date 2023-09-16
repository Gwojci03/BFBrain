"""This module contains code for various performance metrics which BFBrain can track over the course of active learning.
"""

from abc import ABC, abstractmethod
from os import sys
from bfbrain.Score_Functions import QBDC, Max_Entropy, BALD, Variation_Ratios, Random_AL, Predictive_Variance, MC_call_fast
from bfbrain.False_Proximity_Test import combined_false_score
import tensorflow as tf
import numpy as np
from scipy.special import lambertw
import matplotlib.pyplot as plt
from functools import partial

scoring_funcs = {'BALD':BALD, 'uncertainty':QBDC, 'random':Random_AL, 'MaxEntropy':Max_Entropy, 'variation_ratios':Variation_Ratios, 'predictive_variance':Predictive_Variance}

valid_sc_types = ['val', 'train', 'pool', 'model']

metric_reductions = {'mean':np.mean, 'max':np.max, 'min':np.min, 'std': np.std}

def process_score_fn(score_fn, name):
    """A utility function which translates a string specifying one of the predefined acquisition scoring functions into
    the corresponding numerical method.

    Parameters
    ----------
    score_fn : {'BALD', 'uncertainty', 'random', 'MaxEntropy', 'variation_ratios', 'predictive_variance'} or callable.
        If this function is a callable, it must have the signature (tf.keras.model, tf.tensor(tf.float32, tf.float32))-> tf.tensor(tf.float32)
        or (tf.keras.model, tf.tensor(tf.float32, tf.float32), int)-> tf.tensor(tf.float32)

    name : str, optional
        If specified, this name is returned unaltered. Otherwise, a name will be automatically generated based on score_fn.

    Returns
    -------
    callable
        A valid score_fn to be used in various performance metrics.

    str
        A string which will be used to generate a name for an ALMetric object.
    """
    if callable(score_fn):
        if name is None:
            try:
                name = list(scoring_funcs.keys())[list(scoring_funcs.values()).index(score_fn)]
            except ValueError as error:
                name = 'score'
    else:
        if not isinstance(score_fn, str):
            raise Exception('score_fn must be a string which acts as a key in the dict scoring_funcs in score_functions.py, or a callable.')
        if name is None:
            name = score_fn
        try:
            score_fn = scoring_funcs[score_fn]
        except KeyError as error:
            raise Exception('score_fn was a string, but was not recognized as corresponding to a known metric. Valid string inputs are {}'.format(list(scoring_funcs.keys())))
    return score_fn, name

def _get_reduction(red_name):
    """A utility method to connect a string or list of strings specifying certain reductions of a 1-D NumPy array to the corresponding functions.

    Parameters
    ----------
    red_name : {'mean', 'max', 'min', 'std'} or list of these strings.

    Returns
    -------
    list of callables
        A list of callables (possibly of length 1) which correspond to the reduction(s) named in red_name

    list of str
        red_name. If red_name was a single str object, it is returned as a list of str with length 1.
    """
    try:
        if isinstance(red_name, str):
            reduction = [metric_reductions[red_name]]
            red_name = [red_name]
        else:
            reduction = [metric_reductions[red] for red in red_name]
        return reduction, red_name
    except KeyError as error:
        raise Exception('Unrecognized value for argument "reduction". Must be one of {} or a list of those values.'.format(list(metric_reductions.keys())))

def _check_reduction(reduction, red_name):
    """A utility function which checks that an input string is in a list of strings (specifically keys in metric_reductions), and
    raises an error if it's not.

    Parameters
    ----------
    reduction : str
    red_name : list of str
    """
    if reduction not in red_name:
            raise Exception('Please specify a reduction that the metric has recorded. Options are {}'.format(red_name))


class ALMetric(ABC):
    """A generic abstract class for computing and recording performance metrics for active learning. All performance metrics in BFBrain inherit from this class.

    Attributes
    ----------
    status_history : list
        A list which contains a record, for each round of active learning, for whichever
        metric the subclass will measure. The entries of status_history may be, depending on the
        subclass, virtually any kind of data or data structure, as long as the elements are picklable.

    sc_type : {'val', 'train', 'pool', 'model'}
        A string which denotes what type of metric the ALMetric object is, since different metrics are recorded at different
        points in the active learning loop. If sc_type is 'val', this metric is computed using a validation data set immediately after each active learning round completes.
        If sc_type is 'train', this metric is computed immediately after new training data is generated in the active learning loop, but before the neural network's weights are
        reset and training commences. It is evaluated using the newly-generated training data. If sc_type is 'pool', this metric is computed using the pool of candidate points
        from which new training samples are drawn at each iteration. It is computed immediately after the new training data is selected from the pool. If sc_type is model, the
        a metric is computed without reference to any data set (validation, training, or pool) present in the active learning loop, at the end of each active learning iteration. 
        The only implemented metrics which have sc_type 'model' measure predictive stability on some specified unlabelled set of points, namely UnlabelledAgreement and UnlabelledDeltaFScore,
        but the possibility remains that different sorts of metrics in this class, for example the one based on error stability computed directly from the neural network weights
        discussed in arXiv:2104.01836, may be desirable for a user to implement.

    name : str
        A string which denotes a name for this metric. In a list of metrics passed to a BFBLearner class, the names of each member of the list should be unique.

    Parameters
    ----------
    sc_type : {'val', 'train', 'pool', 'model'}

    name : str
    """
    def __init__(self, sc_type, name):
        self.status_history = []
        if sc_type not in valid_sc_types:
            raise Exception('sc_type must be a string, and must be one of {}'.format(valid_sc_types))
        self.sc_type = sc_type
        self.name = sc_type + '_' + name

    def record_score(self, *args):
        """Appends the latest value for the performance metric to the status_history object.
        This method calls an abstract method "performance_check" which will turn whatever input is specified in the method into the metric
        the object is supposed to track. The method performance_check, and therefore the arguments going into this method, will
        vary depending on the specific subclass of ALMetric.
        """
        self.status_history.append(self.performance_check(*args))
        return self.status_history[-1]
    
    def print_status(self, file = sys.stdout):
        """A method which prints the last entry in status_history to a file (or the console). Uses
        the method perf_message (which is often overwritten in the child class) to identify the 
        metric being printed and separates status_history elements that are tuples into different printout lines, for clarity. 
        """
        last_status = self.status_history[-1]
        out_message = self.perf_message()
        if type(self.status_history[-1]) == tuple:
            for i, stat in enumerate(last_status):
                print(out_message[i], file = file)
                print(stat, file = file)
        else:
            print(out_message, file = file)
            print(last_status, file = file)

    # A method which prints out the message 
    def perf_message(self):
        """A method which prints out a message that is helpful in identifying what metric is being reported when a user calls print_status.
        Often overwritten in a child class.
        """
        return self.name + ':'

    @abstractmethod
    def performance_check(self, *args):
        """An abstract class which takes some arguments (depending on the type of performance metric) and computes the quantity or quantities that the
        performance metric is supposed to track. This method is called by record_score and its results are appended to the status_history attribute.
        """
        pass
    
    def get_metric(self, *args):
        """A function which reduces the status_history object to a list of single numbers (usually some sort of figure of merit) in the event that the members of status_history are a list or a tuple.
        By default, it simply returns the full status_history list and must be overwritten in subclasses which have lists or tuples as entries in status_history.

        Parameters
        ----------
        *args : Any
            Some overwritten versions of this class can accept optional arguments, although the method does not in the parent class.

        Returns
        -------
        A NumPy array featuring information from status_history for plotting.
        """
        return np.transpose(np.array([self.status_history]))
    
    def reset_data(self):
        """
        A function which resets the metric data entirely. In some subclasses, this must be overloaded to properly reset the class.
        """
        self.status_history = []

    def get_legend(self, *args):
        """Returns a legend for a plot of the metric given by plot_metric.
        Often must be overwritten in subclasses.

        Parameters
        ----------
        *args : Any
            Some overwritten versions of this class can accept optional arguments, although the method does not in the parent class. Must take the same arguments as get_metric.

        Returns
        -------
        list of strings
            A list of strings which are usable to specify a legend in matplotlib.
        """
        return [self.name]
    
    def plot_metric(self, filepath = None, **kwargs):
        """Plots the performance metric as a function of the number of active learning iterations.

        Parameters
        ----------
        filepath : str, optional
            If this argument is specified, then the plot of the metric will be saved as a .png file in the directory with the name given by filepath.

        **kwargs : dict, optional
            Many subclasses of ALMetric have get_metric and get_legend methods which take some keyword arguments-- these can be specified when calling plot_metric.
        """
        metrics = self.get_metric(**kwargs)
        legend = self.get_legend(**kwargs)
        plt.figure()
        plt.plot(metrics)
        plt.legend(legend)
        plt.xlabel("AL Iterations")
        plt.ylabel(self.name)
        if(filepath is None):
            plt.show()
        else:
            plt.savefig(filepath+ '/' + self.name + '.png')
        plt.close()

class UnlabelledPreds(ALMetric):
    """An abstract class for handling metrics which go by the predictions of the model on some unlabelled set of quartic coefficients.

    Attributes
    ----------
    lams : np.array(np.float32, np.float32)
        A 2-D NumPy array representing sets of quartic potential coefficients. This will be an unlabelled set of points the model will make predictions on.

    ds : tf.data.Dataset
        A Tensorflow dataset generated from lams.

    batch_size : int, default=200000
        The maximum size of batches of lams that will be transferred to the GPU and computed with at one time.

    name : str
        The unique identifier for the metric in the list of metrics traced by BFBLearner.
    Parameters
    ----------
    lams : np.array(np.float32, np.float32)
        A 2-D NumPy array representing sets of quartic potential coefficients. This will be an unlabelled set of points the model will make predictions on.

    name : str
        The name will provide a unique identifier for the metric in the list of metrics tracked by BFBLearner-- this identifier will be 'model_'+name.

    batch_size : int, default=200000
        The maximum size of batches of lams that will be transferred to the GPU and computed with at one time.
    """
    def __init__(self, lams, name, batch_size = 200000):
        super().__init__(sc_type = 'model', name = name)
        self.lams = lams
        self.batch_size = 200000
        self.ds = tf.data.Dataset.from_tensor_slices(lams).batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE).cache()
    
    @abstractmethod
    def performance_check(self, model):
        pass

    def __getstate__(self):
        """Used to pickle the metric.
        """
        state = self.__dict__.copy()
        del state['ds']
        return state
    
    def __setstate__(self, state):
        """Used to unpickle the metric.
        """
        self.__dict__.update(state)
        self.ds = tf.data.Dataset.from_tensor_slices(self.lams).batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE).cache()
    
class UnlabelledAgreement(UnlabelledPreds):
    """A metric which computes agreement (Cohen's kappa) among the model between successive iterations of active learning on a specified set of unlabelled points.

    Attributes
    ----------
    status_history : list of floats
        The entries of this status_history object will be Cohen's kappa between successive iterations of active learning on lams.

    old_preds : np.array(np.float32)
        The previous model's predictions on lams. Preserved to compare to the current model.

    lams : np.array(np.float32, np.float32)
        A 2-D NumPy array representing sets of quartic potential coefficients. This will be an unlabelled set of points the model will make predictions on.

    ds : tf.data.Dataset
        A Tensorflow dataset generated from lams.

    batch_size : int, default=200000
        The maximum size of batches of lams that will be transferred to the GPU and computed with at one time.

    name : str
        The unique identifier for the metric in the list of metrics traced by BFBLearner. By default this will be 'model_agreement'

    n_trials : int, default=100
        The number of forward passes through the network to get the predictions from Monte Carlo dropout.

    Parameters
    ----------
    lams : np.array(np.float32, np.float32)
        A 2-D NumPy array representing sets of quartic potential coefficients. This will be an unlabelled set of points the model will make predictions on.

    name : str, default='agreement'
        The name will provide a unique identifier for the metric in the list of metrics tracked by BFBLearner-- this identifier will be 'model_'+name.

    batch_size : int, default=200000
        The maximum size of batches of lams that will be transferred to the GPU and computed with at one time.

    n_trials : int, default=100
        The number of forward passes through the network to get the predictions from Monte Carlo dropout.
    """
    def __init__(self, lams, name = 'agreement', batch_size = 200000, n_trials = 100):
        super().__init__(lams, name, batch_size, n_trials)
        self.n_trials = n_trials
        self.old_preds = None

    def performance_check(self, model):
        """Computes Cohen's kappa for the classifier between successive active learning iterations, on the unlabelled quartic coefficients lams.
        If there is no previous model (i.e., no active learning has been done), returns 0. and saves the current model's predictions as old_preds.
        """
        if self.old_preds is None:
            out_preds = []
            for x in self.ds:
                out_preds.append(tf.reshape(MC_call_fast(model, x, self.n_trials), shape = [-1]).numpy())
            out_preds = np.concatenate(out_preds, axis = 0)
            self.old_preds = out_preds
            return 0.

        out_preds = []
        for x in self.ds:
            out_preds.append(tf.reshape(MC_call_fast(model, x, self.n_trials), shape = [-1]).numpy())
        out_preds = np.concatenate(out_preds, axis = 0)
        Ao = np.count_nonzero(np.logical_or(np.logical_and(out_preds >= 0.5, self.old_preds >=0.5), np.logical_and(out_preds < 0.5, self.old_preds < 0.5))) / len(out_preds)
        p_pos_new = np.count_nonzero(out_preds >= 0.5)/len(out_preds)
        p_pos_old = np.count_nonzero(self.old_preds >= 0.5)/len(self.old_preds)
        Ae = 2.*p_pos_new*p_pos_old - p_pos_new - p_pos_old + 1.
        self.old_preds = out_preds
        return (Ao -Ae) / (1. - Ae)
    
    def get_metric(self):
        """Simply returns Cohen's kappa as a function of the number of active learning iterations.
        """
        return np.transpose(np.array([[stat for stat in self.status_history][1:]]))
    
    def reset_data(self):
        """Resets the data in the metric.
        """
        super().reset_data()
        self.old_preds = None

class UnlabelledDeltaF(UnlabelledPreds):
    """A metric which computes the estimated change in F score on a specified unlabelled set of points
    for the model between successive iterations of active learning on a specified set of unlabelled points,
    based on the methodology of arXiv:cs/1901.09118.

    Attributes
    ----------
    status_history : list of floats
        The entries of this status_history object will be the estimated change in F score between successive iterations of active learning on lams.

    old_preds : np.array(np.float32)
        The previous model's predictions on lams. Preserved to compare to the current model.

    lams : np.array(np.float32, np.float32)
        A 2-D NumPy array representing sets of quartic potential coefficients. This will be an unlabelled set of points the model will make predictions on.

    ds : tf.data.Dataset
        A Tensorflow dataset generated from lams.

    batch_size : int, default=200000
        The maximum size of batches of lams that will be transferred to the GPU and computed with at one time.

    name : str
        The unique identifier for the metric in the list of metrics traced by BFBLearner. By default this will be 'model_delta_F'

    n_trials : int, default=100
        The number of forward passes through the network to get the predictions from Monte Carlo dropout.

    Parameters
    ----------
    lams : np.array(np.float32, np.float32)
        A 2-D NumPy array representing sets of quartic potential coefficients. This will be an unlabelled set of points the model will make predictions on.

    name : str, default='delta_F'
        The name will provide a unique identifier for the metric in the list of metrics tracked by BFBLearner-- this identifier will be 'model_'+name.

    batch_size : int, default=200000
        The maximum size of batches of lams that will be transferred to the GPU and computed with at one time.

    n_trials : int, default=100
        The number of forward passes through the network to get the predictions from Monte Carlo dropout.
    """
    def __init__(self, lams, name = 'delta_F', batch_size = 200000, n_trials = 100):
        super().__init__(lams, name, batch_size)
        self.old_preds = None
        self.n_trials = n_trials

    def performance_check(self, model):
        """Computes the estimated change in F score for the classifier between successive active learning iterations, on the unlabelled quartic coefficients lams.
        If there is no previous model (i.e., no active learning has been done), returns np.inf and saves the current model's predictions as old_preds.
        """
        if self.old_preds is None:
            out_preds = []
            for x in self.ds:
                out_preds.append(tf.reshape(MC_call_fast(model, x, self.n_trials), shape = [-1]).numpy())
            out_preds = np.concatenate(out_preds, axis = 0)
            self.old_preds = out_preds
            return np.inf

        out_preds = []
        for x in self.ds:
            out_preds.append(tf.reshape(MC_call_fast(model, x, self.n_trials), shape = [-1]).numpy())
        out_preds = np.concatenate(out_preds, axis = 0)
        a = np.count_nonzero(np.logical_and(out_preds >= 0.5, self.old_preds >= 0.5))
        b = np.count_nonzero(np.logical_and(out_preds < 0.5, self.old_preds >=0.5))
        c = np.count_nonzero(np.logical_and(out_preds >= 0.5, self.old_preds < 0.5))
        self.old_preds = out_preds
        return 1. - ((2*a)/(2*a + b + c))
    
    def get_metric(self):
        """Simply returns the change in F score as a function of the number of active learning iterations.
        """
        return np.transpose(np.array([[stat for stat in self.status_history][1:]]))
    
    def reset_data(self):
        """Resets the data in the metric.
        """
        super().reset_data()
        self.old_preds = None

class ValidationMetric(ALMetric):
    """An abstract class for handling metrics which measure performance of the model on a validation set. This class exists primarily to specify that any validation-set-based performance metrics must 
    have their performance_check method take the inputs (tf.keras.Model, tf.data.Dataset)
    """
    def __init__(self, name):
        super().__init__(sc_type = 'val', name = name)
    
    @abstractmethod
    def performance_check(self, model, ds):
        pass

# An abstract class for handling metrics which measure performance of the model on a validation set. This class exists primarily to specify that any validation-set-based performance metrics must have their performance_check
# method take the inputs (tf.keras.Model, tf.Tensor)
class TrainMetric(ALMetric):
    """An abstract class for handling metrics which measure predictions of the model on newly-added training data. This class exists primarily to specify that any 
    training-set-based performance metrics must have their performance_check
    method take the inputs (tf.keras.Model, tf.Tensor)
    """
    def __init__(self, name, check_initial = False):
        super().__init__(sc_type = 'train', name = name)
        self.check_initial = check_initial
    
    @abstractmethod
    def performance_check(self, model, lams, labels):
        pass

class PoolMetric(ALMetric):
    """An abstract class for handling metrics which measure predictions of the model on the pools of candidate points from which new training data is drawn.
    This class exists primarily to ensure that these metrics have additional abstract methods which much be specified to implement a class of this sort.

    Attributes
    ----------
    batch_scores : list
        Because the pool of candidate points are generated in discrete manageable batches, the metrics are computed over each individual batch and then combined,
        the precise manner of which depends on the specific metric in question. However, all pool metrics must have this attribute to act as temporary storage of
        the individual batch results before they can be combined.
    """
    def __init__(self, name):
        super().__init__(sc_type = 'pool', name = name)
        self.batch_scores = []
    
    @abstractmethod
    def record_batch(self, *args):
        """An abstract method which will record an individual batch's results to batch_scores, in a manner that must be 
        specified in a subclass.
        """
        pass
    
    @abstractmethod
    def record_score(self):
        """The concrete record_score method of the ALMetric class must be overwritten with an abstract version,
        which must in turn be specified in subclasses.
        """
        pass



class PoolMetricReduction(PoolMetric):
    """An abstract class which inherits from PoolMetric, but features methods to automatically take the mean/min/max of the scores determined in record_batch.
    This is an abstract class which shouldn't be instantiated directly, but allows for rapid prototyping of a variety of pool metrics.

    Attributes
    ----------
    reduction : a list containing elements of {np.mean, np.min, np.max}
        This is the reduction that is performed on individual batches, and finally, among the results for all batches, to produce the entries in status_history.
        If a list of reductions are applied, then the elements of status_history will be lists with each element being a different reduction being applied to the pool score.
    red_name : a list containing elements of {'mean', 'min', 'max'}
        This list of strings will contain the same information as reduction, but is used for labelling purposes.

    Parameters
    ----------
    name : str
        A unique identifier for the metric in the list of metrics in a BFBLearner object.
    red : {'mean', 'min', 'max'} or a list of those values
        This argument specifies the reduction that is performed on individual batches, and finally, among the results for all batches, to produce the entries in status_history.
        If a list of reductions are applied, then the elements of status_history will be lists with each element being a different reduction being applied to the pool score.
    """
    def __init__(self, name, red = ['mean', 'min', 'max']):
        self.reduction, self.red_name = _get_reduction(red)
        if 'std' in red:
            raise Exception('Do not request standard deviation for scores found from the pool of candidate points-- insufficient data is stored to compute this quantity.')
        super().__init__(name = name)
        self.batch_scores = []
    
    def record_batch(self, *args):
        """Records a score generated by performance_check (which must be specified in a subclass) for an individual batch in the pool of candidate points
        to the batch_scores list.
        """
        self.batch_scores.append(self.performance_check(*args))

    def record_score(self):
        """Combines the metrics computed for each batch into a single status_history entry, and append that entry to status_history.
        """
        self.status_history.append([self.reduction[i]([b_score[i] for b_score in self.batch_scores]) for i in range(len(self.reduction))])
        self.batch_scores = []
        return self.status_history[-1]
    
    def perf_message(self):
        """The perf_message method labels any printed output of the metric.
        """
        return self.name + ' (unlabelled pool) {}:'.format(list(self.red_name))

    def get_metric(self, reduction = None):
        """In this subclass, get_metric can take a keyword argument.

        Parameters
        ----------
        reduction : str in self.red_name, optional
            If specified, get_metric will only return the values corresponding to the specified reduction. If not specified, get_metric will return the status_history object in its entirety.

        Returns
        -------
        np.array
            Represents some plottable set of values from status_history.
        """
        if reduction is not None:
            _check_reduction(reduction, self.red_name)
            return np.transpose(np.array([[stat[self.red_name.index(reduction)] for stat in self.status_history]]))
        else:
            return np.array(self.status_history)
    
    def reset_data(self):
        """Reset the data in status_history.
        """
        super().reset_data()
        self.batch_scores = []
    
    def get_legend(self, reduction = None):
        """In this subclass, get_legend can take a keyword argument.

        Parameters
        ----------
        reduction : str in self.red_name, optional
            get_legend will return a legend consistent with the get_metric result with the same reduction argument passed.

        Returns
        -------
        list of str
            An argument to specify a legend in matplotlib.
        """
        if reduction is not None:
            return [reduction + '_' + self.name]
        else:
            return [red + '_' + self.name for red in self.red_name]

# A metric for keeping track of the estimated change in F score for successive iterations of the active learning algorithm on unlabelled data, based on arXiv:cs/1901.09118
class PoolDeltaF(PoolMetric):
    """A metric for keeping track of the estimated change in F score for successive iterations of the active learning algorithm on unlabelled data, based on arXiv:cs/1901.09118.
    Each time active learning produces a new unlabelled pool of candidate points to draw training examples from, this metric computes the model's predicted labels for all of these points, and
    stores both the pool of points and the predictions. Then, after another round of active learning, the metric computes the NEW model's predicted labels on the stored pool of points.
    Then, the two sets of predictions are compared and the estimated change in F score over the pool distribution is computed from the level of agreement between the two sets of predictions,
    following the procedure outlined in arXiv:cs/1901.09118. Predictions are based on Monte Carlo dropout with 100 forward passes through the neural network.

    Attributes
    ----------
    status_history : list of floats
        A list which contains a record, for each round of active learning, of the estimated change in F score.

    name : str, default='delta_F'
        A string which denotes a name for this metric. In a list of metrics passed to a BFBLearner class, the names of each member of the list should be unique.

    old_pool : list of np.array(np.float32, np.float32)
        A list of 2-D NumPy arrays, each of which represents a batch of points generated as part of the pool of candidate points in active learning. This array stores
        the points that made up the PREVIOUS round's pool of points, so that the current model can make predictions on them.

    new_pool : list of np.array(np.float32, np.float32)
        A list of 2-D NumPy arrays, each of which represents a batch of points generated as part of the pool of candidate points in active learning. This array stores
        the points that made up the CURRENT round's pool of points, so that the current model can make predictions on them. After recording the status_history value for this metric,
        new_pool's values are transferred to old_pool, and then new_pool is cleared.

    old_preds : list of np.array(np.float32)
        A list to contain all of the PREVIOUS model's predictions on old_pool.

    new_preds : list of np.array(np.float32)
        A list to contain all of the CURRENT model's predictions on old_pool.

    newer_preds : list of np.array(np.float32)
        A list to contain all of the CURRENT model's predictions on new_pool. After recording the status_history value for this metric,
        newer_preds values are transferred to old_preds, and then newer_preds and new_preds are both cleared.

    old_pool_iter : iter
        An iterator over old_pool. Replaced whenever old_pool is overwritten.
    """
    def __init__(self, name = 'delta_F'):
        super().__init__(name = name)
        # Predictions of the last model on old_pool
        self.old_preds = []
        # Predictions of the current model on old_pool
        self.new_preds = []
        # Predictions of the current model on new_pool
        self.newer_preds = []

        self.old_pool = []
        self.new_pool = []
        self.old_pool_iter = iter(self.old_pool)
    
    def record_batch(self, model, L):
        """Records newly-generated pool points and the current model's prediction on them. After the first active learning iteration, 
        also records the current model's predictions on the corresponding element of old_pool, since after the first active learning iteration the old_pool and new_pool
        will always have the same number of elements.

        Parameters
        ----------
        model : tf.keras.model
            The current Tensorflow model of a BFBLearner object.

        L : np.array(np.float32, np.float32)
            A 2-D NumPy array representing a batch of pool points proposed to the neural network as possible training points.
        """
        scores, pool = self.performance_check(model, L)
        if len(self.old_pool) > 0:
            old_scores = tf.reshape(MC_call_fast(model, next(self.old_pool_iter), 1000),shape = [-1]).numpy()
            self.new_preds.append(old_scores)
        self.newer_preds.append(scores)
        self.new_pool.append(pool)
    
    def performance_check(self, model, L):
        """Computes the predictions of the neural network on an input batch of pool points, and returns a tuple of the predictions and the pool points.

        Parameters
        ----------
        model : tf.keras.model
            The current Tensorflow model of a BFBLearner object.

        L : np.array(np.float32, np.float32)
            A 2-D NumPy array representing a batch of pool points proposed to the neural network as possible training points.

        Returns
        -------
        np.array(np.float32)
            A 1-D NumPy array of model predictions on L

        np.array(np.float32, np.float32)
            The array L
        """
        return tf.reshape(MC_call_fast(model, L, 1000), shape = [-1]).numpy(), L.numpy()

    def record_score(self):
        """Combines the predictions made on individual batches in order to produce an estimate of the change in F score on old_pool,
        and appends this estimate onto status_history. Then, overwrites old_pool with new_pool, old_preds with newer_preds, and then resets
        new_pool, new_preds, and newer_preds.

        Returns
        -------
        float
            The last element of status_history.
        """
        if len(self.newer_preds) == len(self.new_preds):
            a, b, c = 0, 0, 0
            for i in range(len(self.new_preds)):
                a += np.count_nonzero(np.logical_and(self.new_preds[i] >= 0.5, self.old_preds[i] >= 0.5))
                b += np.count_nonzero(np.logical_and(self.new_preds[i] < 0.5, self.old_preds[i] >=0.5))
                c += np.count_nonzero(np.logical_and(self.new_preds[i] >= 0.5, self.old_preds[i] < 0.5))
            self.status_history.append(1. - ((2*a)/(2*a + b + c)))
        else:
            self.status_history.append(np.inf)
        self.old_preds = self.newer_preds
        self.newer_preds = []
        self.old_pool = self.new_pool
        self.new_pool = []
        self.new_preds = []
        self.old_pool_iter = iter(self.old_pool)
        return self.status_history[-1]
    
    def reset_data(self):
        """Clears all data in the metric.
        """
        super().reset_data()
        # Predictions of the last model on old_pool
        self.old_preds = []
        # Predictions of the current model on old_pool
        self.new_preds = []
        # Predictions of the current model on new_pool
        self.newer_preds = []

        self.old_pool = []
        self.new_pool = []
        self.old_pool_iter = iter(self.old_pool)


# A metric that simply records the model.evaluate() method on a Tensorflow dataset. Generally used to check the accuracy of the model on a validation set.
class ModelEvaluation(ValidationMetric):
    """A metric that simply records the model.evaluate() method on a labelled Tensorflow dataset (the validation set in our context). BFBrain's evaluate method keeps
    track of the model accuracy, false positives, and false negatives via evaluate(). Note that this method does NOT use Monte Carlo dropout to compute these quantities,
    but instead approximates the mean of Monte Carlo dropout via a single pass through the network with no dropout (and all model weights divided by 1 - <dropout probability>).

    Attributes
    ----------
    status_history : list of lists of np.float32
        The entries of status_history here will be lists of the form [<binary accuracy>, <false positives>, <false negatives>], evaluated over the validation set.

    name : str
        The unique identifier for the metric in the list of metrics traced by BFBLearner. By default, this will be 'val_accuracy'.

    Parameters
    ----------
    name : str, default='accuracy'
        The name will provide a unique identifier for the metric in the list of metrics tracked by BFBLearner-- this identifier will be 'val_'+name.
    """
    def __init__(self, name = 'accuracy'):
        super().__init__(name)

    def performance_check(self, model, ds):
        """This performance_check simply calls Tensorflow's model.evaluate() method, and ignores the first term (the loss)
        """
        return model.evaluate(ds, verbose = 0)[1:]
    
    def perf_message(self):
        """The perf_message method labels any printed output of the metric.
        """
        return 'Validation accuracy [accuracy, false positives, false negatives]:'
    
    def get_metric(self):
        """For plotting, this metric simply requests the binary accuracy over the active learning iterations.
        """
        return np.transpose(np.array([[stat[0] for stat in self.status_history][1:]]))

# A metric that records the binary accuracy, false positives, and false negatives evaluated with Monte Carlo dropout on a Tensorflow dataset.
class MCModelEvaluation(ValidationMetric):
    """A metric that records the same data as ModelEvaluation on a labelled Tensorflow dataset (the validation set in our context),
    but using Monte Carlo dropout with 100 forward passes through the neural network. Otherwise functions identically to ModelEvaluation.

    Attributes
    ----------
    name : str
        The unique identifier for the metric in the list of metrics traced by BFBLearner. By default, this will be 'val_MC_accuracy'.

    Parameters
    ----------
    name : str, default='MC_accuracy'
        The name will provide a unique identifier for the metric in the list of metrics tracked by BFBLearner-- this identifier will be 'val_'+name.
    """
    def __init__(self, name = 'MC_accuracy'):
        super().__init__(name)

    def performance_check(self, model, ds):
        """This performance_check finds the binary accuracy, false positives, and false negatives over a validation data set.
        """
        false_pos = 0
        false_neg = 0
        total = 0
        for x, y in ds:
            total += tf.shape(y)[0]
            result = MC_call_fast(model, x, 100)
            false_pos = false_pos + tf.math.count_nonzero(tf.logical_and(result >= 0.5, ~y))
            false_neg = false_neg + tf.math.count_nonzero(tf.logical_and(result < 0.5, y))
        return [1.-(tf.cast(false_pos + false_neg, tf.float32) / tf.cast(total, tf.float32)).numpy(), false_pos.numpy(), false_neg.numpy()]
    
    def perf_message(self):
        """The perf_message method labels any printed output of the metric.
        """
        return 'MC validation accuracy [accuracy, false positives, false negatives]:'
    
    def get_metric(self):
        return np.transpose(np.array([[stat[0] for stat in self.status_history][1:]]))

class DecisionBoundaryScore(ValidationMetric):
    """A metric which records a "decision boundary score"-- for each point that the (non-MC-dropout) neural network classifies incorrectly in a validation set,
    this method uses gradient ascent/descent to determine the angular distance on the hypersphere of quartic coeffecients to the decision boundary.
    Reports the results of the mean, standard deviation, and max of these scores in radians for both false positives and false negatives,
    as well as the number of points in both groups which exceed some input number of radians distance from the decision boundary.
    This metric can be extremely computationally intensive, and generally can reflect the deterministic forward pass's tendency to
    occasionally be incorrect and very overconfident. However, if a user is insistent on only using a single
    forward pass of the neural network to evaluate a network, this method enables them to be somewhat confident that
    any points that are mislabelled will be close in parameter space to points which are correctly labelled.

    Attributes
    ----------
    status_history : list of tuples of lists of np.float32
        The entries of status_history here will contain a tuple of two lists of the form [<mean>, <std>, <max>, # > tol_dist radians], the first for
        false positives and the second for false negatives. The mean, standard deviation, and max values are computed from the angular distance (in radians) of the incorrectly classified points
        to points along the decision boundary. The final entry is the number of points for which this distance is greater than some user-specified cutoff, tol_dist.

    name : str
        The unique identifier for the metric in the list of metrics traced by BFBLearner. By default, this will be 'val_combined_false_score'.

    tol_dist : float
        The maximum angular distance of an incorrectly-classified point to the decision boundary that the user considers acceptable. For small (O(0.01)) values of this angle,
        it roughly corresponds to the fractional degree of correction of the quartic coefficients to reach the decision boundary-- so an angular deformation of 0.01 represents approximately
        a 1% correction to the quartic coupling coefficients.

    Parameters
    ----------
    tol_dist : float
        The maximum angular distance of an incorrectly-classified point to the decision boundary that the user considers acceptable. For small (O(0.01)) values of this angle,
        it roughly corresponds to the fractional degree of correction of the quartic coefficients to reach the decision boundary-- so an angular deformation of 0.01 represents approximately
        a 1% correction to the quartic coupling coefficients.

    name : str, default='combined_false_score'
        The name will provide a unique identifier for the metric in the list of metrics tracked by BFBLearner-- this identifier will be 'val_'+name.
    """
    def __init__(self, tol_dist, name = 'combined_false_score'):
        super().__init__(name)
        self.tol_dist = tol_dist
    
    def performance_check(self, model, ds):
        """Computes the angular distances between incorrectly classified points and the decision boundary.
        """
        bin_acc = model.evaluate(ds, verbose = 0)[1]
        if bin_acc < 0.99:
            return ([np.inf, np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf, np.inf])
        else:
            return (combined_false_score(model, ds, tf.constant(self.tol_dist)))
    
    def perf_message(self):
        """The perf_message method labels any printed output of the metric.
        """
        return ('false positive score [mean, std, max, # > {}]:'.format(self.tol_dist), 'false negative score [mean, std, max, # > {}]:'.format(self.tol_dist))
    
    def get_metric(self):
        """Returns the number of false positives and false negatives greater than tol_dist radians from the decision boundary (tracked separately) for plotting over the course of active learning.
        """
        return np.transpose(np.array([[stat[0][3] + stat[1][3] for stat in self.status_history]]))


# A metric that finds the elements of the confusion matrix (correctly labelled positives, false positives, correctly labelled negatives, false negatives) for the validation set.
# Also calculates the confusion matrix with points which score higher than given quantiles (given by the input quantiles) on some uncertainty metric (given by score_fn), evaluated over all points which have the same
# predicted classification, omitted from the validation set. This metric in turn has all the information necessary for the extraction of binary classifier quality metrics such as precision, recall, or F score.
# The status_history elements are tuples of lists, with each count given for each quantile cutoff in quantiles: The true positives, the false positives, the true negatives, and the false negatives.
class ValidationConfusionMatrix(ValidationMetric):
    """A metric that finds the elements of the confusion matrix (correctly labelled positives, false positives, correctly labelled negatives, false negatives) for the validation set.
    Also calculates the confusion matrix with points which score higher than specified quantiles on some specified uncertainty metric, evaluated over all points which have the same
    predicted classification, omitted from the validation set. This metric in turn has all the information necessary for the extraction of binary classifier quality metrics such as precision, recall, or F score.

    Attributes
    ----------
    status_history : list of tuples of lists of ints
        The elements of the status_history object here are tuples of the form (<true positives>, <false positives>, <true negatives>, <false negatives>), where each element of the tuple is a list
        of length equal to the length of the attribute quantiles. Each element of the lists are the values of that observable assuming that we only include points with an uncertainty score (given by score_fn)
        less than or equal to the corresponding quantile (evaluated for all points of the same predicted class) in quantiles.

    name : str
        The unique identifier for the metric in the list of metrics traced by BFBLearner. By default this will be 'val_<score_fn>_confusion'

    score_fn : callable
        A callable of the signature (tf.keras.model, tf.tensor(tf.float32, tf.float32))-> tf.tensor(tf.float32) or (tf.keras.model, tf.tensor(tf.float32, tf.float32), int)-> tf.tensor(tf.float32).
        The pre-implemented functions for different uncertainty metrics can be specified in the constructor by using any of 'BALD' (mutual information),
        'MaxEntropy' (Shannon entropy), 'variation_ratios' (variation ratios), 'predictive_variance' (variance of the prediction distribution),
        or 'uncertainty' (score*(1-score), where score is the Monte Carlo dropout-evaluated prediction for an input)

    tf_score_fn : jit-compiled callable
        Tensorflow jit-compiled version of score_fn.

    n_trials : int or None

    Parameters
    ----------
    score_fn : {'BALD', 'MaxEntropy', 'variation_ratios', 'random', 'uncertainty', 'predictive_variance'} or callable
        Specifies the score function that the metric will apply to the pool of candidate points. If a callable (corresponding to a custom score function) is used,
        it must have the signature (tf.keras.model, tf.tensor(tf.float32, tf.float32))-> tf.tensor(tf.float32) or (tf.keras.model, tf.tensor(tf.float32, tf.float32), int)-> tf.tensor(tf.float32),
        depending on whether or not n_trials is specified.

    name : str, optional
        Allows for a custom name of the metric. If this argument is not specified, a name will be automatically generated as val_<score_fn>_confusion.

    quantiles : list of floats, default=[0.5, 0.75, 0.9, 0.95, 0.99, 1.]
        The uncertainty quantiles for which the metric is tracked (see the status_history documentation)

    n_trials : int, optional
        For score_fn arguments that take a n_trials argument, which includes every pre-implemented score_fn except 'random', this argument can be specified here.
        If it is not specified, the default value for the given score_fn is used.
    """
    def __init__(self, score_fn = 'BALD', name = None, quantiles = [0.5, 0.75, 0.9, 0.95, 0.99, 1.], n_trials = None):
        auto_name = (name is None)
        score_fn, name = process_score_fn(score_fn, name)
        self.score_fn = score_fn
        self.n_trials = n_trials
        if n_trials is not None:
            self.tf_score_fn = tf.function(partial(score_fn, n_trials = n_trials), jit_compile = True)
        else:
            self.tf_score_fn = tf.function(score_fn, jit_compile = True)
        self.quantiles = quantiles
        
        if auto_name:
            super().__init__(name + '_confusion')
        else:
            super().__init__(name)
    
    def performance_check(self, model, ds):
        """Finds the confusion matrix (true positives, false positives, true negatives, false negatives) for the model over the validation set
        """
        uncert = []
        fp_inds = []
        fn_inds = []
        pos_inds = []
        neg_inds = []
        ind_displacement = 0
        for x, y in ds:
            uncert.append(self.tf_score_fn(model, x).numpy())
            results = tf.reshape(MC_call_fast(model, x, 1000), shape = [-1])
            pos_inds.append((tf.reshape(tf.where(results >= 0.5), shape = [-1])).numpy() + ind_displacement)
            neg_inds.append((tf.reshape(tf.where(results < 0.5), shape = [-1])).numpy() + ind_displacement)
            fp_inds.append((tf.reshape(tf.where(tf.logical_and(results >= 0.5, ~y)), shape = [-1])).numpy() + ind_displacement)
            fn_inds.append((tf.reshape(tf.where(tf.logical_and(results < 0.5, y)), shape = [-1])).numpy() + ind_displacement)
            ind_displacement += (tf.shape(y)[0])
        
        uncert = np.concatenate(uncert, axis = 0)
        pos_inds = np.concatenate(pos_inds, axis = 0)
        neg_inds = np.concatenate(neg_inds, axis = 0)
        fp_inds = np.concatenate(fp_inds, axis = 0)
        fn_inds = np.concatenate(fn_inds, axis = 0)

        if len(pos_inds) > 0:
            p_quantiles = np.quantile(uncert[pos_inds], self.quantiles)
        if len(neg_inds) > 0:
            n_quantiles = np.quantile(uncert[neg_inds], self.quantiles)
        
        def get_confusion(f_inds, inds, quants):
            if len(inds) == 0:
                false = [0 for quant in quants]
                true = [0 for quant in quants]
                return true, false
            if len(f_inds) == 0:
                true = [np.count_nonzero(uncert[inds] <= quant) for quant in quants]
                false = [0 for quant in quants]
                return true, false
            total = [np.count_nonzero(uncert[inds] <= quant) for quant in quants]
            false = [np.count_nonzero(uncert[f_inds] <= quant) for quant in quants]
            true = [total[i] - false[i] for i in range(len(quants))]
            return true, false

        true_pos, false_pos = get_confusion(fp_inds, pos_inds, p_quantiles)
        true_neg, false_neg = get_confusion(fn_inds, neg_inds, n_quantiles)
        return (true_pos, false_pos, true_neg, false_neg)
        
    
    def perf_message(self):
        """The perf_message method labels any printed output of the metric.
        """
        return (self.name + ' (validation true positives) {}:'.format(list(self.quantiles)), self.name + ' (validation false positives) {}:'.format(list(self.quantiles)), 
                self.name + ' (validation true negatives) {}:'.format(list(self.quantiles)), self.name + ' (validation false negatives) {}:'.format(list(self.quantiles)))

    def get_metric(self, quantile = None):
        """In this subclass, get_metric can take a keyword argument.

        Parameters
        ----------
        quantile : float in self.quantiles, optional
            If specified, get_metric will only return the values corresponding to the false positives and false negatives found with uncertainty scores
            less than or equal to the given quantile. Otherwise, all false positives and false negatives for all quantiles will be returned for plotting.
        
        Returns
        -------
        np.array
            Represents some plottable set of values from status_history.
        """
        if quantile is not None:
            if quantile not in self.quantiles:
                raise Exception('Please specify a quantile that the metric has recorded. Options are {}'.format(self.quantiles))
            return np.transpose(np.array([[stat[1][self.quantiles.index(quantile)] for stat in self.status_history], [stat[3][self.quantiles.index(quantile)] for stat in self.status_history]]))
        else:
            return np.array([stat[1] + stat[3] for stat in self.status_history])

    def get_legend(self, quantile = None):
        """In this subclass, get_metric can take a keyword argument.

        Parameters
        ----------
        quantile : float in self.quantiles, optional
            The method will return a legend appropriate for plotting get_metric, when get_metric is given the same arguments.
        
        Returns
        -------
        list of str
            A list of strings representing a plot legend for matplotlib.
        """
        if quantile is not None:
            if quantile not in self.quantiles:
                raise Exception('Please specify a quantile that the metric has recorded. Options are {}'.format(self.quantiles))
            return [self.name + 'false_positives_quantile_{}'.format(quantile), self.name + 'false_negatives_quantile_{}'.format(quantile)]
        else:
            return [self.name + 'false_positives_quantile_{}'.format(quant) for quant in self.quantiles] + [self.name + 'false_negatives_quantile_{}'.format(quant) for quant in self.quantiles]
    
    def __getstate__(self):
        """Used to pickle the metric.
        """
        state = self.__dict__.copy()
        del state['tf_score_fn']
        return state
    
    def __setstate__(self, state):
        """Used to unpickle the metric.
        """
        self.__dict__.update(state)
        if self.n_trials is not None:
            self.tf_score_fn = tf.function(partial(self.score_fn, n_trials = self.n_trials), jit_compile = True)
        else:
            self.tf_score_fn = tf.function(self.score_fn, jit_compile = True)

# A metric which gives the precision, recall, and F score with various quantiles of an uncertainty metric excluded from the validation set.
class ValidationFScore(ValidationConfusionMatrix):
    """A metric that finds the precision, recall, and F score for the validation set.
    Also calculates the confusion matrix with points which score higher than specified quantiles on some specified uncertainty metric, evaluated over all points which have the same
    predicted classification, omitted from the validation set. This metric in turn has all the information necessary for the extraction of binary classifier quality metrics such as precision, recall, or F score.

    Attributes
    ----------
    status_history : list of lists of lists of floats
        The elements of the status_history object here are lists of the form [<precision>, <recall>, <F score>], where each element of the tuple is a list
        of length equal to the length of the attribute quantiles. Each element of the lists are the values of that observable assuming that we only include points with an uncertainty score (given by score_fn)
        less than or equal to the corresponding quantile (evaluated for all points of the same predicted class) in quantiles.

    name : str
        The unique identifier for the metric in the list of metrics traced by BFBLearner. By default this will be 'val_<score_fn>_fscore'

    score_fn : callable
        A callable of the signature (tf.keras.model, tf.tensor(tf.float32, tf.float32))-> tf.tensor(tf.float32) or (tf.keras.model, tf.tensor(tf.float32, tf.float32), int)-> tf.tensor(tf.float32).
        The pre-implemented functions for different uncertainty metrics can be specified in the constructor by using any of 'BALD' (mutual information),
        'MaxEntropy' (Shannon entropy), 'variation_ratios' (variation ratios), 'predictive_variance' (variance of the prediction distribution),
        or 'uncertainty' (score*(1-score), where score is the Monte Carlo dropout-evaluated prediction for an input)

    tf_score_fn : jit-compiled callable
        Tensorflow jit-compiled version of score_fn.

    n_trials : int or None

    Parameters
    ----------
    score_fn : {'BALD', 'MaxEntropy', 'variation_ratios', 'random', 'uncertainty', 'predictive_variance'} or callable
        Specifies the score function that the metric will apply to the pool of candidate points. If a callable (corresponding to a custom score function) is used,
        it must have the signature (tf.keras.model, tf.tensor(tf.float32, tf.float32))-> tf.tensor(tf.float32) or (tf.keras.model, tf.tensor(tf.float32, tf.float32), int)-> tf.tensor(tf.float32),
        depending on whether or not n_trials is specified.

    name : str, optional
        Allows for a custom name of the metric. If this argument is not specified, a name will be automatically generated as val_<score_fn>_fscore.

    quantiles : list of floats, default=[0.5, 0.75, 0.9, 0.95, 0.99, 1.]
        The uncertainty quantiles for which the metric is tracked (see the status_history documentation)

    n_trials : int, optional
        For score_fn arguments that take a n_trials argument, which includes every pre-implemented score_fn except 'random', this argument can be specified here.
        If it is not specified, the default value for the given score_fn is used.
    """
    def __init__(self, score_fn = 'BALD', name = None, quantiles = [0.5, 0.75, 0.9, 0.95, 0.99, 1.], n_trials = None):
        auto_name = (name is None)
        score_fn, name = process_score_fn(score_fn, name)
        if auto_name:
            super().__init__(score_fn, name+'_fscore', quantiles, n_trials)

    def performance_check(self, model, ds):
        """Computes the precision, recall, and F score over the validation set.
        """
        confusion = super().performance_check(model, ds)
        prec = [confusion[0][i] / (confusion[0][i] + confusion[1][i]) for i in range(len(confusion[0]))]
        rec = [confusion[0][i] / (confusion[0][i] + confusion[3][i]) for i in range(len(confusion[0]))]
        fscore = [2*prec[i]*rec[i] / (prec[i] + rec[i]) for i in range(len(prec))]
        return (prec, rec, fscore)
    
    def perf_message(self):
        """The perf_message method labels any printed output of the metric.
        """
        return (self.name + ' (validation precision) {}:'.format(list(self.quantiles)), self.name + ' (validation recall) {}:'.format(list(self.quantiles)), 
                self.name + ' (validation F score) {}:'.format(list(self.quantiles)))

    def get_metric(self, quantile = None):
        """In this subclass, get_metric can take a keyword argument.

        Parameters
        ----------
        quantile : float in self.quantiles, optional
            If specified, get_metric will only return the values corresponding to the F score found with uncertainty scores
            less than or equal to the given quantile. Otherwise, all F scores for all quantiles will be returned for plotting.
        
        Returns
        -------
        np.array
            Represents some plottable set of values from status_history.
        """
        if quantile is not None:
            if quantile not in self.quantiles:
                raise Exception('Please specify a quantile that the metric has recorded. Options are {}'.format(self.quantiles))
            return np.array([stat[2][self.quantiles.index(quantile)] for stat in self.status_history])
        else:
            return np.array([stat[2] for stat in self.status_history])
        
    def get_legend(self, quantile = None):
        """In this subclass, get_metric can take a keyword argument.

        Parameters
        ----------
        quantile : float in self.quantiles, optional
            The method will return a legend appropriate for plotting get_metric, when get_metric is given the same arguments.
        
        Returns
        -------
        list of str
            A list of strings representing a plot legend for matplotlib.
        """
        if quantile is not None:
            if quantile not in self.quantiles:
                raise Exception('Please specify a quantile that the metric has recorded. Options are {}'.format(self.quantiles))
            return [self.name + ' quantile  <= {}'.format(quantile)]
        else:
            return [self.name + ' quantile <= {}'.format(quant) for quant in self.quantiles]


class PoolScore(PoolMetricReduction):
    """A metric which applies the function score_fn to the pool of candidate points at every active learning iteration, before the model is trained on any new data drawn from the pool.
    Evaluates score_fn on the pool points and records specified reductions of these scores.

    Attributes
    ----------
    status_history : list of lists of floats (or np.float32)
        Each entry in this status_history object has entries corresponding to the score_fn results applied to each active learning iteration's pool of candidate points, with 
        reduction(s) specified in the constructor. If the constructor specifies multiple reductions, each entry is a list with each value's reduction (so an entry will be [<mean>, <max>, <min>], for example).

    name : str
        A string which denotes a name for this metric. In a list of metrics passed to a BFBLearner class, the names of each member of the list should be unique. By default will be 'pool_<score_fn>'

    score_fn : callable
        A callable of the signature (tf.keras.model, tf.tensor(tf.float32, tf.float32))-> tf.tensor(tf.float32) or (tf.keras.model, tf.tensor(tf.float32, tf.float32), int)-> tf.tensor(tf.float32).
        The pre-implemented functions for different uncertainty metrics can be specified in the constructor by using any of 'BALD' (mutual information),
        'MaxEntropy' (Shannon entropy), 'variation_ratios' (variation ratios), 'predictive_variance' (variance of the prediction distribution),
        or 'uncertainty' (score*(1-score), where score is the Monte Carlo dropout-evaluated prediction for an input)

    tf_score_fn : jit-compiled callable
        Tensorflow jit-compiled version of score_fn.

    n_trials : int or None

    Parameters
    ----------
    score_fn : {'BALD', 'MaxEntropy', 'variation_ratios', 'random', 'uncertainty', 'predictive_variance'} or callable
        Specifies the score function that the metric will apply to the pool of candidate points. If a callable (corresponding to a custom score function) is used,
        it must have the signature (tf.keras.model, tf.tensor(tf.float32, tf.float32))-> tf.tensor(tf.float32) or (tf.keras.model, tf.tensor(tf.float32, tf.float32), int)-> tf.tensor(tf.float32),
        depending on whether or not n_trials is specified.

    name : str, optional
        Allows for a custom name of the metric. If this argument is not specified, a name will be automatically generated as pool_<score_fn>.

    reduction : {'mean', 'min', 'max'} or a list of these values, default=['mean','min','max']
        Specifies what reductions should be done on the scores computed for the pool candidate points. If a list is specified, all reductions in the list are computed.

    n_trials : int, optional
        For score_fn arguments that take a n_trials argument, which includes every pre-implemented score_fn except 'random', this argument can be specified here.
        If it is not specified, the default value for the given score_fn is used.
    """
    def __init__(self, score_fn = 'BALD', name = None, reduction = ['mean', 'min', 'max'], n_trials = None):
        score_fn, name = process_score_fn(score_fn, name)
        super().__init__(name, reduction)
        self.score_fn = score_fn
        self.n_trials = n_trials
        if n_trials is not None:
            self.tf_score_fn = tf.function(partial(score_fn, n_trials = n_trials), jit_compile = True)
        else:
            self.tf_score_fn = tf.function(score_fn, jit_compile = True)
    
    def performance_check(self, *args):
        """performance_check for this metric records the score function evaluated on a pool of candidate points, subject to the reduction(s) specified in the constructor.

        Parameters
        ----------
        *args : (tf.keras.Model, tf.tensor(tf.float32, tf.float32)) or tf.tensor(tf.float32)
            If the class's score_fn were already evaluated as part of active learning (namely, if the score_fn corresponds to the one used as the acquisition function in active learning),
            then this method can take an already-evaluated Tensorflow tensor consisting of a list of scores. If not, then we can pass the arguments appropriate for score_fn
            in order to evaluate the results directly.
            
        Returns
        -------
        list of np.float32
            A list of the specified reductions in the score function for a batch of pool candidate points. These are then
            combined into a single entry into status_history using the methods in the parent class.
        """
        if isinstance(args[0], tf.keras.Model):
            score = self.tf_score_fn(*args).numpy()
        else:
            score = args[0].numpy()
        return [red(score) for red in self.reduction]
    
    def __getstate__(self):
        """Used to pickle the metric.
        """
        state = self.__dict__.copy()
        del state['tf_score_fn']
        return state
    
    def __setstate__(self, state):
        """Used to unpickle the metric.
        """
        self.__dict__.update(state)
        if self.n_trials is not None:
            self.tf_score_fn = tf.function(partial(self.score_fn, n_trials = self.n_trials), jit_compile = True)
        else:
            self.tf_score_fn = tf.function(self.score_fn, jit_compile = True)

# A metric which applies the function score_fn to the points selected to be added to the training set at every active learning iteration, before the model is trained on this new data.
# Evaluates score_fn on the newly selected data set and records the reductions specified in the input reduction (can be any of 'mean', 'min', 'max', 'std', or a list of those) in the status_history object.
class NewDataScore(TrainMetric):
    """A metric which applies the function score_fn to the set of points that are added to the training set
    at every active learning iteration, before the model is trained on the new data.
    Evaluates score_fn on these points and records specified reductions of these scores.

    Attributes
    ----------
    status_history : list of lists of floats (or np.float32)
        Each entry in this status_history object has entries corresponding to the score_fn results applied to each active learning iteration's new training data, with 
        reduction(s) specified in the constructor. If the constructor specifies multiple reductions, each entry is a list with each value's reduction (so an entry will be [<mean>, <max>, <min>], for example).

    name : str
        A string which denotes a name for this metric. In a list of metrics passed to a BFBLearner class, the names of each member of the list should be unique. By default will be 'train_<score_fn>'

    score_fn : callable
        A callable of the signature (tf.keras.model, tf.tensor(tf.float32, tf.float32))-> tf.tensor(tf.float32) or (tf.keras.model, tf.tensor(tf.float32, tf.float32), int)-> tf.tensor(tf.float32).
        The pre-implemented functions for different uncertainty metrics can be specified in the constructor by using any of 'BALD' (mutual information),
        'MaxEntropy' (Shannon entropy), 'variation_ratios' (variation ratios), 'predictive_variance' (variance of the prediction distribution),
        or 'uncertainty' (score*(1-score), where score is the Monte Carlo dropout-evaluated prediction for an input)

    tf_score_fn : jit-compiled callable
        Tensorflow jit-compiled version of score_fn.

    n_trials : int or None

    reduction : a list containing elements of {np.mean, np.min, np.max}
        This is the reduction that is performed on the scores to produce the entries in status_history.
        If a list of reductions are applied, then the elements of status_history will be lists with each element being a different reduction being applied to the scores.

    red_name : a list containing elements of {'mean', 'min', 'max'}
        This list of strings will contain the same information as reduction, but is used for labelling purposes.

    Parameters
    ----------
    score_fn : {'BALD', 'MaxEntropy', 'variation_ratios', 'random', 'uncertainty', 'predictive_variance'} or callable
        Specifies the score function that the metric will apply to the pool of candidate points. If a callable (corresponding to a custom score function) is used,
        it must have the signature (tf.keras.model, tf.tensor(tf.float32, tf.float32))-> tf.tensor(tf.float32) or (tf.keras.model, tf.tensor(tf.float32, tf.float32), int)-> tf.tensor(tf.float32),
        depending on whether or not n_trials is specified.

    name : str, optional
        Allows for a custom name of the metric. If this argument is not specified, a name will be automatically generated as 'train_<score_fn>'.

    red : {'mean', 'min', 'max'} or a list of these values, default=['mean','min','max']
        Specifies what reductions should be done on the scores computed for the new training data. If a list is specified, all reductions in the list are computed.

    n_trials : int, optional
        For score_fn arguments that take a n_trials argument, which includes every pre-implemented score_fn except 'random', this argument can be specified here.
        If it is not specified, the default value for the given score_fn is used.
    """
    def __init__(self, score_fn = 'BALD', name = None, red = ['mean', 'min', 'max'], n_trials = None):
        self.reduction, self.red_name = _get_reduction(red)
        score_fn, name = process_score_fn(score_fn, name)
        super().__init__(name = name)
        self.score_fn = score_fn
        self.n_trials = n_trials
        if n_trials is not None:
            self.tf_score_fn = tf.function(partial(score_fn, n_trials = n_trials), jit_compile = True)
        else:
            self.tf_score_fn = tf.function(score_fn, jit_compile = True)
    
    def performance_check(self, model, lams, labels):
        """Computes score_fn on the points added to the training set with each active learning iteration, and then
        records the specified reductions.
        """
        score = self.tf_score_fn(model, lams).numpy()
        return [red(score) for red in self.reduction]
    
    def perf_message(self):
        """The perf_message method labels any printed output of the metric.
        """
        return self.name + ' (new queried data) {}:'.format(list(self.red_name))

    def get_metric(self, reduction = None):
        """In this subclass, get_metric can take a keyword argument.

        Parameters
        ----------
        reduction : str in self.red_name, optional
            If specified, get_metric will only return the values corresponding to the specified reduction. If not specified, get_metric will return the status_history object in its entirety.

        Returns
        -------
        np.array
            Represents some plottable set of values from status_history.
        """
        if reduction is not None:
            _check_reduction(reduction, self.red_name)
            return np.transpose(np.array([[stat[self.red_name.index(reduction)] for stat in self.status_history]]))
        else:
            return np.array(self.status_history)
        
    def get_legend(self, reduction = None):
        """In this subclass, get_legend can take a keyword argument.

        Parameters
        ----------
        reduction : str in self.red_name, optional
            get_legend will return a legend consistent with the get_metric result with the same reduction argument passed.

        Returns
        -------
        list of str
            An argument to specify a legend in matplotlib.
        """
        if reduction is not None:
            return [reduction + '_' + self.name]
        else:
            return [red + '_' + self.name for red in self.red_name]
    
    def __getstate__(self):
        """Used to pickle the metric.
        """
        state = self.__dict__.copy()
        del state['tf_score_fn']
        return state
    
    def __setstate__(self, state):
        """Used to unpickle the metric.
        """
        self.__dict__.update(state)
        if self.n_trials is not None:
            self.tf_score_fn = tf.function(partial(self.score_fn, n_trials = self.n_trials), jit_compile = True)
        else:
            self.tf_score_fn = tf.function(self.score_fn, jit_compile = True)

class StoppingCondition:
    """A generic class for implementing early stopping conditions for active learning.
    A StoppingCondition object is called each round immediately after the metric it follows is evaluated.
    Then, if the call returns True, the active learning loop is terminated.

    Attributes
    ----------
    metric_name : str
        Denotes the name of the performance metric (that is, the ALMetric object's name str) that the StoppingCondition should track
    metric_func : callable
        A callable which takes a metric and an (optional) integer index as input and returns True if the stopping condition has been met, and False otherwise.

    Parameters
    ----------
    metric_name : str
        Denotes the name of the performance metric (that is, the ALMetric object's name str) that the StoppingCondition should track
    metric_func : callable
        A callable which takes a metric and an (optional) integer index as input and returns True if the stopping condition has been met, and False otherwise.
    """
    def __init__(self, metric_name, metric_func):
        self.metric_name = metric_name
        self.metric_func = metric_func

    def __call__(self, metrics_dict, ind = None):
        """Calling the StoppingCondition class on a dictionary of metrics (of the form, {metric.name : metric}) will make it perform its function on the metric
        it is following.

        Parameters
        ----------
        metrics_dict : dict
            A dictionary relating the names of ALMetric objects (or rather child classes of this class) to the objects themselves.
        ind : int, optional
            If specified, the StoppingCondition object only considers status_history[:ind] instead of the full status_history. This is useful for retroactively
            determining if a stopping condition would have eliminated unnecessary active learning iterations.

        Returns
        -------
        bool
            True if the StoppingCondition determines we should stop active learning, False otherwise.
        """
        try:
            metric_in = metrics_dict[self.metric_name]
        except KeyError:
            raise Exception('metric_name is not within the metrics being recorded by ActiveLearning. Must be one of {}.'.format(list(metrics_dict.keys())))
        return self.metric_func(metric_in, ind)
    
    def find_stopping_index(self, metrics_dict):
        """Computes the index (active learning iteration) at which this StoppingCondition WOULD have stopped active learning
        if it were applied to the metrics for an already-trained BFBLearner object.

        Parameters
        ----------
        metrics_dict : dict
            A dictionary relating the names of ALMetric objects (or rather child classes of this class) to the objects themselves, extracted from
            a trained BFBLearner object.
        
        Returns
        -------
        int
            An integer representing the active learning round at which the StoppingCondition would have stopped active learning, if it had been implemented during training.
            If the condition would not have been met, returns -1.
        """
        metric = metrics_dict[self.metric_name]
        for ind in range(1, len(metric.status_history) + 1):
            if self(metrics_dict, ind):
                return ind
        return -1

class ScoreNotDecreasing(StoppingCondition):
    """A stopping condition based on when an uncertainty score (in particular BALD or variation ratios) is not decreasing over some data set
    (usually the pool of candidate points proposed by the classifier or the set of training points added as training data). Because mutual information,
    variation ratios, and predictive variance are all in theory metrics of epistemic uncertainty (or in the case of the second, at least highly sensitive 
    to it), some measurement of these scores should be decreasing as more data is added. If it's not, then the network probably reached close to the highest quality 
    it's capable of attaining.

    Attributes
    ----------
    metric_name : str
        Denotes the name of the performance metric (that is, the ALMetric object's name str) that the StoppingCondition should track

    metric_func : callable
        A callable which takes a metric and an (optional) integer index as input and returns 
        True if the stopping condition has been met, and False otherwise.
        In this case, metric_func checks to see if a specified uncertainty
        score on some set of points hasn't achieved a new minimum over some specified number of rounds.

    reduction : {'mean', 'min', 'max', 'std'}
        Must be some reduction over the score that the metric specified by metric_name has evaluated. This is the specific quantity that the stopping condition monitors.'

    patience : int, default=5
        The number of rounds without achieving a new minimum for its monitored quantity that the stopping condition tolerates before halting active learning.

    Parameters
    ----------
    metric_name : str
        Denotes the name of the performance metric (that is, the ALMetric object's name str) that the StoppingCondition should track.
    
    reduction : {'mean', 'min', 'max', 'std'}
        Must be some reduction over the score that the metric specified by metric_name has evaluated. This is the specific quantity that the stopping condition monitors.'
    
    patience : int, default=5
        The number of rounds without achieving a new minimum for its monitored quantity that the stopping condition tolerates before halting active learning.
    """
    def __init__(self, metric_name, reduction = 'mean', patience = 5):
        self.reduction = reduction
        self.patience = patience
        def metric_func(metric, ind):
            if (not hasattr(metric, 'red_name')) or (not hasattr(metric, 'score_fn')):
                raise Exception('The specified metric is not a scoring metric, and so this stopping condition is not applicable.')
            if ind is None:
                ind = len(metric.status_history)
            if ind < self.patience:
                return False
            red_index = metric.red_name.index(self.reduction)
            min_score = min([stat[red_index] for stat in metric.status_history[:ind]])
            min_score_arg = [stat[red_index] for stat in metric.status_history[:ind]].index(min_score)
            if min_score_arg in range(ind-self.patience,ind):
                return False
            last_status_history = metric.status_history[ind-self.patience:ind]
            return all([elem[red_index] >= min_score for elem in last_status_history])
        super().__init__(metric_name, metric_func)

class AccuracyNotImproving(StoppingCondition):
    """A stopping condition that monitors either a ModelEvaluation or MCModelEvaluation metric and stops the active learning after some number of rounds have passed without achieving a new maximum accuracy.
    
    Attributes
    ----------
    metric_name : str
        Denotes the name of the performance metric (that is, the ALMetric object's name str) that the StoppingCondition should track

    metric_func : callable
        A callable which takes a metric and an (optional) integer index as input and returns 
        True if the stopping condition has been met, and False otherwise.
        In this case, metric_func checks to see if the accuracy entry for a ModelEvaluation or MCModelEvaluation metric
        hasn't achieved a new maximum over some specified number of rounds.

    patience : int, default=5
        The number of rounds without achieving a new maximum for its monitored quantity that the stopping condition tolerates before halting active learning.

    Parameters
    ----------
    metric_name : str
        Denotes the name of the performance metric (that is, the ALMetric object's name str) that the StoppingCondition should track

    patience : int, default=5
        The number of rounds without achieving a new maximum for its monitored quantity that the stopping condition tolerates before halting active learning.
    """
    def __init__(self, metric_name, patience = 5):
        self.patience = patience
        def metric_func(metric, ind):
            if (not isinstance(metric, ModelEvaluation)) and (not isinstance(metric, MCModelEvaluation)):
                raise Exception('The specified metric is not a ModelEvaluation or MCModelEvaluation metric, and so this stopping condition is not applicable.')
            if ind is None:
                ind = len(metric.status_history)
            if ind < self.patience:
                return False
            
            max_accuracy = max([stat[0] for stat in metric.status_history[:ind]])
            last_status_history = metric.status_history[ind-self.patience:ind]
            max_accuracy_arg = [stat for stat in metric.status_history[:ind]].index(max_accuracy)
            if max_accuracy_arg in range(ind-self.patience,ind):
                return False
            return all([elem[0] <= max_accuracy for elem in last_status_history])
        super().__init__(metric_name, metric_func)

class FScoreNotImproving(StoppingCondition):
    """A stopping condition that monitors a ValidationConfusionMatrix or ValidationFScore metric and stops the active learning after some number of rounds have passed without achieving a new maximum F score.

    Attributes
    ----------
    metric_name : str
        Denotes the name of the performance metric (that is, the ALMetric object's name str) that the StoppingCondition should track

    metric_func : callable
        A callable which takes a metric and an (optional) integer index as input and returns 
        True if the stopping condition has been met, and False otherwise.
        In this case, metric_func checks to see if the F score evaluated over some uncertainty quantile 
        (see ValidationConfusionMatrix and ValidationFScore documentation for details) hasn't
        achieved a new maximum in patience rounds.

    quant : float, default=1.0
        The uncertainty quantile that the stopping condition should check. Default is 1.0, meaning the entire validation set is considered.
    
    patience : int, default=5
        The number of rounds without achieving a new maximum for its monitored quantity that the stopping condition tolerates before halting active learning.

    Parameters
    ----------
    metric_name : str
        Denotes the name of the performance metric (that is, the ALMetric object's name str) that the StoppingCondition should track

    quant : float, default=1.0
        The uncertainty quantile that the stopping condition should check. Default is 1.0, meaning the entire validation set is considered.
    
    patience : int, default=5
        The number of rounds without achieving a new maximum for its monitored quantity that the stopping condition tolerates before halting active learning.
    """
    def __init__(self, metric_name, quant = 1.0, patience = 5):
        self.patience = patience
        self.quant = quant
        def metric_func(metric, ind):
            if (not isinstance(metric, ValidationConfusionMatrix)):
                raise Exception('The specified metric is not ValidationConfusionMatrix, and so this stopping condition is not applicable.')
            try:
                q_index = metric.quantiles.index(quant)
            except ValueError:
                raise Exception('the specified quantile is not recorded in the metric. Please specify one of {}'.format(metric.quantiles))
            if ind is None:
                ind = len(metric.status_history)
            if ind < self.patience:
                return False
            
            if isinstance(metric, ValidationFScore):
                fscore = np.array([stat[2] for stat in metric.status_history]).T
            else:
                true_pos = np.array([stat[0] for stat in metric.status_history]).T
                false_pos = np.array([stat[1] for stat in metric.status_history]).T
                false_neg = np.array([stat[3] for stat in metric.status_history]).T
                prec = true_pos / (true_pos + false_pos)
                rec = true_pos / (true_pos + false_neg)
                fscore = 2*prec*rec / (prec + rec)

            max_fscore = max(fscore[q_index][:ind])
            max_fscore_arg = fscore[q_index][:ind].index(max_fscore)
            last_status_history = fscore[q_index][ind-self.patience:ind]
            if max_fscore_arg in range(ind-self.patience,ind):
                return False
            return np.all(last_status_history <= max_fscore)
        super().__init__(metric_name, metric_func)


class DeltaFNotDecreasing(StoppingCondition):
    """A stopping condition that monitors a PoolDeltaF metric and stops active learning once the classifier's estimated change in F score has not decreased for
    a specified number of rounds.

    Attributes
    ----------
    metric_name : str
        Denotes the name of the performance metric (that is, the ALMetric object's name str) that the StoppingCondition should track

    metric_func : callable
        A callable which takes a metric and an (optional) integer index as input and returns 
        True if the stopping condition has been met, and False otherwise.
        In this case, metric_func checks to see if the estimated change in F score
        has not achieved a new minimum in patience rounds.
    
    patience : int, default=5
        The number of rounds without achieving a new maximum for its monitored quantity that the stopping condition tolerates before halting active learning.

    Parameters
    ----------
    metric_name : str
        Denotes the name of the performance metric (that is, the ALMetric object's name str) that the StoppingCondition should track
    
    patience : int, default=5
        The number of rounds without achieving a new maximum for its monitored quantity that the stopping condition tolerates before halting active learning.
    """
    def __init__(self, metric_name, patience = 5):
        self.patience = patience
        def metric_func(metric, ind):
            if (not isinstance(metric, PoolDeltaF)) and (not isinstance(metric, UnlabelledDeltaF)):
                raise Exception('The specified metric is not a PoolDeltaF metric, and so this stopping condition is not applicable.')
            if ind is None:
                ind = len(metric.status_history)
            if ind < self.patience:
                return False
            min_score = min([stat for stat in metric.status_history[:ind]])
            min_score_arg = [stat for stat in metric.status_history[:ind]].index(min_score)
            last_status_history = metric.status_history[ind-self.patience:ind]
            if min_score_arg in range(ind-self.patience,ind):
                return False
            return all([elem >= min_score for elem in last_status_history])
        super().__init__(metric_name, metric_func)