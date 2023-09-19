.. _classifier:

Tutorial Step 2: Initializing the Classifier
============================================

After initializing the oracle wrapped in a :class:`DataManager <bfbrain.Data_Manager.DataManager>` object, the next step of a BFBrain analysis is to initialize the classifier which will be trained to predict whether
points in parameter space are bounded-from-below. This is done by initializing a :class:`BFBLearner <bfbrain.BFB_Learner.BFBLearner>` object. The :class:`BFBLearner <bfbrain.BFB_Learner.BFBLearner>` object
contains a variety of data and objects, most notably:

* The neural network that shall act as the classifier (a Bayesian neural network approximated using concrete dropout [1]_)

* User-specified performance metrics gathered during each active learning iteration

* The :class:`DataManager <bfbrain.Data_Manager.DataManager>` object initialized before training.

The full :class:`BFBLearner <bfbrain.BFB_Learner.BFBLearner>` object can be saved and loaded in order to resume training after an initial training script finishes, or
to access and use a fully trained classifier for numerical studies. Instantiating the object for the first time should be done with :meth:`BFBLearner.init_for_first_run <bfbrain.BFB_Learner.BFBLearner.init_for_first_run>`.
A typical example of a constructor, following the example 2HDM :class:`DataManager <bfbrain.Data_Manager.DataManager>` that we instantiated in :ref:`the previous section of the tutorial <defaultoracle>`,
is given by

.. code-block:: python

    from bfbrain import BFBLearner, ValidationFScore, UnlabelledDeltaF
    
    # Running this line should take a few minutes, due to the need to label the validation set.
    AL = BFBLearner.init_for_first_run(dm, 5, 128, [ValidationFScore(), UnlabelledDeltaF(dm.create_random_lambdas(1000000, validation = True))], 1000)

    # After initializing the constructor, it's a good idea to save a backup in case we want to modify
    # without relabelling a whole new validation and initial training set.
    AL.save_AL_state('saved_AL_untrained')

In the above constructor, we have specified that we should use our previously defined :class:`DataManager <bfbrain.Data_Manager.DataManager>`, dm, build a classifier with five hidden layers
and 128 neurons per layer, and produce an initial training set of 1000 uniformly sampled points in the space of quartic coefficients. The classifier should also keep track of
two performance metrics during each round: The :math:`F_1` score of the classifier on a separate uniformly sampled validation set that is by default 100 times larger than the specified initial training set size 
(so in our case consisting of :math:`10^5` points), and the *estimated* change in the :math:`F_1` score over an even larger, but unlabelled, uniformly sampled set, which we 
determine via the procedure discussed in [2]_. Note that a validation set will only be produced if one of the performance metrics specified in the list passed to the 
constructor (in this case :class:`ValidationFScore <bfbrain.AL_Metrics.ValidationFScore>`) requires a validation data set-- otherwise the constructor will not engage in the likely computationally expensive 
task of labelling such a set with the oracle function.

Meanwhile, we have also taken the liberty of using BFBrain's serialization capabilities to save a backup of our untrained :class:`BFBLearner <bfbrain.BFB_Learner.BFBLearner>` object under in a new directory,
'saved_AL_untrained'. We can now load this :class:`BFBLearner <bfbrain.BFB_Learner.BFBLearner>` later on, for example in :ref:`the next step of the tutorial <training>`. Saving and loading the
:class:`BFBLearner <bfbrain.BFB_Learner.BFBLearner>` object can also allow us to experiment more easily with different hyperparameters, especially in cases where the constructor is expensive to run.
As an example, we can load a copy of our untrained network and adjust a number of aspects of the :class:`BFBLearner <bfbrain.BFB_Learner.BFBLearner>` object before training.

.. code-block:: python

    # Demonstrate loading and adjusting hyperparameters here.
    from bfbrain import MCModelEvaluation

    # Load the BFBLearner we just saved. It should be a copy of AL.
    loaded_AL = BFBLearner.from_file('saved_AL_untrained')

    # Redefine the model to feature 3 layers of 256 neurons instead of 5 layers of 128 neurons.
    loaded_AL.redefine_model(3, 256)

    # Adjust the prior length scale l (weights have a Bayesian prior of N(0, 1/l**2))
    loaded_AL.set_l_constant(0.01)

    # Add a new performance metric.
    loaded_AL.add_metrics(MCModelEvaluation())

.. _metrics:

Performance Metrics
-------------------

There are a variety of performance metrics which a user can have BFBrain track during active learning, any/all of which can be implemented by including them in the list of metrics required by
:meth:`init_for_first_run <bfbrain.BFB_Learner.BFBLearner.init_for_first_run>`. 

For tracking classifier performance, we recommend (in no particular order):

* :class:`ModelEvaluation <bfbrain.AL_Metrics.ModelEvaluation>` and :class:`MCModelEvaluation <bfbrain.AL_Metrics.MCModelEvaluation>`: These metrics track the binary accuracy, number of false positives, and number of false negatives that the classifier produces on a validation set.

* :class:`ValidationFScore <bfbrain.AL_Metrics.ValidationFScore>`: This metric computes the :math:`F_1` score that the classifier attains on a labelled validation set, which will in general be more informative about classifier performance than the binary accuracy, since validation sets will generally be highly unbalanced in favor of the not bounded-from-below class. Because the neural network model is also capable of quantifying uncertainty through several means (see :mod:`bfbrain.Score_Functions` for details), it will further compute the :math:`F_1` score when points that have a higher uncertainty score than some quantile of the points with the same predicted label are omitted from the validation set. The quantiles that are used here can be user-specified.

* :class:`UnlabelledDeltaF <bfbrain.AL_Metrics.UnlabelledDeltaF>`: This metric computes and *estimated change* in :math:`F_1` on an *unlabelled* user-specified set of inputs between successive rounds of active learning, following [2]_. This allows a user to gauge improving model performance (as the model improves, :math:`\Delta F_1` will approach 0) without needing to use the oracle to label a large validation data set, with the caveat that the *estimated* :math:`\Delta F_1` values tend to significantly overestimate the *actual* change in :math:`F_1` score if the set were labelled with the oracle-- however the metric will still decrease steadily as the classifier performance improves.

For a comprehensive list of available performance metrics and their options, we refer the reader to the documentation for :mod:`the performance metrics module <bfbrain.AL_Metrics>`. All performance metrics
in BFBrain contain an object called :attr:`status_history <bfbrain.AL_Metrics.ALMetric.status_history>` which records the information that the metric wishes to track (usually a number of some sort, or a list numbers, or a tuple of lists of numbers) during each
round of active learning-- the elements of :attr:`status_history <bfbrain.AL_Metrics.ALMetric.status_history>` then correspond to the values of the metric recorded for a given round. Before using a particular performance metric for the first time, it is **highly**
recommended that a user read its documentation and familiarize themselves with what form the elements of :attr:`status_history <bfbrain.AL_Metrics.ALMetric.status_history>` take.

In addition to recording information, all performance metric objects feature methods which allow for human-readable printouts of their values during training and easy plotting of their results
for inspection after training. During training, BFBrain metrics will report the values of their tracked metrics to a .txt file after each round of active learning, as well as (if the user requests it)
print them to the console. For example, :class:`MCModelEvaluation <bfbrain.AL_Metrics.MCModelEvaluation>` will, at the end of each round, report a message of the form

::

    MC validation accuracy [accuracy, false positives, false negatives]:
    [<accuracy>, <false positives>, <false negatives>]

with the appropriate values instead of each carated entry in the above. Metrics that have tuples as entries in :attr:`status_history <bfbrain.AL_Metrics.ALMetric.status_history>` instead report each result with
different headings for each tuple entry-- this can be useful for metrics like :class:`ValidationFScore <bfbrain.AL_Metrics.ValidationFScore>`, which tracks three different quantities (precision, recall, and :math:`F_1` score)
for validation sets where points which exceed specified uncertainty quantiles are omitted. The form of the printouts for this metric are 

::

    val_<uncertainty>_fscore (validation precision) [<quantile1>, <quantile2>, ...]:
    [<precision1>, <precision2>, ...]
    val_<uncertainty>_fscore (validation recall) [<quantile1>, <quantile2>, ...]:
    [<recall1>, <recall2>, ...]
    val_<uncertainty>_fscore (validation F score) [<quantile1>, <quantile2>, ...]:
    [<fscore1>, <fscore2>, ...]

where again carated quantities are replaced with the appropriate values-- <uncertainty> refers to the method used to estimate the model uncertainty, in this case used to exclude points in the validation set from
the computation of the precision, recall, and :math:`F_1` score-- we shall discuss these in detail in :doc:`training`.

The final convenience method for performance metrics in BFBrain relates to their plotting-- BFBrain metrics can automatically produce simple plots of their results in order to give a user a
visual sense of the results of their experiments. This is accomplished simply by calling :meth:`plot_metric <bfbrain.AL_Metrics.ALMetric.plot_metric>`, which will plot the metric in the console or to a specified
.png file. We shall use this capability in :doc:`training` to observe the results of our training.


.. _custommetrics:

Advanced Usage: Custom Performance Metrics
++++++++++++++++++++++++++++++++++++++++++

In addition to the selection of performance metrics that are implemented as part of BFBrain, users may create their own customized performance metrics by writing classes that inherit from
the abstract class :class:`ALMetric <bfbrain.AL_Metrics.ALMetric>`, or one of several abstract child classes that we have implemented for convenience. It should be noted that creating a 
customized performance metric is highly involved, and in almost all cases a user should be able to extract performance metric information from any of BFBrain's pre-implemented metrics--
a user may only wish to read this section if they find they have a need of a performance metric not implemented in :mod:`the performance metrics module <bfbrain.AL_Metrics>`.

To outline the basic steps of implementing a new metric, it is easiest to demonstrate the implementation of a simple custom metric, so we create the custom metric TrainPosFraction.
At each round of active learning, after a new set of points has been selected to be added to the training data and labelled, this metric computes the fraction of these points which
the oracle has labelled as bounded-from-below, and records them in :attr:`status_history <bfbrain.AL_Metrics.ALMetric.status_history>` for later analysis.

.. code-block:: python

    from bfbrain import TrainMetric

    class TrainPosFraction(TrainMetric):
        """
        Implement a new metric which records the fraction of
        each newly-added set of training points that the
        oracle labels as positive.
        """

        def __init__(self, name = 'pos_fraction'):
            super().__init__(name = name)

        def performance_check(self, model, lams, labels):
            """The class overwrites the superclass's abstract 
            performance_check method with a concrete
            computation. This is the value that is recorded
            in the performance metric's status_history object.
            Notice that although not all arguments are used in the
            computation of this result, the arguments of
            performance_check are set by the parent class.
            """
            return np.count_nonzero(labels) / len(labels)
        
To implement the above metric, we needed to take two steps: First, we needed to identify what information the metric needed from the :class:`BFBLearner <bfbrain.BFB_Learner.BFBLearner>` object.
Since our metric uses the new training data generated during each active learning iteration, we determined that it should inherit from the :class:`TrainMetric <bfbrain.AL_Metrics.TrainMetric>` child class of :class:`ALMetric <bfbrain.AL_Metrics.ALMetric>`.
Second, we needed to specify the abstract :meth:`performance_check <bfbrain.AL_Metrics.TrainMetric.performance_check>` method with a concrete method that would return the quantity that
we wished the metric to record.

The approach to implementing general custom metrics follows this pattern. First, we identify the information from the :class:`BFBLearner <bfbrain.BFB_Learner.BFBLearner>` object that the
metric requires. BFBrain supports five types of metrics, each of which have a corresponding child class of :class:`ALMetric <bfbrain.AL_Metrics.ALMetric>` that can serve as a basis for customization. They are:

* :class:`ValidationMetric <bfbrain.AL_Metrics.ValidationMetric>`: These metrics track the predictions of a :class:`BFBLearner <bfbrain.BFB_Learner.BFBLearner>`'s model on a labelled set of validation data. They are computed at the end of each active learning step. An example of a fully implemented metric of this class is :class:`MCModelValidation <bfbrain.AL_Metrics.MCModelValidation>`.
* :class:`TrainMetric <bfbrain.AL_Metrics.TrainMetric>`: These metrics track the predictions of a :class:`BFBLearner <bfbrain.BFB_Learner.BFBLearner>`'s model on the new labelled data that is appended to the training set during each active learning round. They are computed immediately after the new training data is appended to the training set, before the network is trained on the new data. An example of a fully implemented metric of this class is :class:`NewDataScore <bfbrain.AL_Metrics.NewDataScore>`.
* :class:`PoolMetric <bfbrain.AL_Metrics.PoolMetric>`: These metrics track the predictions of a :class:`BFBLearner <bfbrain.BFB_Learner.BFBLearner>`'s model on the unlabelled pool of candidate points from which training data is drawn during each active learning round. Because the pool is produced and evaluated in a series of discrete manageable batches of points, this metric is computed immediately after each batch of pool points is produced in active learning, and then the metric's computations on each batch are merged into a single entry of :attr:`status_history <bfbrain.AL_Metrics.ALMetric.status_history>`. It is highly recommended to **thoroughly** inspect the documentation and source code of :class:`PoolMetric <bfbrain.AL_Metrics.PoolMetric>` before attempting to implement a metric of this class, since the procedure here is considerably more complex than for other metric types. An example of a fully implemented metric of this type is :class:`PoolScore <bfbrain.AL_Metrics.PoolScore>`
* :class:`ModelMetric <bfbrain.AL_Metrics.ModelMetric>`: These metrics are a catch-all category for any class of metric that only requires the model from the :class:`BFBLearner <bfbrain.BFB_Learner.BFBLearner>` object to record their values. Because other information can be stored in the specific object, this type of metric can be used to make a variety of different computations, for example observing aspects of the weights of the network or the predicted labels on some external dataset. They are computed at the end of each active learning iteration. There are no direct implemented versions of this type of metric in BFBrain, although :class:`UnlabelledPredsMetric <bfbrain.AL_Metrics.UnlabelledPredsMetric>` inherits from this class.
* :class:`UnlabelledPredsMetric <bfbrain.AL_Metrics.UnlabelledPredsMetric>`: These metrics are a special case of :class:`ModelMetric <bfbrain.AL_Metrics.ModelMetric>` that hold an internal set of unlabelled input points, so that the predictions of the model along these points can be tracked. A fully implemented metric of this type is :class:`UnlabelledDeltaF <bfbrain.AL_Metrics.UnlabelledDeltaF>`.

For metrics that inherit from any of these types except :class:`PoolMetric <bfbrain.AL_Metrics.PoolMetric>`, the procedure to implementing a metric of this type can be as simple as implementing 
the abstract :meth:`performance_check <bfbrain.AL_Metrics.ALMetric.performance_check>` method to return whatever value(s) the user wishes to record, ensuring that the arguments for their implementation
match the method's arguments in the documentation of the appropriate parent class. If :meth:`performance_check <bfbrain.AL_Metrics.ALMetric.performance_check>` returns a more complex structure than a single
number, then it will likely be necessary to overwrite some other methods of :class:`ALMetric <bfbrain.AL_Metrics.ALMetric>`. For example, a user may wish to overwrite 
:meth:`perf_message <bfbrain.AL_Metrics.ALMetric.perf_message>` in order to customize the message(s) which appear when printing the results of the metric to the output.txt file (as discussed in :ref:`Performance Metrics <metrics>`),
or overwrite :meth:`get_metric <bfbrain.AL_Metrics.ALMetric.get_metric>` and :meth:`get_legend <bfbrain.AL_Metrics.ALMetric.get_legend>` in order to alter the plotting logic used in :meth:`plot_metric <bfbrain.AL_Metrics.ALMetric.plot_metric>`.
It is highly recommended to inspect the documentation of these methods, as well as examples in subclasses where they are overwritten, such as :class:`ValidationFScore <bfbrain.AL_Metrics.ValidationFScore>`, before attempting
to create metrics which record multicomponent data structures.

Finally, implementing custom performance metrics comes with the same warning as implementing custom oracle functions: Because saving and loading of the :class:`BFBLearner <bfbrain.BFB_Learner.BFBLearner>` class relies
on the pickle module, any attempt to load a :class:`BFBLearner <bfbrain.BFB_Learner.BFBLearner>` with a custom performance metric must be done from a module with top-level access to the customized metric class,
as well as all nonstandard attributes within the class.

.. [1] Gal, Yarin, Jiri Hron, and Alex Kendall. "Concrete dropout." Advances in neural information processing systems 30 (2017).

.. [2] Altschuler, Michael, and Michael Bloodgood. "Stopping active learning based on predicted change of f measure for text classification." 2019 IEEE 13th International Conference on Semantic Computing (ICSC). IEEE, 2019.