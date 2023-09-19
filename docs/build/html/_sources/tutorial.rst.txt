.. _tutorial:

Tutorial and User Guide
=======================

Here we present a more in-depth step-by-step detailed guide to implementing a BFBrain analysis than in :doc:`usage`, including a detailed breakdown of each step and 
discussion of advanced features and finer usage points that are less well-described as part of :doc:`the reference documentation <modules>`. After following this tutorial, a user 
should be able to use all of BFBrain's features and rapidly and produce highly accurate approximations of scalar bounded-from-below conditions for arbitrary renormalizable scalar 
potentials with minimal up-front effort.

Before beginning, it is useful to establish some familiarity with the basics of active learning as used in BFBrain. Much of this information is repeated in our paper introducing BFBrain [1]_,
and so a reader familiar with the contents of that work can readily skip straight to :ref:`step one of the tutorial <oracle>`.

BFBrain's purpose is to create an efficiently-evaluated neural network classifier that can serve as a good approximation to scalar bounded-from-below conditions in parameter space scans of BSM theories
where it is computationally expensive to evaluate boundedness-from-below numerically and exact symbolic expressions do not exist. It creates this classifier via *active learning*, following these fundamental
steps:

#. A *classifier* is trained to convergence on a set of training points in the scalar model's parameter space labelled as bounded-from-below or not by some computationally expensive *oracle*

#. A large number of new unlabelled points are proposed to the classifier as possible new additions to the training set. 

#. The pool of candidate points are scored according to some *query strategy* which estimates the degree of uncertainty that the neural network has about its prediction for a given point. The highest-scoring (most uncertain) points are labelled by the oracle and added to the training data.

#. A new classifier is trained to convergence on the training set, which now features all of the old training data as well as the points that were chosen based on the query strategy in the previous step.

#. Steps 2-4 are repeated for a prespecified number of cycles (active learning iterations).

After completing training, the classifier produced by BFBrain is saved in a portable format as a Tensorflow model, which can be loaded and used for predictions on points in scans of parameter space on any device
with Tensorflow installed. Because BFBrain's classifiers are Bayesian neural networks approximated by concrete dropout [2]_, useful metrics of multiple sources of uncertainty can be extracted from their outputs. 
These uncertainty metrics can in turn be used to estimate the reliability of the neural network's predictions on unknown data, improving the utility of an already highly performant classifier.

.. toctree::
   :maxdepth: 2
   :caption: Tutorial:

   data_manager
   classifier
   training
   analysis

.. [1] Wojcik, George. In Preparation [arXiv:2309.XXXXX]

.. [2] Gal, Yarin, Jiri Hron, and Alex Kendall. "Concrete dropout." Advances in neural information processing systems 30 (2017).
