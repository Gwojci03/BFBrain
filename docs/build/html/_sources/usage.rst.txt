Usage
=====

.. _installation:

Installation
------------

To use BFBrain, it is highly advised to work in an environment with both Tensorflow and 
Jax installed with GPU support. Consult Tensorflow and Jax's documentations for installation
instructions. Once this is done, BFBrain can be installed using pip:

.. code-block:: console

   python3 -m pip install BFBrain

.. _quickstart:

Quickstart
----------

BFBrain creates a Bayesian neural network approximated with Monte Carlo dropout which 
is trained to identify whether sets of quartic coefficients lead to potentials which are
bounded-from-below. It does this via supervised learning with a computationally expensive oracle
function which can numerically label bounded-from-below points. Since the oracle function
can be too computationally expensive to be practical, BFBrain uses active learning techniques
to reduce the number of explicit oracle labels it needs to achieve high performance.

The following code demonstrates a simple training script for the analysis of 
the most general Two-Higgs doublet model (2HDM) with BFBrain:

.. code-block:: python

   import sympy as sym
   from sympy import I
   from sympy.physics.quantum.dagger import Dagger

   from bfbrain import DataManager, BFBLearner, ValidationFScore, UnlabelledDeltaF

   # Write a SymPy function representing the scalar potential.
   def V_2HDM(phi, lam):
      Phi1 = sym.Matrix([0, phi[0]])
      Phi2 = sym.Matrix([phi[1] + I*phi[3], phi[2] + I*phi[4]])
      phi1sq = Dagger(Phi1).dot(Phi1)
      phi2sq = sym.simplify(Dagger(Phi2).dot(Phi2))
      phi12 = sym.simplify(Dagger(Phi1).dot(Phi2))
      phi21 = sym.simplify(Dagger(Phi2).dot(Phi1))

      QVec = (sym.Matrix([(phi1sq**2)/2, (phi2sq**2)/2,
                     phi1sq*phi2sq, phi12*phi21, 
                     (phi12**2 + phi21**2)/2,
                     I*(phi12**2 - phi21**2)/2,
                     phi1sq*(phi12 + phi21),
                     I*phi1sq*(phi12 - phi21),
                     phi2sq*(phi12 + phi21),
                     I*phi2sq*(phi12-phi21)])).applyfunc(sym.simplify)
      return QVec.dot(lam)

   # Initialize a DataManager object which will handle 
   # data generation and oracle labelling.
   dm = DataManager.from_func(V_2HDM, 5, 10, niter = 100)

   # Specify performance metrics we wish to keep track of during active learning
   # generate an unlabelled sample of 10^6 points for evaluation with the metric UnlabelledDeltaF
   unlabelled_lams = dm.create_random_lambdas(1000000, validation = True)

   # ValidationFScore tracks the F score on a labelled validation data set.
   # UnlabelledDeltaF tracks the estimated change in the F score on an 
   # unlabelled data set, computed from the stability of predictions over 
   #different iterations of active learning. 
   metrics = [ValidationFScore(), UnlabelledDeltaF(unlabelled_lams)]

   # Initialize the active learning object. This specifies that 
   # the neural network should have 5 hidden layers of 128 neurons each, and 
   # use 1000 randomly generated sets of quartic coefficients as the Initia
   # training sample (which will then grow with active learning)
   AL = BFBLearner.init_for_first_run(dm, 5, 128, metrics, 1000)

   #  Run the active learning loop.
   #  BFBrain is instructed to conduct 20 rounds of active learning
   #  using the BALD (Bayesian Active Learning by Disagreement)
   #  acquisition function, and save the resulting classifier, and
   #  then save the BFBLearner object (including the classifier)
   #  in a directory 'saved_AL'.
   AL.AL_loop('saved_AL', score_fn = 'BALD', nstop = 20)

The neural network can then be loaded and used in an analysis. Because the neural network uses Monte Carlo dropout to quantify uncertainty, performing many forward passes through the network with the
:meth:`MC_call_fast <bfbrain.Score_Functions.MC_call_fast>` function will enable rapid classification of unknown points as bounded from below or not, while other specialized methods can be used
to extract different uncertainty estimats. As an example:

.. code-block:: python

   from bfbrain import BFBLearner, MC_call_fast, BALD, Max_Entropy, Variation_Ratios
   import tensorflow as tf
   from numpy.random import default_rng

   # Generate 100000 random sets of 2HDM quartic coefficients,
   # all of which have values between -5 and 5.
   rng = default_rng(12345)
   rand_lams = rng.uniform(-5., 5., (100000, 10))

   # Load the neural network produced by the last script
   model = tf.keras.models.load_model('saved_AL/model')
   # Can alternatively be loaded as...
   # model = BFBLearner.from_file('saved_AL').model

   # Make predictions on rand_lams using Monte Carlo dropout with 
   # 100 forward passes through the network.
   preds = (MC_call_fast(model, rand_lams, n_trials = 100) >= 0.5).numpy()
   
   # preds is now a NumPy array where the kth entry is True if the BFBrain model 
   # predicts the kth element of rand_lams represents a bounded-from-below potential, 
   # and False otherwise. We can evaluate uncertainty metrics on the predictions,
   # like mutual information, Shannon entropy, and variation ratios as well.
   mutual_info = (BALD(model, rand_lams, n_trials = 100)).numpy()
   entropy = (Max_Entropy(model, rand_lams, n_trials = 100)).numpy()
   var_ratios = (Variation_Ratios(model, rand_lams, n_trials = 100)).numpy() 


