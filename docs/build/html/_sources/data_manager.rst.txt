.. _oracle:

Tutorial Step 1: Oracle and Data Generation
===========================================

.. _sympypotentials:

Writing the Potential Function
------------------------------

The first step to any BFBrain analysis is writing down the scalar potential which
the user wishes to analyze. In :ref:`Quickstart <quickstart>`, this method was written as

.. code-block:: python
    
    import sympy as sym
    from sympy import I
    from sympy.physics.quantum.dagger import Dagger

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

This example exhibits some key requirements for the oracle. First, it must have the same signature
as our example function-- it will take two sympy.Array() symbols-- the first (in this case phi) representing
the independent real parameters that specify a unique vev in the model. The second (in this case lam)
representing the set of independent real quartic coupling coefficients that uniquely specify a scalar potential.

The user-specified potential function will then output a SymPy expression which BFBrain will process into a numerical function for analysis.
For more examples of scalar potentials implemented in BFBrain-compatible manner, we refer the reader to :mod:`the Potential_Functions module <bfbrain.Potential_Functions>`

Potential Pitfalls
++++++++++++++++++

There are several important characteristics to keep in mind when specifying the SymPy potential function:

All components of phi and lam must be real. BFBrain assumes they are real when producing a numeric function. As shown in our example, arbitrary complex quartic coupling coefficients can be 
split into real and imaginary components and parameterized by lists of real numbers to satisfy this requirement.

Terms below fourth order in the vev's should not be included in the potential-- these have no bearing on strict boundedness-from below.

The potential must be linear in the quartic coefficients-- so in our example, V_2HDM(phi, lam1 + lam2) == V_2HDM(phi, lam1) + V_2HDM(phi, lam2),
or the resulting bounded-from-below parameter space will not be convex, which BFBrain depends on.

The potential must be written as a quartic real polynomial in the vev components-- there can be no decomposition of the scalar vev into, for example,
an angular parameterization.

In the final SymPy expression, any matrix operations such as traces or dot products must be
explicitly evaluated-- this is usually accomplished with sympy.simplify and/or sympy.expand
on individual quartic terms in the potential. Otherwise BFBrain's attempt to convert the expression 
into a numerical function will throw an exception. In practice this can be readily checked by
direct inspection of the output of the potential function. In our example, we can write

.. code-block:: python

    phisym = sym.Array(sym.symbols('phi:5', real = True))
    lamsym = sym.Array(sym.symbols('lambda:10', real = True))
    V_2HDM(phisym, lamsym)

which gives the expression

.. math::

    &\frac{\lambda_0}{2} + \frac{\lambda_1}{2} \bigg( \sum_{i = 1}^4 \phi_i^2 \bigg)^2 + \lambda_2 \phi_0^2 \bigg( \sum_{i = 1}^4 \phi_i^2 \bigg) + \lambda_3 \phi_0^2 (\phi_2^2 + \phi_4^2) + \lambda_4 \phi_0^2 (\phi_2^2 - \phi_4^2)\\
    &- 2 \lambda_5 \phi_0^2 \phi_2 \phi_4 + 2 \lambda_6 \phi_0^3 \phi_2 - 2 \lambda_7 \phi_0^3 \phi_4 + 2 \lambda_8 \phi_0 \phi_2 \bigg( \sum_{i = 1}^4 \phi_i^2 \bigg) - 2 \lambda_9 \phi_0 \phi_4 \bigg( \sum_{i = 1}^4 \phi_i^2 \bigg), \nonumber

which we can see avoids any implicit matrix computations.


.. _datamanager:

DataManager
-----------

Once a SymPy expression for the potential is written, the next step of the analysis is
converting the SymPy expression into a form which BFBrain can use to create an oracle
function to label training and test examples. This is done by instantiating the
:class:`DataManager <bfbrain.Data_Manager.DataManager>` class. It is **highly** recommended
to instantiate this class using :meth:`from_func <bfbrain.Data_Manager.DataManager.from_func>`.

For a full documentation of all of the relevant methods of the DataManager, we refer the reader
to the documentation. In summary, the :class:`DataManager <bfbrain.Data_Manager.DataManager>` class:

* Generates new random samples in the space of quartic coefficients by uniformly sampling from the surface of the unit hypersphere in this space (since boundedness-from-below of a potential is invariant under positive rescaling of the quartic potential coefficients, the unit hypersphere represents all possible sets of quartic coupling coefficients we would need to label).

* Applies a (generally computationally expensive) oracle function which labels samples as bounded-from-below  or not for training or testing. Produces an easily stored and manipulable :class:`np_data <bfbrain.Data_Manager.np_data>` object to contain labelled data.

* Converts :class:`np_data <bfbrain.Data_Manager.np_data>` objects into Tensorflow datasets for rapid training and prediction.

The DataManager class is where a user specifies an oracle function (which labels training examples).
BFBrain has a :ref:`default oracle function <defaultoracle>` specified, but retains support for
:ref:`user-defined oracle functions <customoracle>`.

.. _defaultoracle:

Using the Default Oracle
++++++++++++++++++++++++

The simplest (and almost always recommended) choice for an oracle when implementing the DataManager class is to
use the default oracle included in BFBrain, :meth:`label_func <bfbrain.Jax_Oracle.label_func>`, as was done in
in :ref:`Quickstart <quickstart>`. :meth:`label_func <bfbrain.Jax_Oracle.label_func>` estimates if a particular
set of quartic potential coefficients is bounded-from-below by locally minimizing the quartic part of the
scalar potential with respect to the vev a large number of times at random starting vev configurations.
By increasing the number of local minimizations, the algorithm becomes arbitrarily accurate.

The number of local minimizations that :meth:`label_func <bfbrain.Jax_Oracle.label_func>` performs is controlled
by the keyword argument, niter. This keyword argument (along with all other keyword arguments accepted by 
:meth:`label_func <bfbrain.Jax_Oracle.label_func>`) can be specified in the :class:`DataManager <bfbrain.Data_Manager.DataManager>`
constructor :meth:`from_func <bfbrain.Data_Manager.DataManager.from_func>` and will be passed to the oracle from there.
As an example, if we want the oracle to perform 100 local minimizations, then the appropriate DataManager constructor for the 
2HDM potential we specified in :ref:`Writing the Potential Function <sympypotentials>`, which has 5 real parameters in its 
vev configuration and 10 real quartic potential coefficients, will be 

.. code-block:: python

    from bfbrain import DataManager

    dm = DataManager.from_func(V_2HDM, 5, 10, niter = 100)

BFBrain also has a pre-implemented method for testing the performance of different numbers of local minimizations in order
to identify an oracle with a minimal amount of noise (that is, incorrect labels) for use in training. In our studies
in [1]_, we have found that once the oracle achieves highly robust labels (that is, it will label the same
inputs identically every time), the effect of label noise on the performance of a BFBrain classifier is generally
quite small. :meth:`test_labeller <bfbrain.Jax_Oracle.test_labeller>` determines robustness by repeatedly labelling the same (large)
sample of quartic potential coefficients, while increasing the number of local minimizations that the oracle performs by
a fixed interval with every labelling attempt. Once the method has found that identical labels are returned for a 
user-specified number of consecutive labelling attempts, it returns the number of local minimizations that it used
for the first attempt in this streak. In other words, the test estimates the minimum number of local minimizations such that
increasing the number of local minimizations does not change the labels on a large sample set. The robustness test is easily performed
from the :class:`DataManager <bfbrain.Data_Manager.DataManager>` object using :meth:`check_labeller <bfbrain.Data_Manager.DataManager.check_labeller>`. For example, to perform a test
on 100000 sets of quartic coupling coefficients sampled uniformly from the surface of the unit hypersphere, where the number of local
minimizations is increased by 50 with each labelling attempt, and robustness is defined as 5 consecutive consistent labelling attempts,
we need only write

.. code-block:: python

    min_niter = dm.check_labeller(100000, niter_step = 50, count_success = 5)

We direct the reader to the function's documentation for more information on usage and customization of :meth:`test_labeller <bfbrain.Jax_Oracle.test_labeller>`.

.. _customoracle:

Advanced Usage: Customized Oracles
++++++++++++++++++++++++++++++++++

BFBrain also supports customized oracles. The creation and use of a customized oracle is considerably more involved than usage of
:ref:`the default oracle <defaultoracle>`, and is only recommended in highly specialized use cases, especially since the default
oracle noise has exhibited a limited impact on the BFBrain classifier performance when labels are sufficiently robust for the various
complicated scalar potentials discussed in [1]_. If the user wishes to rely on the default oracle, there is no need for them to read this
section.

A custom oracle must process a numerical function produced by the :class:`DataManager <bfbrain.Data_Manager.DataManager>` object
from the SymPy expression for the scalar potential. This numerical function may be a NumPy, SciPy, Jax, or Tensorflow function.
To specify which of these functions the oracle should be passed, one must specify the keyword lambdify_mode in :meth:`the DataManager constructor <bfbrain.Data_Manager.DataManager.from_func>`
as 'numpy', 'scipy', 'jax', or 'tensorflow'.

Regardless of its format, the numerical function will return a tuple consisting of the function value and its gradient with respect
to the scalar vev. A customized oracle must then be a function with the signature,

(func : callable, phi_len : int, polar : bool, rng : numpy.random.Generator, lam : np.array(np.float32, np.float32), \*\*kwargs) :math:`\rightarrow` np.array(bool)

The input parameters (except for \*\*kwargs) are identical to their equivalents in :meth:`the default oracle's label_func method <bfbrain.Jax_Oracle.label_func>`, with the possible exception that
func may be a NumPy, SciPy, or Tensorflow function rather than a Jax Numpy function. As a simple (if unrealistic) example of implementing a customized oracle, we consider an alternate oracle which takes
a NumPy function for the numerical potential function, and labels a point as bounded-from-below if it is bounded-from-below assuming that only one of the two Higgs fields attains a nonzero vev at one time.

.. code-block:: python

    def label_fn(func, phi_len, polar, rng, lam, **kwargs):
        # Assuming our numerical function will come from the V_2HDM function we specified earlier,
        # we specify inputs which correspond to only one of the two Higgs fields having a nonzero vev.
        input1 = np.array([1.,0.,0.,0.,0.])
        input2 = np.array([0., 0., 1., 0., 0.])
        return np.array([func(input1, x)[0] > 0 and func(input2, x)[0] > 0 for x in lam])

    dm = DataManager.from_func(V_2HDM, 5, 10, lambdify_mode = 'numpy', label_fn = label_fn)

At this point, label_fn will be used as the oracle function instead of the default oracle-- in the case of our example leading to significantly less accurate results!
We can also customize the function called by :meth:`DataManager.check_labeller <bfbrain.Data_Manager.DataManager.check_labeller>` in the same manner, this time by writing a new function to replace :meth:`test_labeller <bfbrain.Jax_Oracle.test_labeller>`.
Obviously the custom oracle we have implemented here has no hyperparameters to check. However, we should see that it will label approximately 25% of all 
uniformly sampled inputs as bounded-from-below, since it labels every point where the two coefficients :math:`\lambda_0` and :math:`\lambda_1` are both positive as bounded-from-below.
So, we can adapt our DataManager's test to check classification labels for a uniformly sampled set of points. An example implementation of this is given below:

.. code-block:: python

    def label_fn(func, phi_len, polar, rng, lam, **kwargs):
        # Assuming our numerical function will come from the V_2HDM function we specified earlier,
        # we specify inputs which correspond to only one of the two Higgs fields having a nonzero vev.
        input1 = np.array([1.,0.,0.,0.,0.])
        input2 = np.array([0., 0., 1., 0., 0.])
        return np.array([func(input1, x)[0] > 0 and func(input2, x)[0] > 0 for x in lam])

    #Now we also specify a new function label_check, which will replace the default method called by DataManager.check_labeller
    def label_check(func, phi_len, polar, rng, lam, label_kwargs : dict, **kwargs):
        #Notice that label_check must take the same arguments as label_fn, but can return any type and may take additional keyword arguments.
        n_inputs = len(lam)
        return np.count_nonzero(label_fn(func, phi_len, polar, rng, lam, **label_kwargs)) / n_inputs

    dm = DataManager.from_func(V_2HDM, 5, 10, lambdify_mode = 'numpy', label_fn = label_fn, label_check = label_check)

Now, for example, when dm.check_labeller(100000) is called, the method we have defined above will be called on a set of 100000 uniformly sampled sets of quartic coefficients, instead of
:meth:`test_labeller <bfbrain.Jax_Oracle.test_labeller>`.

A potential pitfall when implementing customized oracles and oracle tests can arise when saving and loading the resulting :class:`DataManager <bfbrain.Data_Manager.DataManager>`, which can occur
often during training. Because the :class:`DataManager <bfbrain.Data_Manager.DataManager>` object is saved using pickle, in order to load an instance of the class which has a custom oracle or 
oracle test function must have the SAME custom function be accessible from the top level of the module-- otherwise the program will throw an exception.

.. [1] Wojcik, George. In Preparation [arXiv:2309.XXXXX]