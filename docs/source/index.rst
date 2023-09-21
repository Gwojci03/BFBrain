.. BFBrain documentation master file, created by
   sphinx-quickstart on Sat Sep 16 19:00:32 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BFBrain's documentation!
===================================

**BFBrain** is a Python library for training Bayesian neural networks to
approximate bounded-from-below conditions for multiscalar potentials in
quantum field theory. In many multiscalar theories, determining if a given
point in parameter space is bounded-from-below is both prohibitively
computationally expensive to do numerically and intractable to resolve
into symbolic conditions. BFBrain works to resolve this issue by encoding
approximate bounded-from-below conditions in a Bayesian neural network,
which can be used as portable, efficiently-computable boundedness-from-below
conditions in parameter space scans for BSM physics studies. In [1]_, our paper
introducing BFBrain, we have found that this methodology can significantly
outperform more conventional methods of approximating bounded-from-below conditions
while being theoretically applicable to *any* renormalizable scalar potential
and exhibiting robust uncertainty determination which can inform a user when 
a model's predictions should be trusted. The methods of the BFBrain package are designed 
to allow rapid implementation of analyses with only a few lines of code, but remain 
open to a high degree of user customization.

Check out the :doc:`usage` section for further information, including how to
:ref:`install <installation>` the project and a simple example of how to get started with :ref:`Quickstart <quickstart>`.
For a more detailed introduction to the software with a step-by-step tutorial, see :doc:`tutorial`.
For a full API reference, see :doc:`modules`

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   tutorial
   modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. [1] G.N. Wojcik. "BFBrain: Scalar Bounded-from-Below Conditions with Bayesian Active Learning" [`arXiv:2309.10959 <https://arxiv.org/abs/2309.10959>`_ [hep-ph]]
