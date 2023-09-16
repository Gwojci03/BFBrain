# BFBrain

Use active learning to determine the bounded-from-below region in parameter space of a multiscalar potential in quantum field theory.

## Installation

```bash
$ pip install BFBrain
```

## Usage

For a scalar potential specified by a SymPy expression, `BFBrain` can be used to train a Bayesian neural network (approximated with Monte Carlo dropout) which classifies sets of quartic coefficients of this scalar potential
as bounded-from-below or not, for parameter space scans of BSM physics models. An example of a typical training script is

```python
import sympy as sym
from sympy import I
from sympy.physics.quantum.dagger import Dagger

from BFBrain import DataManager, BFBLearner, ValidationFScore, PoolDeltaF

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
metrics = [ValidationFScore(), PoolDeltaF()]

# Initialize the active learning object.
AL = BFBLearner.init_for_first_run(dm, 5, 128, metrics, 1000)
#  Run the active learning loop.
AL.AL_loop('saved_AL', score_fn = 'BALD', nstop = 20)
```

Once trained, the neural network created above can be used to make predictions as

```python
from BFBrain import BFBLearner, MC_call_fast, BALD, Max_Entropy, Variation_Ratios
import tensorflow as tf
from numpy.random import default_rng

# Generate 100000 random sets of 2HDM quartic coefficients, all of which have values between -5 and 5.
rng = default_rng(12345)
rand_lams = rng.uniform(-5., 5., (100000, 10))

# Load the neural network produced by the last script
model = tf.keras.models.load_model('saved_AL/model')
# Can alternatively be loaded as...
# model = BFBLearner.from_file('saved_AL').model

# Make predictions on rand_lams using Monte Carlo dropout with 100 forward passes through the network.
preds = (MC_call_fast(model, rand_lams, n_trials = 100) >= 0.5).numpy()
# preds is now a NumPy array where the kth entry is True if the BFBrain model predicts the kth element of rand_lams represents a bounded-from-below potential, and False otherwise
# We can evaluate uncertainty metrics on the predictions, like mutual information, Shannon entropy, and variation ratios as well.
mutual_info = (BALD(model, rand_lams, n_trials = 100) >= 0.5).numpy()
entropy = (Max_Entropy(model, rand_lams, n_trials = 100) >= 0.5).numpy()
var_ratios = (Variation_Ratios(model, rand_lams, n_trials = 100) >= 0.5).numpy()
```

## Contributing

Interested in contributing? Check out the contributing guidelines. 
Please note that this project is released with a Code of Conduct. 
By contributing to this project, you agree to abide by its terms.

## License

`BFBrain` was created by George Wojcik. It is licensed under the terms
of the MIT license.

## Credits



[Github-flavored Markdown](https://guides.github.com/features/mastering-markdown/)
to write your content.
