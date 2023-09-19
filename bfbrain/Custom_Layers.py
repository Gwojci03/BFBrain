"""This module contains the various neural network layers used by BFBrain.
"""

import tensorflow as tf

# A custom neural network preprocessing layer which projects any input quartic coefficients onto the unit hypersphere. 
class HypersphereProjectionLayer(tf.keras.layers.Layer):
    """A custom neural network preprocessing layer which projects any 
    input quartic coefficients onto the unit hypersphere.
    """
    def __init__(self):
        super(HypersphereProjectionLayer, self).__init__()
    def build(self, input_shape):
        return
    
    def get_config(self):
        config = super().get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, inputs):
        return inputs / (tf.norm(inputs, axis = 1, keepdims = True))


def get_weight_regularizer(N, l=1e-2, tau=0.1):
    """Determines the weight decay constant which should be applied in the 
    loss function with a given precision, prior length scale, and number 
    of training data points.

    Parameters
    ----------
    N : int
        The number of data points in the training data.

    l : float, default=1e-2

    tau : float, defulat = 0.1
        neural network precision. For classification networks this is just 
        set to 1.

    Returns
    -------
    float
    """
    return l**2 / (tau * N)


def get_dropout_regularizer(N, tau=0.1, cross_entropy_loss=False):
    """Controls the regularization term associated with the entropy
    of the cells' dropout probabilities.

    Parameters
    ----------
    N : int
        The number of data points in the training data.

    tau : float, defulat = 0.1
        neural network precision. For classification networks this is just 
        set to 1.

    cross_entropy_loss : bool, default=False
        Should be True if the loss function is cross entropy (so the 
        neural network is a classifier), and False otherwise.
        
    Returns
    -------
    float
    """
    reg = 1 / (tau * N)
    if not cross_entropy_loss:
        reg *= 2
    return reg

class ConcreteDenseDropout(tf.keras.layers.Dense):
    """Code for the implementation of concrete dropout. Based 
    heavily on https://github.com/aurelio-amerio/ConcreteDropout, 
    a Tensorflow 2.0 implementation of the concrete dropout algorithm 
    described in arXiv:1705.07832. Modified from that implementation in 
    order to save the model more easily at the expense of some 
    flexibility. IMPORTANT: these layers perform dropout BEFORE the 
    wrapped operation.
    """
    def __init__(self, units, weight_regularizer=1e-6, dropout_regularizer=1e-5, init_min = 0.1, init_max = 0.1, temperature = 0.1, **kwargs):
        super().__init__(units, **kwargs)
        self.weight_regularizer = tf.keras.backend.cast_to_floatx(weight_regularizer)
        self.dropout_regularizer = tf.keras.backend.cast_to_floatx(dropout_regularizer)
        self.supports_masking = True
        self.p_logit = None
        self.init_min = (tf.math.log(init_min)-tf.math.log(1.-init_min))
        self.init_max = (tf.math.log(init_min)-tf.math.log(1.-init_max))
        self.temperature = temperature

    def build(self, input_shape = None):
        self.input_spec = tf.keras.layers.InputSpec(shape = input_shape)
        super().build(input_shape)
        self.p_logit = self.add_weight(shape = (1,), initializer=tf.keras.initializers.RandomUniform(self.init_min, self.init_max), name = 'p_logit', trainable = True)
        self.p = tf.nn.sigmoid(self.p_logit[0])
        self.input_dim = input_shape[-1]

    def _get_noise_shape(self, inputs):
        input_shape = tf.shape(inputs)
        return input_shape

    def spatial_concrete_dropout(self, x, p):
        eps = tf.keras.backend.cast_to_floatx(tf.keras.backend.epsilon())
        noise_shape = self._get_noise_shape(x)
        unif_noise = tf.keras.backend.random_uniform(shape = noise_shape)
        drop_prob = (tf.math.log(p + eps) - tf.math.log1p(eps - p) + tf.math.log(unif_noise+eps) - tf.math.log1p(eps - unif_noise))
        drop_prob = tf.math.sigmoid(drop_prob / self.temperature)
        random_tensor = 1. - drop_prob

        retain_prob = 1.-p
        x *= random_tensor
        x /= retain_prob
        return x
    
    def call(self, inputs, training = None):
        p = tf.nn.sigmoid(self.p_logit)
        weight = self.kernel
        bias = self.bias
        kernel_regularizer = self.weight_regularizer * tf.reduce_sum(tf.square(weight))/ (1. - p)
        if self.use_bias:
            bias_regularizer = self.weight_regularizer * tf.reduce_sum(tf.square(bias))
        else:
            bias_regularizer = 0.
        dropout_regularizer = p * tf.math.log(p) + (1.-p)*tf.math.log1p(-p)
        dropout_regularizer *= self.dropout_regularizer * self.input_dim
        regularizer = tf.reduce_sum(kernel_regularizer + dropout_regularizer + bias_regularizer)
        self.add_loss(regularizer)
        return tf.keras.backend.in_train_phase(super().call(self.spatial_concrete_dropout(inputs, p)), super().call(inputs), training = training)
    
    def get_config(self):
        config = super().get_config()
        config.update({"units":self.units, "weight_regularizer":float(self.weight_regularizer), "dropout_regularizer":float(self.dropout_regularizer), "init_min":float(self.init_min), "init_max":float(self.init_max), "temperature":float(self.temperature)})
        return config