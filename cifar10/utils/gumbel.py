from keras.engine.topology import Layer
from keras import backend as K
import tensorflow as tf
import numpy as np

class Sample_Concrete(Layer):
    """
    Layer for sample Concrete / Gumbel-Softmax variables.

    """

    def __init__(self, tau0, k, **kwargs):
        self.tau0 = tau0
        self.k = k
        super(Sample_Concrete, self).__init__(**kwargs)

    def call(self, logits):
        # logits: [BATCH_SIZE, d]
        logits_ = K.expand_dims(logits, -2)  # [BATCH_SIZE, 1, d]

        batch_size = tf.shape(logits_)[0]
        d = tf.shape(logits_)[2]
        uniform = tf.random_uniform(shape=(batch_size, self.k, d),
                                    minval=np.finfo(tf.float32.as_numpy_dtype).tiny,
                                    maxval=1.0)

        gumbel = - K.log(-K.log(uniform))
        noisy_logits = (gumbel + logits_) / self.tau0
        samples = K.softmax(noisy_logits)
        samples = K.max(samples, axis=1)

        # Explanation Stage output.
        threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted=True)[0][:, -1], -1)
        discrete_logits = tf.cast(tf.greater_equal(logits, threshold), tf.float32)

        return K.in_train_phase(samples, discrete_logits)

    def compute_output_shape(self, input_shape):
        return input_shape


