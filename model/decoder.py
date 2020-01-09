import tensorflow as tf


class Decoder:
    def __init__(self, output_shape, name=None, weights=None, end=False):
        """ Initialize the layer with an output shape, and an optional name. """
        self.output_shape = output_shape
        self.name = name
        self.weights = weights
        self.end = end

    def __call__(self, inputs, enc_weights):
        """ Wrapper around self.layer(input). """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.activation = tf.keras.activations.linear # tf.nn.leaky_relu
            self.initializer = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)

            self.kernel = tf.transpose(enc_weights)
            self.bias = tf.get_variable("bias", [self.output_shape], initializer=self.initializer, dtype=tf.float32)
            self.output = self.activation(tf.matmul(inputs, self.kernel) + self.bias)

        return self.output
