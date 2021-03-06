import tensorflow as tf


class Encoder:

    def __init__(self, output_shape, name=None, end=False):
        """ Initialize the layer with an output shape, and an optional name. """
        # self.layer = tf.layers.Dense(output_shape, name=name, activation=tf.nn.leaky_relu)
        self.output_shape = output_shape
        self.name = name
        self.end = end

    def __call__(self, input):
        """ Wrapper around self.layer.call(input) for simplicity. """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.activation = tf.nn.leaky_relu
            self.initializer = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)
            self.regularizer = tf.nn.l2_loss

            self.kernel = tf.get_variable("kernel", [input.shape[1], self.output_shape], initializer=self.initializer,
                                          dtype=tf.float32, regularizer=self.regularizer)
            self.bias = tf.get_variable("bias", [self.output_shape], initializer=self.initializer, dtype=tf.float32)
            self.output = self.activation(tf.matmul(input, self.kernel) + self.bias)
            # self.output = tf.matmul(input, self.kernel) + self.bias
            #
            # if self.end:

        return self.output
