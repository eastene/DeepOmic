import tensorflow as tf


class Encoder:
    def __init__(self, output_shape, name=None):
        """ Initialize the layer with an output shape, and an optional name. """
        # self.layer = tf.layers.Dense(output_shape, name=name, activation=tf.nn.leaky_relu)
        self.output_shape = output_shape
        self.name = name

    def __call__(self, input):
        """ Wrapper around self.layer.call(input) for simplicity. """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.activation = tf.nn.leaky_relu
            self.initializer = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)
            self.kernel = tf.get_variable("kernel", [input.shape[1], self.output_shape], initializer=self.initializer,
                                          dtype=tf.float32)
            self.bias = tf.get_variable("bias", [self.output_shape], initializer=self.initializer, dtype=tf.float32)
            # self.output = self.activation(tf.matmul(input,self.kernel) + self.bias)
            self.output = self.activation(tf.matmul(input, self.kernel) + self.bias)
        return self.output
        # return self.layer(input)
