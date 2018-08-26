import tensorflow as tf

class Decoder:
    def __init__(self, output_shape, name = None):
        """ Initialize the layer with an output shape, and an optional name. """
        self.layer = tf.layers.Dense(output_shape, name=name, activation=tf.nn.sigmoid)

    def __call__(self, input):
        """ Wrapper around self.layer(input). """
        return self.layer(input)


    def reverse_transform(self, sess, input):
        """
        :param sess: The open TF session for used for the call
        :param input: The input for the reverse transform
        :return: A tensor of shape output_shape
        """
        return sess.run(self.layer, {self.input : input})