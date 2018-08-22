import tensorflow as tf

class Encoder:
    def __init__(self, input, output_shape, name = None, dropout=False):
        self.input = input
        #self.input = tf.placeholder(tf.float32, shape=[None, input_shape])
        if dropout == True:
            self.d_layer = tf.layers.dense(self.input, output_shape,name=name,activation=tf.nn.sigmoid)
            self.layer = tf.layers.dropout(self.d_layer)
        else:
            self.layer = tf.layers.dense(self.input, output_shape,name=name, activation=tf.nn.sigmoid)

    def transform(self, sess, input):
        """
        :param sess: The open TF session used for the call
        :param input: The input for the transform
        :return: A tensor of shape output_shape
        """
        return sess.run(self.layer,{self.input : input})