import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Activation
from tensorflow_addons.layers import InstanceNormalization


class matmulCombiner(Layer):
    def call(self, inputs):
        a = tf.transpose(inputs[0], perm=[0, 2, 1, 3])
        b = tf.transpose(inputs[1], perm=[0, 2, 3, 1])
        multiplied = tf.matmul(a, b)
        return tf.transpose(multiplied, perm=[0, 2, 3, 1])


class concCombiner(Layer):
    def __init__(self, filters):
        self.resizeConv = Conv2D(filters=filters, kernel_size=1, padding='same')
        self.norm = InstanceNormalization()
        self.nonlinear = Activation('swish')

    def call(self, inputs):
        # transpose each tensor so relevant channels are last
        a = tf.transpose(inputs[0], perm=[0, 2, 1, 3])
        b = tf.transpose(inputs[1], perm=[0, 2, 3, 1])
        # get shapes
        aShape = tf.shape(a)[2]
        bShape = tf.shape(b)[3]
        # copy and paste vectors along the other tensors highest dimension so they are equal size
        b = tf.tile(b, [1, 1, aShape, 1])
        a = tf.tile(a, [1, 1, 1, bShape])
        # Concat and transpose back to 'normal'
        comb = tf.concat([a, b], axis=1)
        comb = tf.transpose(comb, perm=[0, 2, 3, 1])
        # pass through 1D conv to return original dimensions
        comb = self.resizeConv(comb)
        comb = self.norm(comb)
        return self.nonlinear(comb)


def getCombiner(style='matmul', filters=64):
    if style == 'conc':
        return concCombiner(filters)
    elif style == 'matmul':
        return matmulCombiner()
    return matmulCombiner  # fallback
