from tensorflow.keras.layers import Dropout, Conv2D, Activation, Input, Reshape, Layer, \
    Add, GlobalAveragePooling2D, Reshape, Multiply
from tensorflow_addons.layers import InstanceNormalization


class SqueezeExcite(Layer):
    def __init__(self, input_sz, se_ratio=16):
        super(SqueezeExcite, self).__init__()
        self.pool = GlobalAveragePooling2D()
        targShape = (1, 1, input_sz)
        self.reshape = Reshape(targShape)
        reduced_filters = int(input_sz / se_ratio)
        self.squeeze = Conv2D(reduced_filters, 1, activation="swish", padding="same")
        self.excite = Conv2D(input_sz, 1, activation="swish", padding="same")
        self.mult = Multiply()

    def call(self, x):
        se_tensor = self.pool(x)
        se_tensor = self.reshape(se_tensor)
        se_tensor = self.squeeze(se_tensor)
        se_tensor = self.excite(se_tensor)
        return self.mult([se_tensor, x])


class ConvBlock(Layer):
    def __init__(self, input_sz, dilation=(1, 1), drop_rate=0.0, squeeze_excite=False):
        super(ConvBlock, self).__init__()
        self.conv1 = Conv2D(filters=input_sz, kernel_size=1, padding='same')
        self.conv2 = Conv2D(filters=input_sz/2, kernel_size=1, padding='same', dilation_rate=dilation)
        self.conv3 = Conv2D(filters=input_sz, kernel_size=1, padding='same')
        self.activation = Activation('swish')
        self.norm1 = InstanceNormalization()
        self.norm2 = InstanceNormalization()
        self.norm3 = InstanceNormalization()
        self.se = squeeze_excite
        if self.se:
            self.squeeze_excite = SqueezeExcite(input_sz)
        self.dropout = Dropout(drop_rate)
        self.add = Add()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        if self.se: #squeeze excite
            x = self.squeeze_excite(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.dropout(x)
        return self.add([x, inputs])


def wave_block2D(x, filters, drop_rate=0.0, squeeze_excite=False, waves=4):
    dilation_rates = [2 ** i for i in range(waves)]  # e.g. waves = 3 -> [2**0, 2**1, 2**2, 2**3] = [1,2,4,8]
    for d in dilation_rates:
        x = ConvBlock(filters, dilation=(d, d), drop_rate=drop_rate, squeeze_excite=squeeze_excite)(x)
    return x


class OutputLayer(Layer):
    def __init__(self, drop_rate):
        super(OutputLayer, self).__init__()
        self.drop = Dropout(drop_rate)
        self.conv = Conv2D(filters=1, kernel_size=1, padding="same")
        self.activate = Activation("sigmoid", name="Output", dtype='float32')

    def call(self, inputs):
        x = self.drop(inputs)
        x = self.conv(x)
        return self.activate(x)
