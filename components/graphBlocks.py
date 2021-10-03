from tensorflow.keras.layers import Conv1D, Activation, Dropout, Add, Layer, Reshape, Input
from tensorflow.keras.models import Model
from tensorflow_addons.layers import InstanceNormalization
from spektral.layers import GCNConv, GraphSageConv


class GraphBlock(Layer):
    def __init__(self, filters=128, style='gcn', drop_rate=0.0, **kwargs):
        super(GraphBlock, self).__init__(**kwargs)
        graph_conv = getGraphConv(style)
        self.gconv = graph_conv(filters)
        self.norm = InstanceNormalization()
        self.activate = Activation('swish')
        self.dropout = Dropout(drop_rate)
        self.add = Add()

    def call(self, inputs):
        x = self.gconv(inputs)
        x = self.norm(x)
        x = self.activate(x)
        x = self.dropout(x)
        return self.add([x, inputs[0]])

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'gconv': self.gconv,
            'norm': self.norm,
            'activate': self.activate,
            'dropout': self.dropout,
            'add': self.add,
        })
        return config


def getGraphConv(style):
    if style == 'gcn':
        return GCNConv
    elif style == 'sage':
        return GraphSageConv
    return GraphConv  # default option


def oneDgraph(input_shape, filters, blocks, style, drop_rate=0.0):
    features = Input(shape=(input_shape), name="node_scores")
    adj = Input(shape=(None, None), name="AdjMat")
    r = Conv1D(filters=filters, kernel_size=1, padding="same", name="conv_reshape")(features)
    r = InstanceNormalization()(r)
    r = Activation("swish")(r)

    x = GraphBlock(filters, style, drop_rate)([r, adj])
    for i in range(blocks - 1):
        x = GraphBlock(filters, style, drop_rate)([x, adj])

    #x = Reshape(target_shape=(-1, filters, 1), name="1D_to_2D")(x)
    return Model([features, adj], x, name="graphLayer")
