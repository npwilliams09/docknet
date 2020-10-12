from .graphBlocks import *
from tensorflow.keras.layers import Input
from .combiner import *
from .twod_conv import *


def docknet(config, features):
    nodesA = Input(shape=(None, features), name="Input_A")
    adjA = Input(shape=(None, None), name="adj_A")
    nodesB = Input(shape=(None, features), name="Input_B")
    adjB = Input(shape=(None, None), name="adj_B")

    branch = oneDgraph(input_shape=(None, features), filters=config["filters"], blocks=config["graph_blocks"], \
                       style=config["graph_style"], drop_rate=config["drop_rate"])

    procA = branch([nodesA, adjA])
    procB = branch([nodesB, adjB])

    Combiner = getCombiner(config['combine_style'], filters=config['filters'])

    combined = Combiner([procA, procB])

    combined = wave_block2D(combined, filters=config["filters"],drop_rate=config["drop_rate"],
                            squeeze_excite=config["squeeze_excite"])

    for i in range(config["wave_blocks"]-1):
        combined = wave_block2D(combined, filters=config["filters"], drop_rate=config["drop_rate"],
                                squeeze_excite=config["squeeze_excite"])

    combined = OutputLayer(config["drop_rate"])(combined)

    return Model([nodesA, adjA, nodesB, adjB], combined)