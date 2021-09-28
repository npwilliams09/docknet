from .graphBlocks import *
from tensorflow.keras.layers import Input
from .combiner import *
from .twod_conv import *
import tensorflow as tf


def single_docknet(config, features):
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

def quad_docknet(config, features):
    nodesA = Input(shape=(None, features), name="Input_A")
    adjA = Input(shape=(None, None), name="adj_A")
    nodesB = Input(shape=(None, features), name="Input_B")
    adjB = Input(shape=(None, None), name="adj_B")

    branch = oneDgraph(input_shape=(None, features), filters=config["filters"], blocks=config["graph_blocks"], \
                       style=config["graph_style"], drop_rate=config["drop_rate"])

    with tf.device_scope("/physical_device:GPU:0"):
        procA = branch([nodesA, adjA])

    with tf.device_scope("/physical_device:GPU:1"):
        procB = branch([nodesB, adjB])

    with tf.device_scope("/physical_device:GPU:2"):
        Combiner = getCombiner(config['combine_style'], filters=config['filters'])

        combined = Combiner([procA, procB])

        combined = wave_block2D(combined, filters=config["filters"],drop_rate=config["drop_rate"],
                                squeeze_excite=config["squeeze_excite"])
    with tf.device_scope("/physical_device:GPU:3"):
        for i in range(config["wave_blocks"]-1):
            combined = wave_block2D(combined, filters=config["filters"], drop_rate=config["drop_rate"],
                                    squeeze_excite=config["squeeze_excite"])

        combined = OutputLayer(config["drop_rate"])(combined)

    return Model([nodesA, adjA, nodesB, adjB], combined)


def docknet(config,features,quad_mode=False):
    if quad_mode:
        return quad_docknet(config,features)
    return single_docknet(config,features)