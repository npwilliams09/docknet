import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from spektral.utils.convolution import normalized_adjacency

def lr_scheduler(epoch, lr):
    LR = 0.001
    if epoch > 35:
        return 0.00001
    if epoch > 25:
        return 0.00005
    if epoch > 15:
        return 0.0001
    if epoch > 5:
        return 0.0005
    return LR


def graph_mode(key):  # to be finished
    modes = {
        'gcn': 'laplacian',
        'sage': 'binary'
    }
    return modes[key]


def get_crossentropy_loss(weight):
    def weighted_crossEntropy(y_true, y_pred):
        true = tf.cast(K.flatten(y_true), tf.float32)
        pred = tf.cast(K.flatten(y_pred), tf.float32)

        weights = (true * (weight - 1)) + 1

        bce = K.binary_crossentropy(true, pred)

        return K.mean(weights * bce)
    return weighted_crossEntropy


def get_lr_schedule(steps):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.001,
        decay_steps=steps,
        decay_rate=0.95,
        staircase=True)
    return lr_schedule


def get_checkpoint_callback(out_path):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=out_path,
        save_weights_only=True,
        monitor='val_auc',
        mode='max',
        save_best_only=True)
    return model_checkpoint_callback


def zeroDiagonal(mat):
    mask = 1 - np.eye(mat.shape[0])
    return mask * mat


def graph_preprocess(graph, mode):
    if mode == 'laplacian':
        return normalized_adjacency(graph)
    elif mode == 'binary':
        return zeroDiagonal(graph)
