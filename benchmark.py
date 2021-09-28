import tensorflow as tf
import tensorflow_addons as tfa
import spektral
import pandas as pd
import numpy as np
from components.data_loader import *
from components.graphBlocks import *
from components.twod_conv import *
from components.combiner import *
from components.Docknet import *
from components.utils import *
from string import ascii_lowercase
import time

def get_config():
    return {
        "filters": 128,
        "graph_blocks": 2,
        "graph_style": 'gcn',
        "drop_rate": 0.2,
        "combine_style": 'matmul',
        "squeeze_excite": False,
        "wave_blocks": 4,
        "epochs": 14,
        "lr": 0.00006
    }


def swap_aug(x):
    a_input, a_graph, b_input, b_graph = x
    a_graph, b_graph = b_graph, a_graph
    a_input, b_input = b_input, a_input
    x = [a_input, a_graph, b_input, b_graph]
    return x


def unswap_aug(target):
    target = np.squeeze(target)
    return target.T


def a_flip(x):
    a_input, a_graph, b_input, b_graph = x
    a_input = np.flip(a_input, axis=0)
    a_graph = np.fliplr(np.flipud(a_graph))
    x = [a_input, a_graph, b_input, b_graph]
    return x


def unflip_a(target):
    target = np.squeeze(target)
    return np.flip(target, axis=0)


def b_flip(x):
    a_input, a_graph, b_input, b_graph = x
    b_input = np.flip(b_input, axis=0)
    b_graph = np.fliplr(np.flipud(b_graph))
    x = [a_input, a_graph, b_input, b_graph]
    return x


def unflip_b(target):
    target = np.squeeze(target)
    return np.flip(target, axis=1)


def predict(x, model):
    a_input, a_graph, b_input, b_graph = x
    a_graph = np.expand_dims(a_graph, axis=0)
    b_graph = np.expand_dims(b_graph, axis=0)
    x = [a_input, a_graph, b_input, b_graph]
    return model.predict(x, batch_size=1)


def master_eval(x, y, model, tta=False):
    if tta:
        predictions = []

        # a - no aug
        out = predict(x, model)
        out = np.squeeze(out)
        predictions.append(out)

        # b - swap
        if tta == 'b':
            xt = swap_aug(x)
            out = predict(xt, model)
            out = unswap_aug(out)
            predictions.append(out)

        # c - a flip
        if tta == 'c':
            xt = a_flip(x)
            out = predict(xt, model)
            out = unflip_a(out)
            predictions.append(out)

        # d - b flip
        if tta == 'd':
            xt = b_flip(x)
            out = predict(xt, model)
            out = unflip_b(out)
            predictions.append(out)

        #e - swap + a flip
        if tta == 'e':
            xt = swap_aug(x)
            xt = a_flip(xt)
            out = predict(xt,model)
            out = unswap_aug(out)
            out = unflip_a(out)
            predictions.append(out)

        #f - swap + b flip
        if tta == 'f':
            xt = swap_aug(x)
            xt = b_flip(xt)
            out = predict(xt,model)
            out = unswap_aug(out)
            out = unflip_b(out)
            predictions.append(out)

        #g - a flip + b flip
        if tta == 'g':
            xt = a_flip(x)
            xt = b_flip(xt)
            out = predict(xt,model)
            out = unflip_a(out)
            out = unflip_b(out)
            predictions.append(out)

        #h - swap + a flip + b flip
        if tta == 'h':
            xt = a_flip(x)
            xt = b_flip(xt)
            xt = swap_aug(xt)
            out = predict(xt,model)
            out = unflip_a(out)
            out = unflip_b(out)
            out = unswap_aug(out)
            predictions.append(out)

        # take mean of prediction
        sz = out.shape
        master = np.zeros(sz)
        for p in predictions:
            master += p
        master /= len(predictions)  # divide by number of augs

    else:  # no tta scenario
        master = predict(x, model)
        master = np.squeeze(master)

    return np.squeeze(y), master


def all_aug_eval(x, y, model):
    predictions = []

    # a - no aug
    out = predict(x, model)
    out = np.squeeze(out)
    predictions.append(out)

    # b - swap

    xt = swap_aug(x)
    out = predict(xt, model)
    out = unswap_aug(out)
    predictions.append(out)

    # c - a flip

    xt = a_flip(x)
    out = predict(xt, model)
    out = unflip_a(out)
    predictions.append(out)

    # d - b flip

    xt = b_flip(x)
    out = predict(xt, model)
    out = unflip_b(out)
    predictions.append(out)

    #e - swap + a flip

    xt = swap_aug(x)
    xt = a_flip(xt)
    out = predict(xt,model)
    out = unswap_aug(out)
    out = unflip_a(out)
    predictions.append(out)

    #f - swap + b flip

    xt = swap_aug(x)
    xt = b_flip(xt)
    out = predict(xt,model)
    out = unswap_aug(out)
    out = unflip_b(out)
    predictions.append(out)

    #g - a flip + b flip

    xt = a_flip(x)
    xt = b_flip(xt)
    out = predict(xt,model)
    out = unflip_a(out)
    out = unflip_b(out)
    predictions.append(out)

    #h - swap + a flip + b flip

    xt = a_flip(x)
    xt = b_flip(xt)
    xt = swap_aug(xt)
    out = predict(xt,model)
    out = unflip_a(out)
    out = unflip_b(out)
    out = unswap_aug(out)
    predictions.append(out)

    # take mean of prediction
    sz = out.shape
    master = np.zeros(sz)
    for p in predictions:
        master += p
    master /= len(predictions)  # divide by number of augs

    return np.squeeze(y), master


def validate(ls,data,model,tta=False):
    m = tf.keras.metrics.AUC()
    gen = seqGenerator(ls, data)
    i = 0
    while i < len(ls):
        x, y = next(gen)
        if tta == 'all':
            y_true, y_pred = all_aug_eval(x, y, model)
        else:
            y_true, y_pred = master_eval(x,y,model,tta)
        m.update_state(y_true,y_pred)
        i += 1
        print(f"\rCompleted {i}/{len(ls)}, AUC = {m.result()}", end="",flush=True)
    print()
    print(f"Final AUC = {m.result().numpy()}")
    return m.result().numpy()

def main():
    dir = Path("/scratch/gi80/")
    cwd_path = dir / "docknet"
    model_path =  cwd_path / "model-best.h5"
    config = get_config()
    features = 44
    model = docknet(config, features)
    model.load_weights(model_path)
    model.compile(metrics=[tf.keras.metrics.AUC(dtype='float32')], run_eagerly=True)

    data_path = dir / "dataset"
    train_path = data_path / "train"
    test_path = data_path / "test"

    g_mode = graph_mode(config["graph_style"])
    train_ls, train_data = process_dataset(train_path, g_mode, custom= cwd_path / "less_than_250k_train_0.94.txt")
    test_ls, test_data = process_dataset(test_path, g_mode, custom=cwd_path / "less_than_250k_test_0.97.txt")

    val_ls = test_ls[::2]
    test_ls = test_ls[1::2]

    results = {}

    for c in ascii_lowercase[1:8]:
        m = validate(test_ls, test_data, model, tta=c)
        results[c] = m

    #all augs
    m = validate(val_ls, test_data, model, tta='all')
    results["all"] = m

    #train
    m = validate(train_ls, train_data, model, tta=False)
    results["train"] = m

    #val
    m = validate(val_ls, test_data, model, tta=False)
    results["val"] = m

    start = time.time()
    #test
    m = validate(test_ls, test_data, model, tta=False)
    results["test"] = m
    end = time.time()

    delta = end - start
    delta /= len(test_ls)

    #write to file
    with open(cwd_path / "results.txt", 'w') as f:
        for key in results.keys():
            f.write(f"{key} resulted in AUC {results[key]}\n")

        f.write(f"\nAveraged {delta} seconds per prediction\n")

if __name__ == '__main__':
    main()