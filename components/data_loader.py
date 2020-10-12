import numpy as np
import pandas as pd
from .utils import graph_preprocess
import random
import time
from multiprocessing import set_start_method, get_context


def load_file(data):
    x, dir, graph_mode = data
    dic = {}
    dic["a_input"] = np.nan_to_num(pd.read_pickle(dir / "a_input.pkl").values.astype(np.float32),0.0)
    dic["b_input"] = np.nan_to_num(pd.read_pickle(dir / "b_input.pkl").values.astype(np.float32),0.0)
    dic["a_adj"] = np.nan_to_num(graph_preprocess(np.load(dir / "a_adj.npy"), graph_mode),0.0)
    dic["b_adj"] = np.nan_to_num(graph_preprocess(np.load(dir / "b_adj.npy"), graph_mode),0.0)
    dic["target"] = np.nan_to_num(np.load(dir / "target.npy"),0.0)
    return x, dic


def fetch_data(data):
    with get_context("spawn").Pool() as p:
        res = p.map_async(load_file, data)
        track_job(res, len(data))
        print()
        return res.get()


def process_dataset(path, graph_mode):
    files = [x for x in path.iterdir() if x.is_dir()][:10]
    data = [(x, path / x, graph_mode) for x in files]
    processed = fetch_data(data)
    print("Compiling...")
    dataset = {item[0]: item[1] for item in processed}
    return files, dataset


def track_job(job, total, update_interval=3):
    while job._number_left > 0:
        print("\rCompleted = {0} / {1}".format(total - \
                                               (job._number_left * job._chunksize), total), end='', flush=True)
        time.sleep(update_interval)


def seqGenerator(ls, dic, aug=False):
    indexes = list(range(len(ls)))

    while True:
        random.shuffle(indexes)
        for i in indexes:
            prot = ls[i]

            a_input = dic[prot]["a_input"]
            b_input = dic[prot]["b_input"]

            a_graph = dic[prot]["a_adj"]
            b_graph = dic[prot]["b_adj"]

            target = dic[prot]["target"]

            if aug:
                if (np.random.uniform() < 0.5):  # augment swap
                    a_graph, b_graph = b_graph, a_graph
                    a_input, b_input = b_input, a_input
                    target = target.T

                if (np.random.uniform() < 0.5):  # sequence A flip
                    a_input = np.flip(a_input, axis=0)
                    a_graph = np.fliplr(np.flipud(a_graph))
                    target = np.flip(target, axis=0)

                if (np.random.uniform() < 0.5):  # sequence B flip
                    b_input = np.flip(b_input, axis=0)
                    b_graph = np.fliplr(np.flipud(b_graph))
                    target = np.flip(target, axis=1)

            assert (a_input.shape[0], b_input.shape[0]) == target.shape, (target.shape, a_input.shape, b_input.shape)

            a_input = np.expand_dims(a_input, axis=0)
            b_input = np.expand_dims(b_input, axis=0)

            targ_shape = target.shape
            target = target.reshape((1, targ_shape[0], targ_shape[1], 1))

            yield [a_input, a_graph, b_input, b_graph], target


def get_parameters(dataset):
    features = 0
    outputs = 0
    ones = 0
    for item in dataset.values():
        if features == 0: #only check once
            features = item["a_input"].shape[1]
        target = item["target"]
        outputs += target.shape[0] * target.shape[1]
        ones += np.count_nonzero(target)

    zeros = outputs - ones
    weight = int(zeros / ones)

    print(f"Weight applied = {weight}\n Counted {features} features")

    return features, weight
