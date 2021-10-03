import numpy as np
import pandas as pd
from .utils import graph_preprocess
import random
import time
from multiprocessing import set_start_method, get_context
from pathlib import Path

def process_dataset(path,gmode,custom=None, width_limit=512):
    if custom:
        with open(custom) as f:
            paths = [Path(line.rstrip()) for line in f]
            paths = [str(p).rsplit('/', 1)[-1] for p in paths]
            paths = [path / p for p in paths]
    else:
        paths = [x for x in path.iterdir() if x.is_dir()]
    zipset = [(str(x).rsplit('/', 1)[-1], x, gmode) for x in paths]
    names = set([x[0] for x in zipset])
    dic = dict.fromkeys(names)

    start = time.time()
    endGoal = len(zipset)
    for i, pack in enumerate(zipset):
        result = load_file(pack)
        if result[1]["a_input"].shape[0] > width_limit: #skip
            del dic[result[0]]
            continue
        dic[result[0]] = result[1]
        print(f"\r{i}/{endGoal} files loaded",end='',flush=True)
    end = time.time()
    delta = end - start
    print(f"\nTook {'{:.2f}'.format(delta)} seconds to load {len(list(dic.keys()))} files")
    return list(dic.keys()), dic


def load_file(data):
    x, dir, graph_mode = data
    dic = {}
    dic["a_input"] = np.nan_to_num(pd.read_feather(dir / "a_input.ftr").values.astype(np.float32),0.0)
    dic["b_input"] = np.nan_to_num(pd.read_feather(dir / "b_input.ftr").values.astype(np.float32),0.0)
    dic["a_adj"] = np.nan_to_num(graph_preprocess(np.load(dir / "a_adj.npy"), graph_mode),0.0)
    dic["b_adj"] = np.nan_to_num(graph_preprocess(np.load(dir / "b_adj.npy"), graph_mode),0.0)
    dic["target"] = np.nan_to_num(np.load(dir / "target.npy"),0.0)
    return x, dic


def fetch_data(data):
    with get_context("spawn").Pool(8) as p:
        res = p.map_async(load_file, data)
        track_job(res, len(data))
        print()
        return res.get()


def track_job(job, total, update_interval=3):
    while job._number_left > 0:
        print("\rCompleted = {0} / {1}".format(total - \
                                               (job._number_left * job._chunksize), total), end='', flush=True)
        time.sleep(update_interval)

def pad_input(x, dim=512):
  npad = ((0, dim - x.shape[0]), (0,0))
  return np.pad(x, npad)

def pad_adj(a, dim=512):
  pad_len = dim - a.shape[0]
  npad = ((0, pad_len), (0, pad_len))
  return np.pad(a, npad)

def pad_targ(y, dim=512):
  pad_len = (dim - y.shape[0], dim - y.shape[1])
  npad = ((0, pad_len[0]), (0, pad_len[1]))
  t = np.pad(y, npad)
  return t.reshape(1, -1)

def seqGenerator(ls, dic, aug=False, pad_dim=512):
    indexes = list(range(len(ls)))

    while True:
        random.shuffle(indexes)
        for i in indexes:
            prot = ls[i]

            a_input = pad_input(dic[prot]["a_input"], pad_dim)
            b_input = pad_input(dic[prot]["b_input"], pad_dim)

            a_graph = pad_adj(dic[prot]["a_adj"], pad_dim)
            b_graph = pad_adj(dic[prot]["b_adj"], pad_dim)

            target = pad_targ(dic[prot]["target"], pad_dim)
            '''
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
            '''
            assert (a_input.shape[0] * b_input.shape[0]) == target.shape[1], (target.shape, a_input.shape, b_input.shape)

            a_input = np.expand_dims(a_input, axis=0)
            b_input = np.expand_dims(b_input, axis=0)

            #targ_shape = target.shape
            #target = target.reshape((1, -1))

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
