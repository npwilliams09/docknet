import numpy as np
import pandas as pd
from .utils import graph_preprocess
import random
import time
from multiprocessing import set_start_method, get_context




def createSet(txtFile, mode="contact", graph_mode='gcn'):
    prefix = "./docktact-graph/"
    prots = []
    with open(prefix + txtFile, 'r') as file:  # read in txt file
        for line in file:
            prots.append(line.strip())

    dic = {}

    for prot in prots:
        path = f"{prefix}{prot}/"
        chains = prot[-2:]
        dic[prot] = {}
        for chain in chains:
            dic[prot][f"{chain}_input"] = pd.read_feather(f"{path}{chain}_input.ft").values
            graph = np.load(f"{path}{chain}_adjMat.npy")
            dic[prot][f"{chain}_graph"] = graph_preprocess(graph, graph_mode)
        dic[prot]["target"] = np.load(f"{path}{mode[:4]}.npy")
    return prots, dic


def seqGenerator(ls, dic, aug=False):
    indexes = list(range(len(ls)))

    while True:
        random.shuffle(indexes)
        for i in indexes:
            prot = ls[i]
            chains = prot[-2:]
            a_input = dic[prot][f"{chains[0]}_input"]
            b_input = dic[prot][f"{chains[1]}_input"]

            a_graph = dic[prot][f"{chains[0]}_graph"]
            b_graph = dic[prot][f"{chains[1]}_graph"]

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

            assert (a_input.shape[0], b_input.shape[0]) == target.shape, target.shape

            a_input = np.expand_dims(a_input, axis=0)
            b_input = np.expand_dims(b_input, axis=0)

            #a_graph = np.expand_dims(a_graph, axis=0)
            #b_graph = np.expand_dims(b_graph, axis=0)

            targShape = target.shape
            target = target.reshape((1, targShape[0], targShape[1], 1))
            yield ([a_input, a_graph, b_input, b_graph], target)


def get_parameters(dataset):
    features = 0
    outputs = 0
    ones = 0
    for item in dataset.values():
        if features == 0: #only check once
            try:
                features = item["A_input"].shape[1]
            except:
                features = 0
                print("retry")
        target = item["target"]
        outputs += target.shape[0] * target.shape[1]
        ones += np.count_nonzero(target)

    zeros = outputs - ones
    weight = int(zeros / ones)

    print(f"Weight applied = {weight}\n Counted {features} features")

    return features, weight
