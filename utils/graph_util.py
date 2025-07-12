from collections import OrderedDict
from os.path import join
import math
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# for graph construction
import nmslib
import math


class Hnsw:
    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=True):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        # this is the actual nmslib part, hopefully the syntax should
        # be pretty readable, the documentation also has a more verbiage
        # introduction: https://nmslib.github.io/nmslib/quickstart.html
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        # the knnQuery returns indices and corresponding distance
        # we will throw the distance away for now
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices


def pt2graph(coords, features, threshold=256, radius=9):
    from torch_geometric.data import Data as geomData
    from itertools import chain
    coords, features = np.array(coords.cpu().detach()), np.array(features.cpu().detach())
    assert coords.shape[0] == features.shape[0]
    num_patches = coords.shape[0]

    model = Hnsw(space='l2')
    model.fit(coords)
    a = np.repeat(range(num_patches), radius - 1)
    b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]), dtype=int)
    edge_spatial = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)

    model = Hnsw(space='l2')
    model.fit(features)
    a = np.repeat(range(num_patches), radius - 1)
    b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]), dtype=int)
    edge_latent = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)
    start_point = edge_spatial[0, :]
    end_point = edge_spatial[1, :]
    start_coord = coords[start_point]
    end_coord = coords[end_point]
    tmp = start_coord - end_coord
    edge_distance = []
    for i in range(tmp.shape[0]):
        distance = math.hypot(tmp[i][0], tmp[i][1])
        edge_distance.append(distance)

    filter_edge_spatial = edge_spatial[:, np.array(edge_distance) <= threshold]

    G = geomData(x=torch.Tensor(features),
                 edge_index=filter_edge_spatial,
                 edge_latent=edge_latent,
                 centroid=torch.Tensor(coords))

    return G


def pairwise_distances(x):
    bn = x.shape[0]
    x = x.view(bn, -1)
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def calculate_gram_mat(x, sigma):
    dist = pairwise_distances(x)
    return torch.exp(-dist / sigma)


def reyi_entropy(x, sigma):
    alpha = 1.01
    k = calculate_gram_mat(x, sigma)
    k = k / torch.trace(k)
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x, y, s_x, s_y):
    alpha = 1.01
    x = calculate_gram_mat(x, s_x)
    y = calculate_gram_mat(y, s_y)
    k = torch.mul(x, y)
    k = k / torch.trace(k)
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def calculate_MI(x, y, s_x, s_y):
    Hx = reyi_entropy(x, sigma=s_x)
    Hy = reyi_entropy(y, sigma=s_y)
    Hxy = joint_entropy(x, y, s_x, s_y)
    Ixy = Hx + Hy - Hxy
    return Ixy


if __name__ == '__main__':
    s = 0
