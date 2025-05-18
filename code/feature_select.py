import os
import csv
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import scipy.sparse as sp
from SVD_five import SVD_count
from models.function import feature_drop_weights_dense,drop_feature_weighted_2

from torch_geometric.data import DataLoader


def x1_x2(x,feature_martix,drop_rate):

    feature_martix = feature_martix.T

    feature_martix = torch.from_numpy(feature_martix)

    feature_weights = feature_drop_weights_dense(x, feature_martix)

    x = drop_feature_weighted_2(x, feature_weights, drop_rate)

    return x


def feature_enhance(Dataset,drop_rate):
    AD_matrix = []
    emci_matrix = []
    mci_matrix = []
    lmci_matrix= []
    nc_matrix= []


    for data in Dataset:
        label = data.y.item()
        if label==0:
            AD_matrix.append(data.x)
        elif label==1:
            emci_matrix.append(data.x)
        elif label==2:
            mci_matrix.append(data.x)
        elif label==3:
            lmci_matrix.append(data.x)
        else :
            nc_matrix.append(data.x)

    AD_feature = SVD_count(tuple(AD_matrix))
    LMCI_feature = SVD_count(tuple(emci_matrix))
    MCI_feature = SVD_count(tuple(mci_matrix))
    EMCI_feature = SVD_count(tuple(lmci_matrix))
    NC_feature = SVD_count(tuple(nc_matrix))

    for data in Dataset:
        if data.y==0:
            data.x=x1_x2(data.x,AD_feature,drop_rate)
        elif data.y==1:
            data.x=x1_x2(data.x,LMCI_feature,drop_rate)
        elif data.y==2:
            data.x=x1_x2(data.x,MCI_feature,drop_rate)
        elif data.y==3:
            data.x=x1_x2(data.x,EMCI_feature,drop_rate)
        else :
            data.x=x1_x2(data.x,NC_feature,drop_rate)

    return Dataset
