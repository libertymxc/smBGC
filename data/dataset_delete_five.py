import os
import csv
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import scipy.sparse as sp
from SVD_five import count_single_group
from models.function import feature_drop_weights_dense,drop_feature_weighted_2

from torch_geometric.data import DataLoader

data_folder='/home/MBGL/data/ADNI_five'
ADNI='/home/MBGL/data/ADNI.csv'

AD_feature = count_single_group(0)
LMCI_feature = count_single_group(1)
MCI_feature = count_single_group(2)
EMCI_feature = count_single_group(3)
NC_feature = count_single_group(4)

def get_ids(num_subjects=None):
    """
    return:
        subject_IDs    : list of all subject IDs
    """
    subject_IDs = os.listdir(os.path.join(data_folder,"all_scn"))

    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    for i in range(len(subject_IDs)):
        subject_IDs[i]=subject_IDs[i][:10]


    return subject_IDs

def get_subject_label(subject_list, label):
    label_dict = {}

    with open(ADNI) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['Subject'] in subject_list:
                label_dict[row['Subject']] = row[label]

    return label_dict

def my_custom_transform(data):
    return data
def my_custom_pretransform(data):
    return data

class MyCustomDataset(Dataset):
    def __init__(self, data_list, transform, pre_transform):
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def x1_x2(x,y,AD_feature,LMCI_feature,MCI_feature,EMCI_feature,NC_feature,drop_rate):


    if y == 0:
        feature_martix = NC_feature
    elif y == 1:
        feature_martix = EMCI_feature
    elif y == 2:
        feature_martix = MCI_feature
    elif y == 3:
        feature_martix = LMCI_feature
    else:
        feature_martix = AD_feature
    feature_martix = feature_martix.T

    feature_martix = torch.from_numpy(feature_martix)

    feature_weights = feature_drop_weights_dense(x, feature_martix)

    x = drop_feature_weighted_2(x, feature_weights, drop_rate)

    return x


def transform_dataset(subject_list,kind,label,droprate1,droprate2,delete=True):
    dataset = []

    for subject in subject_list:
        if kind == "FCN":
            # path_adj = "/home/MBGL/data/ADNI_five/delete_feature/fcn_20/fcn"
            # adj_fl = os.path.join(path_adj,  subject + ".txt")
            path_adj = "/home/MBGL/data/ADNI_five/all_fcn"
            adj_fl = os.path.join(path_adj, "r" + subject + ".txt")
            droprate=droprate1
        else:
            # path_adj = "/home/MBGL/data/ADNI_five/delete_feature/fcn_20/scn"
            # adj_fl = os.path.join(path_adj, subject + ".txt")
            path_adj = "/home/MBGL/data/ADNI_five/all_scn"
            adj_fl = os.path.join(path_adj, subject + ".txt")
            droprate = droprate2

        adj_matrix = np.loadtxt(adj_fl, dtype=np.float64).T
        edge_index_temp = sp.coo_matrix(adj_matrix)
        values = edge_index_temp.data

        indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
        edge_index_A = torch.tensor(indices, dtype=torch.long)
        v = torch.tensor(values, dtype=torch.float32)
        x1=torch.tensor(adj_matrix,dtype=torch.float32)


        #标签0为CN，标签1为emci，标签2为mci,标签三为lmci，标签4为ad
        y1_label=label[subject]
        if y1_label == "CN":
            y1=0
        elif y1_label == "EMCI":
            y1=1
        elif y1_label == "MCI":
            y1 = 2
        elif y1_label == "LMCI":
            y1 = 3
        else :
            y1=4

        if delete:
            x1 = x1_x2(x1,y1,AD_feature, LMCI_feature, MCI_feature, EMCI_feature, NC_feature,
                       drop_rate=droprate)

        dataset.append(
            Data(x=x1, edge_index=edge_index_A, y=torch.tensor(y1, dtype=torch.long),
                 edge_attr=v))

    Dataset = MyCustomDataset(dataset, transform=my_custom_transform, pre_transform=my_custom_pretransform)

    return Dataset



if __name__ == '__main__':
    subject_IDs = get_ids()
    print(subject_IDs)
    labels = get_subject_label(subject_IDs, label="Group")
    print(labels)
    FCN_dataset = transform_dataset(subject_IDs, kind="FCN", label=labels, delete=True, droprate1=0.4, droprate2=0.4)
    SCN_dataset = transform_dataset(subject_IDs, kind="SCN", label=labels, delete=True, droprate1=0.4, droprate2=0.4)
    print(FCN_dataset[0].x, FCN_dataset[0].x.shape)
