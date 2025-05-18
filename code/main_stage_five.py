from build_model import build_model
from modified_args import ModifiedArgs
from data.dataset_delete_five import get_ids, transform_dataset,get_subject_label
from data.get_y_five import get_y
# from SVD_five import count_single_group
from feature_select import feature_enhance
import argparse
import sys
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
import nni
import os
import random
from typing import List
import logging
from train_and_eval import train_and_evaluate,evaluate,plot


def seed_everything(seed):
    print(f"seed for seed_everything(): {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # set random seed for numpy
    torch.manual_seed(seed)  # set random seed for CPU
    torch.cuda.manual_seed_all(seed)  # set random seed for all GPUs


class ListDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(ListDataset, self).__init__()
        self.data, self.slices = self.collate(data_list)

    def __len__(self):
        return len(self.slices['x']) - 1

    def get(self, idx):
        return super().get(idx)

def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    if args.enable_nni:
        args = ModifiedArgs(args, nni.get_next_parameter())

    # init model
    model_name = str(args.model_name).lower()
    args.model_name = model_name
    # seed_everything(args.seed) # use args.seed for each run
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self_dir = os.path.dirname(os.path.realpath(__file__))

    subject_IDs = get_ids()
    labels = get_subject_label(subject_IDs, label="Group")
    FCN_dataset = transform_dataset(subject_IDs, kind="FCN", label=labels, delete=True,droprate1=args.drop_rate_fcn,droprate2=args.drop_rate_scn)
    SCN_dataset = transform_dataset(subject_IDs, kind="SCN", label=labels, delete=True,droprate1=args.drop_rate_fcn,droprate2=args.drop_rate_scn)

    # FCN_dataset = transform_dataset(subject_IDs, kind="FCN", label=labels)
    # SCN_dataset = transform_dataset(subject_IDs, kind="SCN", label=labels)

    y = get_y(FCN_dataset)
    num_features = FCN_dataset[0].x.shape[1]
    nodes_num = FCN_dataset[0].x.shape[0]


    # if args.model_name == 'gcn':
    #     bin_edges = calculate_bin_edges(dataset, num_bins=args.bucket_num)
    # else:
    #     bin_edges = None

    accs, macros, aucs, specificities, sensitivities = [], [], [], [], []
    for _ in range(args.repeat):
        seed_everything(random.randint(1, 1000000))  # use random seed for each run
        skf = StratifiedKFold(n_splits=args.k_fold_splits, shuffle=True)
        for train_index, test_index in skf.split(FCN_dataset, y):
            model = build_model(args, device, model_name, num_features, nodes_num)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            train_set1, test_set1 = FCN_dataset[train_index], FCN_dataset[test_index]
            train_set2, test_set2 = SCN_dataset[train_index], SCN_dataset[test_index]

            train_list1 = [FCN_dataset[i] for i in train_index]
            train_list2 = [SCN_dataset[i] for i in train_index]

            # 应用增强
            train_list1 = feature_enhance(train_list1, args.drop_rate_fcn)
            train_list2 = feature_enhance(train_list2, args.drop_rate_scn)

            # 封装为 Dataset
            train_set1 = ListDataset(train_list1)
            train_set2 = ListDataset(train_list2)


            train_loader1 = DataLoader(train_set1, batch_size=args.train_batch_size, shuffle=False)
            train_loader2 = DataLoader(train_set2, batch_size=args.train_batch_size, shuffle=False)
            test_loader1 = DataLoader(test_set1, batch_size=args.test_batch_size, shuffle=False)
            test_loader2 = DataLoader(test_set2, batch_size=args.test_batch_size, shuffle=False)

            # train

            test_micro, test_macro, test_auc, test_specificity, test_sensitivity = train_and_evaluate(
                model, train_loader1, train_loader2, test_loader1, test_loader2, optimizer, device, args
            )
            test_micro, test_macro, test_auc, test_specificity, test_sensitivity = evaluate(model, device, test_loader1,
                                                                                            test_loader2)
            logging.info(f'(Initial Performance Last Epoch) | test_micro={(test_micro * 100):.2f}, '
                         f'test_macro={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f}, '
                         f'test_specificity={test_specificity}, test_sensitivity={test_sensitivity}')


            accs.append(test_micro)
            macros.append(test_macro)
            aucs.append(test_auc)
            specificities.append(test_specificity)
            sensitivities.append(test_sensitivity)

        accs, macros, aucs = np.array(accs), np.array(macros), np.array(aucs)
        specificities = np.array(specificities)
        sensitivities = np.array(sensitivities)

        result_str = f'(K Fold Final Result)| avg_acc={(np.mean(accs) * 100):.2f} +- {(np.std(accs) * 100):.2f}, ' \
                     f'avg_macro={(np.mean(macros) * 100):.2f} +- {(np.std(macros) * 100):.2f}, ' \
                     f'avg_auc={(np.mean(aucs) * 100):.2f} +- {(np.std(aucs) * 100):.2f}, ' \
                     f'avg_specificity={(np.mean(specificities) * 100):.2f} +- {(np.std(specificities) * 100):.2f}, ' \
                     f'avg_sensitivity={(np.mean(sensitivities) * 100):.2f} +- {(np.std(sensitivities) * 100):.2f}\n'
        logging.info(result_str)

    loader1 = DataLoader(FCN_dataset, batch_size=args.train_batch_size, shuffle=False)
    loader2 = DataLoader(SCN_dataset, batch_size=args.train_batch_size, shuffle=False)
    plot(model, device, loader1, loader2)

    with open('result.log', 'a') as f:
        # write all input arguments to f
        input_arguments: List[str] = sys.argv
        f.write(f'{input_arguments}\n')
        f.write(result_str + '\n')
    if args.enable_nni:
        nni.report_final_result(np.mean(accs))


def count_degree(data: np.ndarray):  # data: (sample, node, node)
    count = np.zeros((data.shape[1], data.shape[1]))
    for i in range(data.shape[1]):
        count[i, :] = np.sum(data[:, i, :] != 0, axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str,
                        choices=['PPMI', 'HIV', 'BP', 'ABCD', 'PNC', 'ABIDE'],
                        default="BP")
    parser.add_argument('--view', type=int, default=1)
    parser.add_argument('--node_features', type=str,
                        choices=['identity', 'degree', 'degree_bin', 'LDP', 'node2vec', 'adj', 'diff_matrix',
                                 'eigenvector', 'eigen_norm'],
                        default='adj')
    parser.add_argument('--pooling', type=str,
                        choices=['sum', 'concat', 'mean'],
                        default='sum')

    parser.add_argument('--model_name', type=str, default='gcn')
    # gcn_mp_type choices: weighted_sum, bin_concate, edge_weight_concate, edge_node_concate, node_concate
    parser.add_argument('--gcn_mp_type', type=str, default="edge_node_concate")
    # gat_mp_type choices: attention_weighted, attention_edge_weighted, sum_attention_edge, edge_node_concate, node_concate
    parser.add_argument('--gat_mp_type', type=str, default="attention_weighted")

    parser.add_argument('--enable_nni', action='store_true')
    parser.add_argument('--n_GNN_layers', type=int, default=1)
    parser.add_argument('--n_MLP_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--gat_hidden_dim', type=int, default=8)
    parser.add_argument('--edge_emb_dim', type=int, default=256)
    parser.add_argument('--bucket_sz', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--k_fold_splits', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--test_interval', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--lamda', type=float, default=1)
    parser.add_argument('--seed', type=int, default=112078)
    parser.add_argument('--diff', type=float, default=0.2)
    parser.add_argument('--mixup', type=int, default=0)  # [0, 1]
    parser.add_argument('--drop_rate_fcn', type=float, default=0.4)
    parser.add_argument('--drop_rate_scn', type=float, default=0.6)
    parser.add_argument('--delete_feature', type=int, default=20)
    main(parser.parse_args())
