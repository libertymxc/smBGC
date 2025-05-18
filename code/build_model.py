import torch
from models.gcn import GCN
from models.brainnn import BrainNN
from models.mlp import MLP
from torch_geometric.data import Data
from typing import List


def build_model(args, device, model_name, num_features, num_nodes):
    if model_name == 'gcn':
        model = BrainNN(args,
                      GCN(num_features, args, num_nodes),
                      MLP(2 * num_nodes, args.hidden_dim, args.n_MLP_layers, torch.nn.ReLU, n_classes=args.num_classes),
                      ).to(device)
    else:
        raise ValueError(f"ERROR: Model variant \"{args.variant}\" not found!")
    return model
