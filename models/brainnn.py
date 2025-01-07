import torch
from collections import defaultdict
import numpy as np
from itertools import permutations
from torch_geometric.utils import to_dense_adj
from torch.nn import functional as F


class BrainNN(torch.nn.Module):
    def __init__(self, args, gnn, discriminator=lambda x, y: x @ y.t()):
        super(BrainNN, self).__init__()
        self.gnn = gnn
        self.pooling = args.pooling
        self.discriminator = discriminator

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h,z,out = self.gnn(x, edge_index, edge_attr, batch)
        log_logits = F.log_softmax(out, dim=-1)

        return h,z,log_logits

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        return z1 @ z2

    def jsd_loss(self, z1, z2, batch):
        num_graphs = z2.shape[0]
        num_nodes = int(z1.shape[0] / num_graphs)



        z1_pergraph = torch.split(z1, num_nodes)
        l = []

        for nodeidx, graphidx in enumerate(batch):
            pos_mask = torch.zeros((num_nodes)).cuda()
            neg_mask = torch.ones((num_nodes)).cuda()
            similarity = self.sim(z1_pergraph[graphidx],z2[graphidx]).detach()
            if nodeidx >= 90:
                nodeidx = nodeidx - graphidx * 90
            pos_mask[nodeidx]= 1.
            neg_mask[nodeidx]= 0.
            E_pos = (np.log(2) - F.softplus(- similarity * pos_mask)).sum()
            neg_similarity = similarity * neg_mask
            E_neg = (F.softplus(- neg_similarity) + neg_similarity - np.log(2)).sum()
            E_neg = E_neg / (num_nodes - 1)
            l.append(E_neg - E_pos)

        loss_all = sum(l)

        return loss_all / (num_nodes * num_graphs)

