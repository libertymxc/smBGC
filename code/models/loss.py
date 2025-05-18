import torch
import numpy as np
import torch.nn.functional as F



def jsd_loss(z1, z2, batch):
    num_graphs = z2.shape[0]
    num_nodes = int(z1.shape[0]/num_graphs)

    pos_mask = torch.zeros((num_nodes, 1)).cuda()
    neg_mask = torch.ones((num_nodes, 1)).cuda()

    z1_pergraph=torch.split(z1, num_nodes)
    l=[]

    for nodeidx, graphidx in enumerate(batch):
        similarity = z1_pergraph[graphidx] @ z2[graphidx]
        if nodeidx >= 90:
            nodeidx = nodeidx - graphidx * 90
        pos_mask[nodeidx] = 1.
        neg_mask[nodeidx] = 0
        E_pos = (np.log(2) - F.softplus(- similarity * pos_mask)).sum()
        neg_similarity = similarity * neg_mask
        E_neg = (F.softplus(- neg_similarity) + neg_similarity - np.log(2)).sum()
        E_neg = E_neg/(num_nodes - 1)
        l.append(E_neg-E_pos)

    loss_all=sum(l)

    return loss_all/num_nodes
