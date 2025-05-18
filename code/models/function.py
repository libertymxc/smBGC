import torch


def drop_feature_weighted_2(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w
    x = x.clone()
    drop_mask = torch.bernoulli(drop_prob.expand_as(x)).to(torch.bool)
    x[drop_mask] = 0

    return x


def feature_drop_weights_dense(x, feature_matrix):
    x = x.abs()
    w = x * feature_matrix
    w=w.abs()
    w = torch.mean(w, dim=0, keepdim=True)
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


