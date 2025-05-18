import numpy as np
import nni
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.manifold import TSNE
from typing import Optional
from torch.utils.data import DataLoader
import logging
from utils import mixup, mixup_criterion
import matplotlib.pyplot as plt


def train_and_evaluate(model, train_loader1,train_loader2, test_loader1,test_loader2, optimizer, device, args):
    torch.autograd.set_detect_anomaly(True)
    model.train().to(device)
    accs, macros, aucs, specificities, sensitivities = [], [], [], [], []
    epoch_num = args.epochs
    batch_size = args.train_batch_size
    lamda = args.lamda


    for i in range(epoch_num):
        loss_all = 0
        l1_all = 0
        l2_all = 0
        for data1,data2 in zip(train_loader1,train_loader2):
            data1 = data1.to(device)
            data2 = data2.to(device)


            if args.mixup:
                data, y_a, y_b, lam = mixup(data)
            optimizer.zero_grad()

            h1,z1,out1 = model(data1)
            h2,z2,out2 = model(data2)

            out=(out1+out2)/2

            l1 = F.nll_loss(out, data1.y)
            l2 = (model.jsd_loss(h1,z2,data1.batch) + model.jsd_loss(h2,z1,data1.batch))/2
            loss = l1 + l2 * lamda
            loss.backward()
            l1_all +=l1.item()
            l2_all +=l2.item()
            loss_all += loss.item()
            optimizer.step()


        epoch_loss = loss_all / len(train_loader1.dataset)

        train_micro, train_macro, train_auc, train_specificity, train_sensitivity = evaluate(model, device,
                                                                                             train_loader1,
                                                                                             train_loader2)
        logging.info(f'(Train) | Epoch={i:03d}, loss={epoch_loss:.4f}, '
                     f'train_micro={(train_micro * 100):.2f}, train_macro={(train_macro * 100):.2f}, '
                     f'train_auc={(train_auc * 100):.2f}, '
                     f'train_specificity={train_specificity}, train_sensitivity={train_sensitivity}')

        if (i + 1) % args.test_interval == 0:
            test_micro, test_macro, test_auc, test_specificity, test_sensitivity = evaluate(model, device, test_loader1,
                                                                                            test_loader2)
            accs.append(test_micro)
            macros.append(test_macro)
            aucs.append(test_auc)
            specificities.append(test_specificity)
            sensitivities.append(test_sensitivity)
            text = f'(Train Epoch {i}), test_micro={(test_micro * 100):.2f}, ' \
                   f'test_macro={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f}, ' \
                   f'test_specificity={test_specificity}, test_sensitivity={test_sensitivity}\n'
            logging.info(text)

        if args.enable_nni:
            nni.report_intermediate_result(train_micro)

    accs = np.array(accs)
    macros = np.array(macros)
    aucs = np.array(aucs)
    specificities = np.mean(np.array(specificities), axis=0)
    sensitivities = np.mean(np.array(sensitivities), axis=0)

    return accs.mean(), macros.mean(), aucs.mean(), specificities, sensitivities


@torch.no_grad()
def evaluate(model, device, loader1, loader2, test_loader: Optional[DataLoader] = None) -> (
float, float, float, float, float, float, list):
    model.eval()
    preds, trues, preds_prob = [], [], []

    for data1, data2 in zip(loader1, loader2):
        data1 = data1.to(device)
        data2 = data2.to(device)
        h1, z1, out1 = model(data1)
        h2, z2, out2 = model(data2)

        c = (out1 + out2) / 2

        pred = c.max(dim=1)[1]
        preds += pred.detach().cpu().tolist()
        preds_prob += torch.softmax(c, dim=1).detach().cpu().tolist()  # Use softmax for multiclass probabilities
        trues += data1.y.detach().cpu().tolist()

    train_micro = metrics.accuracy_score(trues, preds)
    train_macro = metrics.f1_score(trues, preds, average='macro')

    trues_one_hot = torch.nn.functional.one_hot(torch.tensor(trues), num_classes=5)
    auc = metrics.roc_auc_score(trues_one_hot, preds_prob, multi_class='ovr')

    confusion = metrics.confusion_matrix(trues, preds, labels=[0, 1, 2,4,5])

    sensitivities = []
    specificities = []

    for i in range(5):
        TP = confusion[i, i]
        FN = confusion[i, :].sum() - TP
        FP = confusion[:, i].sum() - TP
        TN = confusion.sum() - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        sensitivities.append(sensitivity)

        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificities.append(specificity)

    macro_sensitivity = np.mean(sensitivities)
    macro_specificity = np.mean(specificities)

    if test_loader is not None:
        test_micro, test_macro, test_auc, test_spe, test_sen = evaluate(model, device, loader1)
        return train_micro, train_macro, auc, macro_specificity, macro_sensitivity, test_micro, test_macro, test_auc, test_spe, test_sen
    else:
        return train_micro, train_macro, auc, macro_specificity, macro_sensitivity

def plot(model,device, loader1,loader2):
    model.eval()
    preds, trues, preds_prob = [], [], []
    embeddings = []
    labels = []
    correct = 0
    for data1, data2 in zip(loader1, loader2):
        data1 = data1.to(device)
        data2 = data2.to(device)
        h1, z1, out1 = model(data1)
        h2, z2, out2 = model(data2)

        c = (out1 + out2) / 2

        pred = c.max(dim=1)[1]
        preds += pred.detach().cpu().tolist()
        preds_prob += torch.exp(c)[:, 1].detach().cpu().tolist()
        trues += data1.y.detach().cpu().tolist()

        embeddings += c.detach().cpu().tolist()
        labels += data1.y.detach().cpu().tolist()


    embeddings = np.array(embeddings)
    labels = np.array(labels)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    plt.figure(figsize=(6, 4))
    class_names = ['NC', 'EMCI', 'MCI', 'LMCI', 'AD']
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_names)))

    for i, label in enumerate(unique_labels):
        indices = labels == label
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], color=colors[label], label=class_names[label],
                    alpha=0.6, s=30)

    plt.legend(loc="best")
    plt.xticks([])
    plt.yticks([])

    plt.savefig('tsne_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot3(model,device, loader1,loader2):
    model.eval()
    preds, trues, preds_prob = [], [], []
    embeddings = []
    labels = []
    correct = 0
    for data1, data2 in zip(loader1, loader2):
        data1 = data1.to(device)
        data2 = data2.to(device)
        h1, z1, out1 = model(data1)
        h2, z2, out2 = model(data2)

        c = (out1 + out2) / 2

        pred = c.max(dim=1)[1]
        preds += pred.detach().cpu().tolist()
        preds_prob += torch.exp(c)[:, 1].detach().cpu().tolist()
        trues += data1.y.detach().cpu().tolist()

        embeddings += c.detach().cpu().tolist()
        labels += data1.y.detach().cpu().tolist()

    # Convert embeddings and labels to tensors

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    plt.figure(figsize=(6, 4))
    class_names = ['NC', 'MCI', 'AD']
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_names)))

    for i, label in enumerate(unique_labels):
        indices = labels == label
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], color=colors[label], label=class_names[label],
                    alpha=0.6, s=30)

    plt.legend(loc="best")
    plt.xticks([])
    plt.yticks([])
    plt.show()