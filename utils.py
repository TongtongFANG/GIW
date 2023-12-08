import numpy as np
import torch.nn.functional as F
from sklearn.svm import OneClassSVM

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


# extract latent representation of data
def get_feature(net, train_loader, val_loader):
    net.cpu()
    net.eval()
    index_val_list = []
    fe_val_list = []
    fe_tr_list = []

    for i, (val_image, val_labels, val_) in enumerate(val_loader):
        out_val = net(val_image)
        for fe_val in F.relu(activation['fc1']):
            fe_val_list.append((fe_val / np.linalg.norm(fe_val)).detach().cpu().numpy())

        index_val_list.extend(val_.detach().cpu().numpy())

    for j, (image, labels, _) in enumerate(train_loader):
        out_train = net(image)
        for fe_tr in F.relu(activation['fc1']):
            fe_tr_list.append((fe_tr / np.linalg.norm(fe_tr)).detach().cpu().numpy())

    return fe_tr_list, fe_val_list, index_val_list


# split validation data by one-class svm model
def val_split(fe_tr, fe_val, index_val):
    clf = OneClassSVM(gamma=10000).fit(fe_tr)
    w = clf.score_samples(fe_val)

    split_labels = []
    for i in w:
        if i > 0.0001:
            split_labels.append(1)
        else:
            split_labels.append(0)

    alpha = np.count_nonzero(split_labels) / len(split_labels)
    val_dic = dict(zip(index_val, split_labels))

    return val_dic, alpha
