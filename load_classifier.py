import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from IPython import embed
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


'''
toy example script that loads a saved classifier and performs inference on a file
that contains in each line a sentence vector. An an output you get a list of
predicted classes and a tensor with the probability scores
'''


def LoadData(embd_path, dim=1024, bsize=32, shuffle=False, quiet=False):
    x = np.fromfile(embd_path, dtype=np.float32, count=-1)
    x.resize(x.shape[0] // dim, dim)

    if not quiet:
        print(
            ' - read {:d}x{:d} elements in {:s}'.format(x.shape[0], x.shape[1], embd_path))

    D = data_utils.TensorDataset(torch.from_numpy(x))
    loader = data_utils.DataLoader(D, batch_size=bsize, shuffle=shuffle)
    return loader


class Net(nn.Module):

    def __init__(self, idim=1024, odim=2, nhid=None,
                 dropout=0.0, gpu=0, activation='TANH'):
        super(Net, self).__init__()
        self.gpu = gpu
        modules = []

        modules = []
        print(' - mlp {:d}'.format(idim), end='')
        if len(nhid) > 0:
            if dropout > 0:
                modules.append(nn.Dropout(p=dropout))
            nprev = idim
            for nh in nhid:
                if nh > 0:
                    modules.append(nn.Linear(nprev, nh))
                    nprev = nh
                    if activation == 'TANH':
                        modules.append(nn.Tanh())
                        print('-{:d}t'.format(nh), end='')
                    elif activation == 'RELU':
                        modules.append(nn.ReLU())
                        print('-{:d}r'.format(nh), end='')
                    else:
                        raise Exception('Unrecognized activation {activation}')
                    if dropout > 0:
                        modules.append(nn.Dropout(p=dropout))
            modules.append(nn.Linear(nprev, odim))
            print('-{:d}, dropout={:.1f}'.format(odim, dropout))
        else:
            modules.append(nn.Linear(idim, odim))
            print(' - mlp %d-%d'.format(idim, odim))
        self.mlp = nn.Sequential(*modules)
        # Softmax is included CrossEntropyLoss !

        if self.gpu >= 0:
            self.mlp = self.mlp.cuda()

    def forward(self, x):
        return self.mlp(x)

    def TestCorpus(self, dset, name=' Dev', nlbl=4):
        correct = 0
        total = 0
        self.mlp.train(mode=False)
        corr = np.zeros(nlbl, dtype=np.int32)

        total_Y = []
        total_predictions = []
        for data in dset:
            X, Y = data
            Y = Y.long()
            if self.gpu >= 0:
                X = X.cuda()
                Y = Y.cuda()
            outputs = self.mlp(X)
            _, predicted = torch.max(outputs.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).int().sum()

            total_predictions.extend(predicted.tolist())
            total_Y.extend(Y.tolist())

            for i in range(nlbl):
                corr[i] += (predicted == i).int().sum()

        print(' | {:4s}: {:5.2f}%'
              .format(name, 100.0 * correct.float() / total), end='')
        print(' | classes:', end='')
        for i in range(nlbl):
            print(' {:5.2f}'.format(100.0 * corr[i] / total), end='')
        confusion_matrix(total_Y, total_predictions)
        if "Test" in name:
            print("\n")
            print(confusion_matrix(total_Y, total_predictions))
            print(classification_report(total_Y, total_predictions))

        return correct, total

    def InferCorpus(self, dset, name=' Test', nlbl=4):
        total = 0
        self.mlp.train(mode=False)
        corr = np.zeros(nlbl, dtype=np.int32)

        total_predictions = []
        total_outputs = torch.Tensor()

        for data in dset:
            X = data[0]
            X = X.cpu()
            # if self.gpu >= 0:
            #     X = X.cuda()
            outputs = self.mlp(X)
            total_outputs = torch.cat((total_outputs, outputs))
            _, predicted = torch.max(outputs.data, 1)
            total += X.size(0)

            total_predictions.extend(predicted.tolist())

            for i in range(nlbl):
                corr[i] += (predicted == i).int().sum()

        sm = nn.Softmax(dim=1)
        return total_predictions, sm(total_outputs)


# dim = 1024
# nb_classes = 5
# nhid = "500 300 100 50 20 10"
# nhid = [int(h) for h in nhid.split()]
# dropout = 0.2
# gpu = 0
# net = Net(idim=dim, odim=nb_classes,
#           nhid=nhid, dropout=dropout, gpu=gpu)
# embed()


def main():
    PATH = "/shared1/vasilis/datasets/sst/models/sst_fine_jan28"
    model = torch.load(PATH)

    model.eval()
    embeddings_path = "/shared1/vasilis/datasets/cmore/cmore_5.embeddings.sv"
    embeddings_path = "/shared1/vasilis/datasets/cmore/cmore_feb20.embeddings.sv"
    embeddings_path = "/shared1/vasilis/datasets/cmore/cmore_feb21.embeddings.sv"
    embeddings_path = "/shared1/vasilis/datasets/cmore/cmore_feb22.embeddings.sv"
    test_loader = LoadData(embeddings_path)
    predictions, softmax_scores = model.InferCorpus(test_loader)
    embed()


if __name__ == "__main__":
    main()
