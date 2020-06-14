import glob

import numpy as np
import torch
from torch.utils.data import DataLoader


def load_data():
    d_array = np.load('data/TimeFreq9D.npz')
    d_label = np.load('data/TimeFreq9L.npz')
    array = []
    label = []
    for name in d_array.files:
        array.extend(d_array[name])
        label.extend(d_label[name])
    array = np.array(array)
    label = np.array(label)
    label = np.argmax(label, 1)

    # shuffule
    permutation = np.random.permutation(array.shape[0])
    array = array[permutation, :, :, : ]
    label = label[permutation]
    len_t = int(array.shape[0] * 0.8)
    tr_dataset = torch.utils.data.TensorDataset(torch.from_numpy(array[:len_t]), torch.from_numpy(label[:len_t]))
    val_dataset =  torch.utils.data.TensorDataset(torch.from_numpy(array[:len_t]), torch.from_numpy(label[:len_t]))

    te_dataset = torch.utils.data.TensorDataset(torch.from_numpy(array[len_t:]), torch.from_numpy(label[len_t:]))

    return tr_dataset, val_dataset, te_dataset



if __name__ == '__main__':
    # data shape (2402, 1, 16, 10, 32)
    a,b,c = load_data()
    pass

