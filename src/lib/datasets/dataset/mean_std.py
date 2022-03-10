from __future__ import print_function, division
import os
from PIL import Image
import torch
import torch.utils.data
import torchvision
# from skimage import io
import torchvision.datasets as Datasat
import random
import numpy as np
# from sklearn.preprocessing import Binarizer


def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: \u81ea\u5b9a\u4e49\u7c7bDataset(\u6216ImageFolder\u5373\u53ef)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    dataset = Datasat.ImageFolder(root="/home/user/CenterNet2022/data/fiveclass/IMG", transform=torchvision.transforms.ToTensor())
    print(dataset)
    print(getStat(dataset))
