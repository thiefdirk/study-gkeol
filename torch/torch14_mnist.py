from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

path = './_data/torch_data/'
train_dataset = MNIST(path, train=True, download=True)
test_dataset = MNIST(path, train=False, download=True)

x_train, y_train = train_dataset.data/255., train_dataset.targets
x_test, y_test = test_dataset.data/255., test_dataset.targets

print(x_train.shape, y_train.shape)
print(x_test.size(), y_test.size())

print(np.min(x_train.numpy()), np.max(x_train.numpy()))

x_train, x_test = x_train.view(-1, 28*28), x_test.reshape(-1, 28*28) # .view()와 .reshape()의 차이점 : https://stackoverflow.com/questions/42479902/what-is-the-difference-between-view-and-reshape-in-pytorch
print(x_train.shape, x_test.shape)



