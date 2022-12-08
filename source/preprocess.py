import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from torchvision import transforms as T


import numpy as np
import pandas as pd
from numba import jit



class SS_MNIST_dataset(Dataset):

    
    def __init__(
            self,
            x_onehot,
            y_class,
            L_disc_accept
        ):

        
        if not torch.is_tensor(x_onehot):
            self.x_onehot = torch.tensor(x_onehot).float()
        else:
            self.x_onehot = x_onehot.float()

        if not torch.is_tensor(y_class):
            self.y_class = torch.tensor(y_class).float()
        else:
            self.y_class = torch.tensor(y_class).float()

        if not torch.is_tensor(L_disc_accept):
            self.L_disc_accept = torch.tensor(L_disc_accept).float()
        else:
            self.L_disc_accept = torch.tensor(L_disc_accept).float()


        
    def __len__(self):
        return len(self.x_onehot)

    def transform_data(self, x):
        nn_transforms = torch.nn.Sequential(
                T.Resize([32, 32]),
                T.Normalize((0.5), (0.5))
        )
        return nn_transforms(x)

    def __getitem__(self, idx):

        # convert the tensor to a simple list
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # image data
        X = self.x_onehot[idx]
        X = self.transform_data(X.unsqueeze(0))
        # class labels
        y_C = self.y_class[idx]
        # accept gradients or not
        L_accept = self.L_disc_accept[idx]

        return X, y_C, L_accept









