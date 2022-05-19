import torch
import os

import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np

class Stacked_MNIST(Dataset):
    def __init__(self, root="./Data", load=True, source_root=None, num_training_sample = 128000):#load=True means loading the dataset from existed files.
        super(Stacked_MNIST, self).__init__()
        if load == True:
            self.data = torch.load(os.path.join(root, "data.pt"))
            self.targets = torch.load(os.path.join(root, "targets.pt"))
        else:
            if source_root == None:
                source_root = "./Data"
                
            train_data = torchvision.datasets.MNIST(source_root, transform=transforms.Compose([
                               transforms.Pad(2),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]), download=True)

            test_data = torchvision.datasets.MNIST(source_root, transform=transforms.Compose([
                               transforms.Pad(2),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]), train=False, download=False)
            
            img = torch.zeros((0, 1, 32, 32))
            label = torch.zeros((0), dtype=torch.int64)
            
            for x, y in tqdm(DataLoader(train_data, batch_size=100, shuffle=False)):
                img = torch.cat((img, x), dim=0)
                label = torch.cat((label, y), dim=0)

            for x, y in tqdm(DataLoader(test_data, batch_size=100, shuffle=False)):
                img = torch.cat((img, x), dim=0)
                label = torch.cat((label, y), dim=0)
                
            ids = np.random.randint(0, img.shape[0], size=(num_training_sample, 3))

            self.data = torch.zeros((num_training_sample, 3, 32, 32))
            self.targets = torch.zeros((num_training_sample), dtype=torch.int64)
            
            for x in range(num_training_sample):
                self.data[x, 0, :, :] = img[ids[x, 0]]
                self.data[x, 1, :, :] = img[ids[x, 1]]
                self.data[x, 2, :, :] = img[ids[x, 2]]
                self.targets[x] = 100 * label[ids[x, 0]] + 10 * label[ids[x, 1]] + label[ids[x, 2]]
                
            torch.save(self.data, os.path.join(root, "data.pt"))
            torch.save(self.targets, os.path.join(root, "targets.pt"))
    
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        return img, targets

    def __len__(self):
        return len(self.targets)
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output