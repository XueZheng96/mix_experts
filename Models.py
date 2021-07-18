#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class CNNFashion(nn.Module):
    def __init__(self,num_classes):
        super(CNNFashion, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 4 * 4, 84)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(84, 42)
        self.fc3 = nn.Linear(42, num_classes)
        self.activation = nn.Softmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 12 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out1 = F.relu(self.fc2(x))
        x = self.fc3(out1)
        out2 = self.activation(x)
        return out2

    
class GateCNNFashion(nn.Module):
    def __init__(self): 
        super(GateCNNFashion, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 4 * 4, 84)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(84, 42)
        self.fc3 = nn.Linear(42,1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 12 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.activation(x)
        return x

class GateCNNFahsionSoftmax(nn.Module):
    def __init__(self):
        super(GateCNNSoftmax, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 4 * 4, 84)
        self.fc2 = nn.Linear(84, 42)
        self.fc3 = nn.Linear(42, 3)
        self.activation = nn.Softmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.activation(x)
        return x

