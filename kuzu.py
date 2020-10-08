# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.fc1 = nn.Linear(28*28, 10)

    def forward(self, x):
        #                [batch_size, 1 , height, width]
        # x : torch.Size([64, 1, 28, 28])
        x = x.reshape(-1, 784)
        output = self.fc1(x)
        output = F.log_softmax(input=output, dim=1)
        return output


class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        self.liner1 = nn.Linear(28*28, 400)
        self.tanh1 = nn.Tanh()
        self.liner2 = nn.Linear(400, 200)
        self.tanh2 = nn.Tanh()
        self.liner3 = nn.Linear(200, 10)


    def forward(self, x):
        x = x.reshape(-1, 784)
        output = self.liner1(x)
        output = self.tanh1(output)
        output = self.liner2(output)
        output = self.tanh2(output)
        output = self.liner3(output)
        output = F.log_softmax(input=output, dim=1)
        return output


class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))
        self.Linear = nn.Linear(32*7*7, 10)


    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = output.view(x.size(0), -1)
        output = self.Linear(output)
        output = F.log_softmax(input=output, dim=1)
        return output
