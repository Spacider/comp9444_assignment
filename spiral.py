# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        # a fully connected neural network with one hidden layer
        # using tanh activation
        # followed by a single output using sigmoid activation
        super(PolarNet, self).__init__()
        self.Linear1 = nn.Linear(194, num_hid)
        self.tanh = nn.Tanh()
        self.Linear2 = nn.Linear(num_hid, 1)

    def forward(self, input):
        # x = input[:, 0]
        # y = input[:, 1]
        # # count (r, a) first
        # r = torch.sqrt(torch.pow(x, 2) + torch.pow(x, 2))
        # a = torch.atan2(y, x)
        # # (r, a)
        # coordinates = 1
        #
        # print(coordinates)
        # output = self.Linear1(coordinates)
        # output = self.tanh(output)
        # output = self.Linear2(output)
        output = 1
        return output

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()gi
        # INSERT CODE HERE

    def forward(self, input):
        output = 0*input[:,0] # CHANGE CODE HERE
        return output

def graph_hidden(net, layer, node):
    plt.clf()
    # INSERT CODE HERE
