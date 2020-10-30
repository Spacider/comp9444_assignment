# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


#For all three models, you should be trying values of num_hid between 5 and 10.
#For init_weight, you should try 0.001, 0.01, 0.1, 0.2, 0.3 and see which one works best.

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        # a fully connected neural network with one hidden layer
        # using tanh activation
        # followed by a single output using sigmoid activation
        super(PolarNet, self).__init__()
        self.layer1 = nn.Linear(2, num_hid)
        self.layer2 = nn.Linear(num_hid, 1)

    def forward(self, input):
        # print(input)  #([97, 2])
        r = torch.sqrt(input[:, 0] * input[:, 0] + input[:, 1] * input[:, 1]).view(-1, 1)
        a = torch.atan2(input[:, 1], input[:, 0]).view(-1, 1)
        input_polar = torch.cat((r, a), 1).view(-1, 2)

        self.hid1 = torch.tanh(self.layer1(input_polar))
        output = self.layer2(self.hid1)
        output = torch.sigmoid(output)
        return output


class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.layer1 = nn.Linear(2, num_hid , True)
        self.layer2 = nn.Linear(num_hid, num_hid, True)
        self.layer3 = nn.Linear(num_hid, 1, True)

    def forward(self, input):
        self.hid1 = torch.tanh(self.layer1(input))
        self.hid2 = torch.tanh(self.layer2(self.hid1))
        output = torch.sigmoid(self.layer3(self.hid2))
        return output


def graph_hidden(net, layer, node):
    #
    # print("------------------net----------------")
    # print(net)
    # print("------------------layer----------------")
    # print(layer)
    # print("------------------node----------------")
    # print(node)


    xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)

    with torch.no_grad(): # suppress updating of gradients
        net.eval()        # toggle batch norm, dropout

        net(grid)
        net.train() # toggle batch norm, dropout back again
        if layer == 1:
            output = net.hid1[:, node]
        elif layer == 2:
            output = net.hid2[:, node]
        pred = (output >= 0.5).float()

        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')
