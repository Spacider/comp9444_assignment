#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch
import string
import torch.nn as tnn
import torch.optim as toptim
import torch.nn.functional as F
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split()

    number_to_word = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six',
                      '7': 'seven', '8': 'eight', '9': 'nine'}
    new_sample = []
    for word in processed:
        if word not in stopWords:
            if word in number_to_word.keys():
                new_sample.append(number_to_word.get(word))
            else:
                new_sample.append(word.lower().strip(string.punctuation))

    return new_sample

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.S
    """

    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

stopWords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about',
             'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be',
             'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself',
             'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each',
             'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his',
             'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down',
             'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
             'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been',
             'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what',
             'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself',
             'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after',
             'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing',
             'it', 'how', 'further', 'was', 'here', 'than'}
wordVectors_dim = 50
wordVectors = GloVe(name='6B', dim=wordVectors_dim) # max 300 dim

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """

    # convert float to integer
    ratingOutput = torch.argmax(torch.round(F.log_softmax(ratingOutput, dim=1)), dim=1)
    categoryOutput = torch.argmax(torch.round(F.log_softmax(categoryOutput, dim=1)), dim=1)
    print(ratingOutput)
    print(categoryOutput)
    return ratingOutput, categoryOutput

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()
        self.LSTM = tnn.LSTM(wordVectors_dim, 50, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
        self.FC1_1 = tnn.Linear(200, 2)
        self.FC2_1 = tnn.Linear(200, 100)
        self.FC2_2 = tnn.Linear(100, 5)

    def forward(self, input, length):
        output, (h, c) = self.LSTM(input) # torch.Size([32, 200])
        output = torch.cat((output[:, -1, :], output[:, 0, :]), dim=1)
        rating_output = F.relu(self.FC1_1(output))
        category_output = F.relu(self.FC2_1(output))
        category_output = self.FC2_2(category_output)
        return rating_output.squeeze(), category_output.squeeze()


class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        rating_loss = F.cross_entropy(ratingOutput, ratingTarget)
        category_loss = F.cross_entropy(categoryOutput, categoryTarget)
        return (1 - loss_lambda) * rating_loss + loss_lambda * category_loss



net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.7
batchSize = 256 # 128 OR 256
epochs = 20 # 10 TO 20
optimiser = toptim.Adam(net.parameters(), lr=0.001)
loss_lambda = 0.65
