#!/usr/bin/env python3
"""
comp9444 hw2
Group id: g023504
Groups members:
    JiaHui Wang(z5274762)
    MinXi Xie(z5245010)

Question: Briefly describe how your program works, and explain any design and training decisions you made along the way.

Answer:
1. how my program works:
        At first, we remove all possible symbols like comma or semicolon, then we remove some recurring
    stopwords and convert numbers from 1 to 10 to English words. In the end of  pre-processing, we
    remove punctuations by String library.
        Secondly, we choose wordVectors_dim as 300, after tried 100 and 200, we found 300 performed better.
    Then the data will be put into network, which is consisted by one LSTM layer and four Linear layers.
    we have compare to use GRU and LSTM and find that LSTM is slightly better than LSTM(around 5%), so
    eventually we choose LSTM. And then we use Linear layers to reduce output_size to 2 or 5, at first
    we use one layer from [32, 600] to [32, 2] and [32, 600] to [32, 5], however learned from hw1, using
    two Linear layers or more may make results better, so eventually we found 3 FC performed best. As for
    activation method, we use relu, which is more stable than Sigmoid and tanh.
        Thirdly, we choose cross_entropy as loss function for both rating and category because cross_entropy
    can support rating output and category output as [batch_size, the number of possible results], which is
    just the same as what we got from Linear layer([batch_size, output_size]).
        In the end, in the function of convertNetOutput, we try to apply logsoftmax activation method first
    and round it to 0 or 1. Eventually we apply argmax function to choose the max values in the tensor and
    consider it as final answer to compare with true answer.

2. explain any design and training decisions
        Besides the design or decisions explained in the first part of answer, I have tried more training options.
    After tried batchSize from 64 to 128 to 256, the result get slightly improved, so we choose 256. And as
    for epochs, we found as the programs going, the loss rate will decrease and stuck when epochs close to
    20, so that we choose 20 as our epochs. When it comes to optimiser, we choose Adam, because it shows that
    Adam is more suitable for NLP processing. In the end, we decided to give a lambda for rating loss and
    category loss, for the reason is that category_loss is larger than rating loss and we guess it is more
    important than rating loss.

"""

import torch
import string
import torch.nn as tnn
import torch.optim as toptim
import torch.nn.functional as F
from torchtext.vocab import GloVe
import re
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

    # remove all possible symbols
    remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    re.sub(remove_chars, '', sample)
    processed = sample.split()

    number_to_word = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six',
                      '7': 'seven', '8': 'eight', '9': 'nine'}
    new_sample = []
    for word in processed:
        if word not in stopWords:  # remove all stopwords
            if word in number_to_word.keys():
                new_sample.append(number_to_word.get(word))
            else:
                new_sample.append(word.lower().strip(string.punctuation))  # remove punctuation in the data


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
wordVectors_dim = 300
wordVectors = GloVe(name='6B', dim=wordVectors_dim)

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

    # convert float to integer and choose the max values as the result
    ratingOutput = torch.argmax(torch.round(F.log_softmax(ratingOutput, dim=1)), dim=1)
    categoryOutput = torch.argmax(torch.round(F.log_softmax(categoryOutput, dim=1)), dim=1)
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
        # network: LSTM -> Linear -> ReLU -> Linear -> Relu -> Linear
        # bidirectional = true help to create a reversed direction
        # num_layers = 2 will stacking two LSTMs together to form a stacked LSTM
        # dropout will help to avoid Overfitting
        self.LSTM = tnn.LSTM(wordVectors_dim, 300, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
        # define dropout to avoid overfitting
        # 3 Linear layers with ReLU
        self.FC1_1 = tnn.Linear(1200, 600)
        self.FC1_2 = tnn.Linear(600, 300)
        self.FC1_3 = tnn.Linear(300, 2)
        # 3 Linear layers with ReLU
        self.FC2_1 = tnn.Linear(1200, 500)
        self.FC2_2 = tnn.Linear(500, 200)
        self.FC2_3 = tnn.Linear(200, 5)

    def forward(self, input, length):
        output, (h, c) = self.LSTM(input)
        # convert the last ouput in the regular direction and reversed direction
        output = torch.cat((output[:, -1, :], output[:, 0, :]), dim=1)

        # Linear 1: for rating output
        rating_output = F.relu(self.FC1_1(output))
        rating_output = F.relu(self.FC1_2(rating_output))
        # F.dropout(rating_output, 0.2)
        rating_output = self.FC1_3(rating_output)


        # Linear 2: for category output
        category_output = F.relu(self.FC2_1(output))
        category_output = F.relu(self.FC2_2(category_output))
        # F.dropout(category_output, 0.2)
        category_output = self.FC2_3(category_output)

        return rating_output.squeeze(), category_output.squeeze()


class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        # apply cross_entropy (batch_size, the number of possible result)
        rating_loss = F.cross_entropy(ratingOutput, ratingTarget)
        category_loss = F.cross_entropy(categoryOutput, categoryTarget)
        # apply lambda for weight of rating loss and category loss
        return rating_loss + loss_lambda * category_loss



net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.85
batchSize = 256
epochs = 15
optimiser = toptim.Adam(net.parameters(), lr=0.001, weight_decay=0.0005)
loss_lambda = 0.65
