import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


def sort_batch(data, label, length):
    batch_size = data.size(0)
    inx = torch.from_numpy(np.argsort(length.numpy())[::-1].copy())
    data = data[inx]
    label = label[inx]
    length = length[inx]
    # length->list without using torch.Tensor
    length = list(length.numpy())
    return data, label, length


class BiLSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, biFlag, dropout=0.5):
        # input_dim
        # hidden_dim
        # output_dim classify（Number of categories）
        # num_layers LSTM hidden layer
        # biFlag whether use bi
        super(BiLSTMNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        if biFlag:
            self.bi_num = 2
        else:
            self.bi_num = 1
        self.biFlag = biFlag
        # define device
        self.device = torch.device("cuda")

        # difine LSTM Network Layer
        self.layer1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                              num_layers=num_layers, batch_first=True,
                              dropout=dropout, bidirectional=biFlag)
        # define linear layer and logsoftmax输出
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim * self.bi_num, output_dim),
            nn.LogSoftmax(dim=2)
        )

        self.to(self.device)

    def init_hidden(self, batch_size):
        # init hidden state
        return (torch.zeros(self.num_layers * self.bi_num, batch_size, self.hidden_dim).to(self.device),
                torch.zeros(self.num_layers * self.bi_num, batch_size, self.hidden_dim).to(self.device))

    def forward(self, data, label, length):
        # input data, label and length
        batch_size = data.size(0)
        max_length = torch.max(length)
        # cut according to the max_length
        data = data[:, 0:max_length, :]
        label = label[:, 0:max_length]
        data, label, length = sort_batch(data, label, length)
        data, label = data.to(self.device), label.to(self.device)
        # pack sequence
        x = pack_padded_sequence(data, length, batch_first=True)

        # run the network
        hidden1 = self.init_hidden(batch_size)
        out, hidden1 = self.layer1(x, hidden1)
        # out,_=self.layerLSTM(x) is also ok if you don't want to refer to hidden state
        # unpack sequence
        out, length = pad_packed_sequence(out, batch_first=True)
        out = self.layer2(out)
        # return the correct label, predict label and length
        return label, out, length
