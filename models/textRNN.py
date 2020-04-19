# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F

# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(RNN, self).__init__()
#
#         self.hidden_size = hidden_size
#
#         self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
#         self.i2o = nn.Linear(input_size + hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)
#
#     def forward(self, input, hidden):
#         combined = torch.cat((input, hidden), 1)
#         hidden = self.i2h(combined)
#         output = self.i2o(combined)
#         output = self.softmax(output)
#         return output, hidden
#
#     def initHidden(self):
#         return torch.zeros(1, self.hidden_size)


class RNNClassifier(nn.Module):
    def __init__(self, hidden_size, sequence_length, embedding_matrix, n_class, dropout, bidirectional, mode):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.n_class = n_class
        self.vocab_size = embedding_matrix.shape[0]
        self.embedding_dim = embedding_matrix.shape[1]
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn_output_shape = (2*self.hidden_size if self.bidirectional else self.hidden_size)

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding_layer.weight = nn.Parameter(embedding_matrix, requires_grad=False)
        if "rnn" in mode.lower():
            self.rnn_layer = nn.RNN(input_size=self.embedding_dim, hidden_size=hidden_size, nonlinearity="relu",
                                num_layers=2, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        elif "gru" in mode.lower():
            self.rnn_layer = nn.GRU(input_size=self.embedding_dim, hidden_size=hidden_size,
                                num_layers=2, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        elif "lstm" in mode.lower():
            self.rnn_layer = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size,
                                num_layers=2, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(self.rnn_output_shape, n_class)

    def forward(self, input_sentence):
        # input_sentence shape:
        # (batch, time_step, input_size), time_step--->seq_len
        input_embedded = self.embedding_layer(input_sentence)
        rnn_output, rnn_hidden = self.rnn_layer(input_embedded)
        if self.bidirectional:
            # concatenate normal RNN's last time step(-1) output and reverse RNN's last time step(0) output
            out_lastleyer = torch.cat([rnn_output[:, -1, :self.hidden_size], rnn_output[:, 0, self.hidden_size:]], dim=1)
        else:
            out_lastleyer = rnn_output[:, -1, :]
        output = self.output_layer(self.dropout(out_lastleyer))
        return F.log_softmax(output, dim=1)



