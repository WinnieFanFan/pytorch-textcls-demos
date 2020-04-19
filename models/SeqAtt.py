import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from torch.nn import functional as F


class SeqAttentionLayer(nn.Module):
    """implementation of sequence attention in paper: Hierarchical Attention Networks for Document Classification"""
    def __init__(self, input_dimension, attention_size, dropout=0.):
        super(SeqAttentionLayer, self).__init__()
        self.attention_size = attention_size
        # sequence attention
        self.seq_attention = nn.Linear(input_dimension, attention_size)
        # context vector
        self.seq_context_vector = nn.Linear(attention_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_sequence):
        assert len(input_sequence.size()) == 3
        seq_att = self.seq_attention(input_sequence)
        seq_att = F.tanh(seq_att)
        seq_att = self.seq_context_vector(seq_att).squeeze()
        seq_weights = F.softmax(seq_att)
        weighted_sequence = (input_sequence * seq_weights.unsqueeze(dim=2))
        weighted_sum = weighted_sequence.sum(dim=1)
        return weighted_sum, seq_weights


class SeqAttClassifier(nn.Module):
    """sequence gru model with attention mechanism"""
    def __init__(self, hidden_size, sequence_length, embedding_matrix, n_class, dropout, bidirectional, attention_size):
        super(SeqAttClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.n_class = n_class
        self.vocab_size = embedding_matrix.shape[0]
        self.embedding_dim = embedding_matrix.shape[1]
        self.bidirectional = bidirectional
        self.rnn_output_shape = (2*self.hidden_size if self.bidirectional else self.hidden_size)

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding_layer.weight = nn.Parameter(embedding_matrix, requires_grad=False)
        self.rnn_layer = nn.GRU(input_size=self.embedding_dim, hidden_size=hidden_size,
                                num_layers=1, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.att_layer = SeqAttentionLayer(input_dimension=self.rnn_output_shape, attention_size=self.sequence_length)
        self.output_layer = nn.Linear(self.rnn_output_shape, n_class)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_sentence):
        # input_sentence shape:
        # (batch, time_step, input_size), time_step--->seq_len
        input_embedded = self.embedding_layer(input_sentence)
        rnn_output, rnn_hidden = self.rnn_layer(input_embedded)
        att_rnn_output, att_weights = self.att_layer(rnn_output)
        output = self.output_layer(self.dropout(att_rnn_output))
        return F.log_softmax(output, dim=1)

# if __name__ == "__main__":
#     v = torch.randn(128, 256, 300)
#
#     att_layer = SeqAttentionLayer(input_dimension=300, attention_size=256)
#     out, att_w = att_layer(v)
