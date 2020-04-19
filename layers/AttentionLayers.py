import torch
from torch import nn
import numpy as np
from torch.nn import functional as F


class AttentionLayer(nn.Module):
    """implementation of scaled-dot-product attention layer"""
    def __init__(self, input_dim=1,  att_dropout=0.,):
        """
        :param att_dropout: dropout of attention layer
        :param input_dim: input dimension. Set input_dim = 1 when ignoring scale of input dimension
        """
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(att_dropout)
        self.softmax = nn.Softmax(1)

    def forward(self, query, key, value, att_mask=None):
        att_weights = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(self.input_dim)

        if att_mask is not None:
            assert att_mask.size() == att_weights.size(), \
                    'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{}.'.format(att_mask.size(), att_weights.size())
            # bug: 目前mask后再softmax 会变成 nan
            att_weights.data.masked_fill_(att_mask, -np.inf)

        att_weights = self.softmax(att_weights)
        att_weights = self.dropout(att_weights)
        output = torch.bmm(att_weights, value)
        return output, att_weights


class SeqAttentionLayer(nn.Module):
    """implementation of sequence attention in paper: Hierarchical Attention Networks for Document Classification"""
    def __init__(self, input_dim, attention_size, dropout=0.):
        super(SeqAttentionLayer, self).__init__()
        self.attention_size = attention_size
        # sequence attention
        self.seq_attention = nn.Linear(input_dim, attention_size)
        # context vector
        self.seq_context_vector = nn.Linear(attention_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_sequence):
        # assert len(input_sequence.size()) == 3
        seq_att = self.seq_attention(input_sequence)
        seq_att = F.tanh(seq_att)
        seq_att = self.seq_context_vector(seq_att).squeeze()
        seq_weights = F.softmax(seq_att)
        weighted_sequence = (input_sequence * seq_weights.unsqueeze(dim=2))
        weighted_sum = weighted_sequence.sum(dim=1)
        return weighted_sum, seq_weights
