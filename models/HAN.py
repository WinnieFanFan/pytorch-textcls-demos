import torch
from torch import nn
from torch.nn import functional as F
from layers.TimeDistributed import TimeDistributed
from layers.AttentionLayers import SeqAttentionLayer
import numpy as np

class sequentitial(nn.Module):
    def __init__(self, emb_layer, rnn_layer, att_layer):
        super(sequentitial, self).__init__()
        self.emb_layer = emb_layer
        self.rnn_layer = rnn_layer
        self.att_layer = att_layer

    def forward(self, input):
        rnn_out, rnn_hidden = self.rnn_layer(self.emb_layer(input))
        weighted_sum, att_weights = self.att_layer(rnn_out)
        return weighted_sum

class HanClassifier(nn.Module):
    """"""
    def __init__(self, hidden_size, embedding_matrix, num_sentences, num_words, nclass, bidirectional=True,
                 dropout=0.5):
        super(HanClassifier, self).__init__()
        self.vocab_size = embedding_matrix.shape[0]
        self.embedding_dim = embedding_matrix.shape[1]
        self.rnn_output_dim = 2*hidden_size if bidirectional else hidden_size

        # word embedding layer
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding_layer.weight = nn.Parameter(embedding_matrix, requires_grad=False)

        # word-level rnn and word-level attention
        self.word_rnn = nn.GRU(input_size=self.embedding_dim, hidden_size=hidden_size, num_layers=2, batch_first=True,
                               bidirectional=bidirectional, dropout=dropout)
        self.word_att = SeqAttentionLayer(input_dim=self.rnn_output_dim, attention_size=num_words)
        self.seq_encoder = sequentitial(self.embedding_layer, self.word_rnn, self.word_att)

        # sentence-level rnn and attention mechanism
        self.sentence_rnn = nn.GRU(input_size=self.rnn_output_dim, hidden_size=hidden_size, num_layers=2,
                                   batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.sentence_att = SeqAttentionLayer(input_dim=self.rnn_output_dim, attention_size=num_sentences)
        self.fn = nn.Linear(in_features=self.rnn_output_dim, out_features=nclass)

    def forward(self, doc_input):
        assert len(doc_input.size()) == 3, "input is not hierarchical sequence"
        # word-level encoding
        doc_encoded = TimeDistributed(self.seq_encoder, batch_first=True)(doc_input)
        # doc-level encoding
        sentence_seq, _ = self.sentence_rnn(doc_encoded)
        doc_vector, sent_att_weights = self.sentence_att(sentence_seq)
        output = self.fn(doc_vector)
        return F.log_softmax(output, dim=1)

if __name__ == "__main__":
    # test model
    # seq = np.random.randint(low=0, high=999, size=16*10*20)
    # seq = seq.reshape((16, 10, 20))
    # seq = torch.from_numpy(seq)
    # emb_matrix = torch.tensor(np.random.random((1000, 32))).float()
    #
    # model = HanClassifier(hidden_size=16,
    #                       embedding_matrix=emb_matrix,
    #                       num_sentences=10,
    #                       num_words=20,
    #                       nclass=2,
    #                       bidirectional=True,
    #                       dropout=0.5)
    # out = model(seq)
    pass