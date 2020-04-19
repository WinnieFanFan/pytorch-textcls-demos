# -*- coding: utf-8 -*-
import os
import time
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
from models.textRNN import RNNClassifier
import torch.nn.functional as F
import torch
from utils import *
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from gensim.models import KeyedVectors


def test(input, target):
    with torch.no_grad():
        model.eval()  # evaluation
        output = model(input)
        loss_test = loss_fn(output, torch.argmax(target, dim=1))
        acc_test = accuracy(output, target)
    return loss_test.item(), acc_test


def train(dl_train):
    t = time.time()
    model.train()   # train
    for x, y in dl_train:
        # x，y 一个batch内的数据
        optimizer.zero_grad()
        output = model(x)  # forward
        loss_train = loss_fn(output, torch.argmax(y, dim=1))
        # acc_train = accuracy(output, y)
        loss_train.backward()
        optimizer.step()
    print('time: {:.4f}s'.format(time.time() - t))


class CONFIG_PARAS():
    def __init__(self):
        # 参数等常量存放
        self.data_dir = "./data"
        self.data_path = self.data_dir + "/20news_home/20ng_train_test_trainlb_testlb_lbnames.pkl"
        self.vec_model_path = self.data_dir + "/my_vec_model.txt"
        self.preprocessed_data_path = self.data_dir + "/20news_home/data_input_rnn.pkl"

        self.n_jobs = 6
        self.cuda = "-1"
        self.random_seed = 42
        self.hidden = 128
        self.epochs = 12
        self.lr = 0.001
        self.weight_decay = 1e-4
        self.dropout = 0.5
        self.n_class = 20
        self.max_sentence_length = 100
        self.max_vocab_size = 50000
        self.model_save_path = "./models_trained/20ng_news/rnn_model"


if __name__ == "__main__":
    args = CONFIG_PARAS()

    if os.path.exists(args.preprocessed_data_path):
        with open(args.preprocessed_data_path, "rb") as f:
            X_train, X_test, Y_train, Y_test, embedding_matrix = pickle.load(f)
    else:
        with open(args.data_path, "rb") as f:
            data_train, data_test, Y_train, Y_test, labels = pickle.load(f)

        # 构造embedding matrix
        texts = data_train + data_test
        tokenizer = Tokenizer(char_level=True, num_words=args.max_vocab_size, lower=False)
        tokenizer.fit_on_texts(texts=texts)
        data = texts_to_idx(texts, tokenizer, args.max_sentence_length)

        X_train = data[:len(data_train)]
        X_test = data[len(data_train):]

        vec_model = KeyedVectors.load_word2vec_format(args.vec_model_path, binary=False)
        embedding_dim = vec_model.vector_size
        embedding_matrix = get_embedding_matrix(vec_model, tokenizer, "glove")

        with open(args.preprocessed_data_path, "wb") as f:
            data_input = X_train, X_test, Y_train, Y_test, embedding_matrix
            pickle.dump(data_input, f, protocol=4)

    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)

    # mode: (RNN, LSTM, GRU)
    model = RNNClassifier(args.hidden, args.max_sentence_length, embedding_matrix, args.n_class,
                          args.dropout, bidirectional=True, mode="gru")

    X_train = torch.from_numpy(X_train).long()
    X_test = torch.from_numpy(X_test).long()
    Y_train = torch.tensor(Y_train).long()
    Y_test = torch.tensor(Y_test).long()
    dataset_train = TensorDataset(X_train, Y_train)
    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)

    if args.cuda != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.cuda(device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        for para in model.parameters():
            para = para.cuda(device)
    else:
        device = None

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = F.cross_entropy
    hist = (np.inf, 0)
    for epoch in range(args.epochs):
        train(dataloader_train)
        loss_train, acc_train = test(X_train, Y_train)
        loss_test, acc_test = test(X_test, Y_test)

        print('Epoch: {:04d}'.format(epoch + 1),
              '\nloss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()))
        print('loss_test: {:.4f}'.format(loss_test.item()),
              'acc_test: {:.4f}'.format(acc_test.item()))

        if loss_test < hist[0]:
            hist = (loss_test, acc_test)
            save_path = args.model_save_path
            save_model(model, optimizer, save_path, epoch, loss_test)

    print("Finished train.")
    print("best model loss:", hist[0], "accuracy:", hist[1])
