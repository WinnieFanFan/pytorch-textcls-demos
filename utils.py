# -*- coding: utf-8 -*-
import time
from tqdm import tqdm
import numpy as np
import torch
import logging
from sklearn.metrics import pairwise_distances
import scipy.sparse as sp


def get_embedding_matrix(vec_model, tokenizer, mode="glove"):
    # values of word_index range from 1 to len
    embedding_matrix = np.random.random((len(tokenizer.word_index) + 1, vec_model.vector_size))
    for word, i in tokenizer.word_index.items():
        word = str(word)
        if word.isspace():
            embedding_vector = vec_model['blank']
        elif word not in vec_model.vocab:
            embedding_vector = vec_model['unk']
        else:
            embedding_vector = vec_model[word]
        embedding_matrix[i] = embedding_vector
    return embedding_matrix


def texts_to_idx(texts, tokenizer, max_sentence_length):
    data = np.zeros((len(texts), max_sentence_length), dtype='int32')
    for i, wordTokens in enumerate(texts):
        k = 0
        for _, word in enumerate(wordTokens):
            try:
                if k < max_sentence_length and tokenizer.word_index[word] < tokenizer.num_words:
                    data[i, k] = tokenizer.word_index[word]
                    k = k + 1
            except:
                if k < max_sentence_length:
                    data[i, k] = 0
                    k = k + 1
    return data


def texts_to_idx_han(texts, tokenizer, max_sentences, max_words):
    data = np.zeros((len(texts), max_sentences, max_words), dtype='int32')
    for i, sentences in enumerate(texts):
        for j, wordTokens in enumerate(sentences):
            if j < max_sentences:
                k = 0
                for _, word in enumerate(wordTokens):
                    try:
                        if k < max_words and tokenizer.word_index[word] < tokenizer.num_words:
                            data[i, j, k] = tokenizer.word_index[word]
                            k = k + 1
                    except:
                        if k < max_words:
                            data[i, j, k] = 0
                            k = k + 1
    return data


def accuracy(output, target):
    if output.size() == target.size():
        target = torch.argmax(target, dim=1)
    preds = torch.argmax(output, dim=1)
    accuracy = (preds == target).float().mean()
    return accuracy


def save_model(model, optimizer, path, epoch, loss):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'epoch': epoch
    }, path + "_ckpt.pt", pickle_protocol=4)
    logging.info("model saved",  path + "_ckpt.pt")


def load_model(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()   # 防止预测时修改模型
    return epoch, loss, path, model, optimizer


def compute_adj_matrix(input):
    """
    计算邻接矩阵，有不同的计算方式:
    方法1：1 - 词向量均值的similarity（满足：对角线为1，两个结点相似性越高，值越大）
    :param input:
    :return:
    """
    sim_matrix = pairwise_distances(input.tolist(), metric="cosine", n_jobs = 6)
    return 1 - sim_matrix


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # return sparse_to_tuple(adj_normalized)
    return adj_normalized.A


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # return sparse_to_tuple(features)
    return features.A


def accuracy(output, target):
    if output.size() == target.size():
        target = torch.argmax(target, dim=1)
    preds = torch.argmax(output, dim=1)
    acc = (preds == target).float().mean()
    return acc

#
# def test(input, target, model, loss_fn, device=None):
#     with torch.no_grad():
#         model.eval()  # evaluation
#         if device is not None:
#             input = input.cuda(device)
#             target = target.cuda(device)
#         output = model(input)
#         loss_test = loss_fn(output, torch.argmax(target, dim=1))
#         acc_test = accuracy(output, target)
#     return loss_test.item(), acc_test
#
#
# def train(dl_train, model, optimizer, loss_fn, device=None):
#     t = time.time()
#     model.train()  # train
#     process_bar = tqdm(dl_train)
#     for x, y in process_bar:
#         # x，y 一个batch内的数据
#         optimizer.zero_grad()
#         if device is not None:
#             x = x.cuda(device)
#             y = y.cuda(device)
#         output = model(x)  # forward
#         loss_train = loss_fn(output, torch.argmax(y, dim=1))
#         # acc_train = accuracy(output, y)
#         process_bar.set_description("batch loss: %4f " %loss_train.item())
#         loss_train.backward()
#         optimizer.step()
#     print('time: {:.4f}s'.format(time.time() - t))
#
