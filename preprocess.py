# -*- coding: utf-8 -*-
# @Time    : 2020/2/23 3:38 PM
# @Author  : Winnie
# @FileName: preprocess.py
# @Software: PyCharm

import re
import nltk
from nltk.corpus import stopwords
import pickle
import jieba
import multiprocessing as mp
from keras.utils import to_categorical
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

NJOBS = 6   # 分词并行进程数


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # 仅保留英文字符
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # words = [w for w in string.split() if w not in stop_words]
    # string = " ".join(words)
    return string.strip().lower()


def text_to_words_sequence(text, dataset="eng", stop_words=[]):
    """
    返回词序列，默认不使用停用词
    :param text: 文本
    :param dataset: 数据集使用的语言
    :param stop_words: 停用词表
    :return:
    """
    text = clean_str(text)
    if dataset == "eng":
        func = lambda x: x.split()
    else:
        func = lambda x: list(jieba.cut(x))
    words_seq = [word for word in func(text) if word not in stop_words]
    return words_seq


def run_preprocess(data, dataset="eng", stop_words=[]):
    """
    数据集合预处理：分词 -- 去停用词
    (-- embedding 暂时不要，不同的模型使用的embedding不同，写入dataloader)
    :param data: data_train, data_test, label_train, label_test
    :param dataset: 数据集英文/中文
    :param stop_words: 停用词表
    :return: X_train, Y_train, X_test, Y_test
    """
    data_train, data_test, label_train, label_test = data

    prcs = mp.Pool(processes=NJOBS)
    X_train = [prcs.apply(text_to_words_sequence, args=(x, dataset, stop_words)) \
               for x in data_train]
    Y_train = to_categorical(label_train)

    X_test = [prcs.apply(text_to_words_sequence, args=(x, dataset, stop_words)) \
              for x in data_test]
    Y_test = to_categorical(label_test)
    prcs.close()
    prcs.join()
    return X_train, Y_train, X_test, Y_test


def get_20ngnews():
    train = fetch_20newsgroups(  # data_home="../data/", # 文件下载的路径
        subset='train',  # 加载那一部分数据集 train/test
        categories=None,  # 选取哪一类数据集[类别列表]，默认20类
        shuffle=True,  # 将数据集随机排序
        random_state=42,  # 随机数生成器
        remove=(),  # ('headers','footers','quotes') 去除部分文本
        download_if_missing=True  # 如果没有下载过，重新下载
    )

    test = fetch_20newsgroups(  # data_home="../data/", # 文件下载的路径
        subset='test',  # 加载那一部分数据集 train/test
        categories=None,  # 选取哪一类数据集[类别列表]，默认20类
        shuffle=True,  # 将数据集随机排序
        random_state=42,  # 随机数生成器
        remove=(),  # ('headers','footers','quotes') 去除部分文本
        download_if_missing=True  # 如果没有下载过，重新下载
    )
    data_train = train["data"]
    label_train = train["target"]
    data_test = test["data"]
    label_test = test["target"]
    labels = train["target_names"]
    print("train size:", len(data_train))
    print("test size:", len(data_test))

    return data_train, data_test, label_train, label_test, labels


if __name__ == "__main__":
    data_dir = "./data/"

    # nltk.download('stopwords')
    data_train, data_test, label_train, label_test, labels = get_20ngnews()
    stop_words = set(stopwords.words('english'))
    X_train, Y_train, X_test, Y_test = run_preprocess(data = (data_train, data_test, label_train, label_test))

    # save
    with open(data_dir + "/20news_home/20ng_train_test_trainlb_testlb_lbnames.pkl", "wb") as f:
        pkg = (X_train, X_test, Y_train, Y_test, labels)
        pickle.dump(pkg, f)