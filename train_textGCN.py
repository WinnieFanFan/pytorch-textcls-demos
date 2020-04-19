import os
import pickle
import time
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from models.GCN import GCN
from keras.utils import to_categorical

os.environ["OMP_NUM_THREADS"] = "12"


def load_features(data_path):
    """
    加载训练需要的变量，其中，idx_val 为训练数据
    :return: adj, features, labels, idx_train, idx_val, idx_test
    """
    with open(data_path, "rb") as f:
        features_tuple = pickle.load(f)
    adj, _, labels, idx_train, idx_val, idx_test = features_tuple
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = preprocess_features(sp.identity(adj.shape[0]))
    adj = preprocess_adj(adj)
    return adj, features, labels, idx_train, idx_val, idx_test


def accuracy(output, target):
    if output.size() == target.size():
        target = torch.argmax(target, dim=1)
    preds = torch.argmax(output, dim=1)
    accuracy = (preds == target).float().mean()
    return accuracy


def test():
    with torch.no_grad():
        model.eval()
        output = model(features, adj)
        loss_test = F.nll_loss(F.log_softmax(output[idx_test], dim=1), labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
    return loss_test, acc_test


def train(epoch, evaluation=True):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.binary_cross_entropy(F.sigmoid(output[idx_train]),
                                        torch.Tensor(to_categorical(np.array(labels[idx_train]))).float())
    # loss_train  = F.nll_loss(F.log_softmax(output[idx_train],dim=1), labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'time: {:.4f}s'.format(time.time() - t))

    if evaluation:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        loss_test, acc_test = test()
        print('loss_test: {:.4f}'.format(loss_test.item()),
              'acc_test: {:.4f}'.format(acc_test.item()))
    return loss_train, acc_train


class CONFIG_PARAS():
    def __init__(self):
        # 参数等常量存放
        self.data_dir = "./data"
        self.vec_model_path = self.data_dir + "/my_vec_model.txt"
        self.preprocessed_data_path = self.data_dir + "/20news_home/data_input_gcn.pkl"

        self.n_jobs = 6
        self.cuda = "-1"
        self.random_seed = 42
        self.hidden = 128
        self.epochs = 400
        self.lr = 0.02
        self.weight_decay = 0.
        self.dropout = 0.5
        self.n_class = 20
        self.model_save_path = "./models_trained/20ng_news/gcn_model"


if __name__ == "__main__":
    args = CONFIG_PARAS()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    torch.manual_seed(args.random_seed)
    if args.cuda != "-1":
        torch.cuda.manual_seed(args.random_seed)

    # load features
    if os.path.exists(args.preprocessed_data_path):
        adj, features, labels, idx_train, idx_val, idx_test = load_features(args.preprocessed_data_path)
    else:
        # 数据预处理过程
        pass

    labels = torch.LongTensor(labels)
    adj = torch.Tensor(adj)
    features = torch.Tensor(features)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    # 数据预处理
    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=args.n_class,
                dropout=args.dropout)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    # 使用gpu训练
    if args.cuda != "-1":
        model.cuda()
        features = features.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()
        adj = adj.cuda()

    # train(epoch=args.epoch)
    hist = (np.inf, 0)
    for epoch in range(args.epochs):
        loss_train, acc_train = train(epoch, evaluation=True)
        loss_test, acc_test = test()
        if loss_test < hist[0]:
            hist = (loss_test, acc_test)
            save_path = args.model_save_path
            save_model(model, optimizer, save_path, epoch, loss_test)
    print("Finished train.")
    print("best model loss:", hist[0], "accuracy:", hist[1])
