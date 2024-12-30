import pickle
import sys
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from utilts import *

import random
import logging
from sklearn import metrics
import gc
import os

from model import *
# from CoaDTI_pro_visualize import *
# from apex import amp
import esm


# model, alphabet = esm.pretrained.esm1_t6_43M_UR50S()  # esm1_t6_43M_UR50S esm_msa1b_t12_100M_UR50S esm1b_t33_650M_UR50S
# batch_converter = alphabet.get_batch_converter()


class Trainer(object):
    def __init__(self, model, batch_size):
        self.model = model
        self.optimizer_inner = optim.SGD(self.model.parameters(),
                                         lr=lr, weight_decay=weight_decay)
        self.optimizer = Lookahead(self.optimizer_inner, k=5, alpha=0.5)

        self.batch_size = batch_size


    def train(self, dataloader, epoch, es):
        # np.random.shuffle(dataloader)
        N = len(dataloader)
        train_labels = []
        train_preds = []
        loss_total = 0
        tk = tqdm(dataloader, desc="Training epoch: " + str(epoch))
        for i, data in enumerate(tk):
            proteins = data.protein

            data, proteins = data.to(device), proteins.to(device)
            loss, logits = self.model(data, proteins)
            preds = logits.max(1)[1]
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(parameters=self.model.parameters(), max_norm=5)
            self.optimizer.step()
            # self.scheduler.step()
            loss_total += loss.item()
            tk.set_postfix(
                {'loss': '%.6f' % float(loss_total / (i + 1)), 'LR': self.optimizer.param_groups[0]['lr'], 'ES':es})

            train_labels.extend(data.y.cpu())
            train_preds.extend(preds.cpu())
            # except:
            if np.isnan(loss_total):
                print(proteins.size())
                print(data)
                print(data.x)
                print(data.y)
                print(data.edge_index)
                print(proteins)
                exit()

            if i % 1000 == 0:
                del loss
                del preds
                gc.collect()

            torch.cuda.empty_cache()
        train_accu = metrics.accuracy_score(train_labels, train_preds)
        return loss_total, train_accu


class Tester(object):
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def test(self, dataset):
        N = len(dataset)
        # print(N)
        T, Y, S = [], [], []
        with torch.no_grad():
            for data in dataset:
                proteins = data.protein

                (correct_labels, predicted_labels,
                 predicted_scores) = self.model(data.to(device), proteins.to(device), train=False)
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)

        tpr, fpr, _ = precision_recall_curve(T, S)
        PRC = auc(fpr, tpr)
        train_accu = metrics.accuracy_score(T, Y)
        AUC = roc_auc_score(T, S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        return AUC, precision, recall,train_accu, PRC

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a+') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy')]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def random_shuffle(dataset, seed):
    random.seed(seed)  # 2021 2345 1234
    random.shuffle(dataset)
    return dataset

def train(fold, random_seed):

    # read data
    dir_input = './dataset/final/'

    setup_seed(random_seed)

    train_dataset = torch.load(dir_input + f'drug-target_train_esm_4000_{fold}.pt')
    test_dataset = torch.load(dir_input + f'drug-target_train_esm_4000_{fold}.pt')

    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    nps_emb = GNN(1)
    head = Classifier(compoundDim=167, proteinDim=512, hiddenDim=[1024, 1024, 512, 64],
                      outDim=2)
    model = FlexibleNNClassifier(nps_emb, head).to(device)

    trainer = Trainer(model, batch_size)
    tester = Tester(model, batch_size)

    AUCs = ('Epoch\tTime(sec)\tLoss_train\t'
            'AUC_test\tPrecision_test\tRecall_test\tAcc_test\tPRC_test')
    print('Training...')
    print(AUCs)
    logging.info(AUCs)  # 将表头写入日志文件
    start = timeit.default_timer()
    es = 0  # early stopping counter

    for epoch in range(0, iteration):


        loss_train, train_accu = trainer.train(train_loader, epoch, es)

        AUC_test, precision_test, recall_test, acc_test, PRC_test = tester.test(test_loader)

        end = timeit.default_timer()
        time = end - start
        #
        AUCs = [epoch, time, loss_train,
                AUC_test, precision_test, recall_test, acc_test, PRC_test]

        print('\t'.join(map(str, AUCs)))
        # 输出到日志文件
        logging.info('\t'.join(map(str, AUCs)))


if __name__ == "__main__":
    flod_n = 1
    logging.basicConfig(
        filename=f'training{flod_n}.log',  # 设置日志文件路径
        level=logging.INFO,  # 设置日志级别
        format='%(asctime)s - %(message)s',  # 设置日志格式
    )

    MAX_LENGTH = 4000

    lr = 5e-2  # 4e-3
    lr_decay = 0.5
    decay_interval = 20
    weight_decay = 1e-4
    iteration = 50
    optimizer = 'lookahead-SGD'
    pretrained_model = 'esm2_t36_3B_UR50D'
    (decay_interval, iteration) = map(int, [decay_interval, iteration])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    config = {'decay_interval':decay_interval,
     'iteration':iteration, 'lr':lr, 'lr_decay':lr_decay, 'weight_decay':weight_decay, 'optimizer':optimizer, 'pretrained_model':pretrained_model}

    """CPU or GPU."""
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device('cuda')
        #device = torch.device('cpu')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    train(flod_n, 1234)
