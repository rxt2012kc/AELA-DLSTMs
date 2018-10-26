# -*- coding: utf8 -*-
import numpy as np
import theano
import argparse
import time
import sys
import random
from Optimizer import OptimizerList
from Evaluator import Evaluators
from DataManager import DataManager
from model_aela import AELADLSTMS as Model


def train(model, train_data, optimizer, epoch_num, batch_size, batch_n):
    st_time = time.time()
    loss_sum = np.array([0.0, 0.0, 0.0])
    total_nodes = 0
    for batch in xrange(batch_n):  ## batch_n = 120
        start = batch * batch_size
        end = min((batch + 1) * batch_size, len(train_data))
        batch_loss, batch_total_nodes = do_train(model, train_data[start:end], optimizer)
        loss_sum += batch_loss
        total_nodes += batch_total_nodes
    end_time = time.time()
    print "train_time:", end_time - st_time
    return loss_sum[0], loss_sum[2]


def do_train(model, train_data, optimizer):
    eps0 = 1e-8
    batch_loss = np.array([0.0, 0.0, 0.0])
    total_nodes = 0
    for _, grad in model.grad.iteritems():
        grad.set_value(np.asarray(np.zeros_like(grad.get_value()), \
                                  dtype=theano.config.floatX))
    for item in train_data:
        sequences, target, tar_scalar, target_content_index, solution = item['seqs'], item['target'], \
                                                                        item['target_index'], \
                                                                        item['target_content_index'], item[
                                                                            'solution']
        find_word = dict(zip(wordlist.values(), wordlist.keys()))
        words = []
        loc = np.arange(len(sequences))
        batch_loss += np.array(model.func_train(sequences, target, solution, target_content_index, loc, len(sequences)))
        total_nodes += len(solution)
    for _, grad in model.grad.iteritems():
        grad.set_value(grad.get_value() / float(len(train_data)))
    optimizer.iterate(model.grad)
    return batch_loss, total_nodes


def test(model, test_data, grained):
    evaluator = Evaluators[grained]()
    keys = evaluator.keys()

    def cross(solution, pred):
        return -np.tensordot(solution, np.log(pred), axes=([0, 1], [0, 1]))

    loss = .0
    total_nodes = 0
    correct = {key: np.array([0]) for key in keys}
    wrong = {key: np.array([0]) for key in keys}
    for item in test_data:
        sequences, target, tar_scalar, target_content_index, solution = item['seqs'], item['target'], \
                                                                        item['target_index'], item[
                                                                            'target_content_index'], item['solution']
        loc = np.arange(len(sequences))
        pred = model.func_test(sequences, target, target_content_index, loc, len(sequences))
        loss += cross(solution, pred)
        total_nodes += len(solution)
        result = evaluator.accumulate(solution[-1:], pred[-1:])

    acc = evaluator.statistic()
    return loss / total_nodes, acc


if __name__ == '__main__':
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='lstm')
    parser.add_argument('--seed', type=int, default=int(1000 * time.time()))
    parser.add_argument('--dim_hidden', type=int, default=300)
    parser.add_argument('--dim_gram', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='data')
    parser.add_argument('--fast', type=int, choices=[0, 1], default=0)
    parser.add_argument('--screen', type=int, choices=[0, 1], default=0)
    parser.add_argument('--optimizer', type=str, default='ADAGRAD')
    parser.add_argument('--grained', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_word_vector', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=80)
    parser.add_argument('--batch', type=int, default=25)
    args, _ = parser.parse_known_args(argv)

    print "args:", args
    random.seed(args.seed)
    data = DataManager(args.dataset)
    wordlist = data.gen_word()

    print "len(wordlist):", len(wordlist)
    train_data, test_data = data.gen_data(args.grained)
    print "len(train_data):", len(train_data)
    print "len(test_data):", len(test_data)

    model = Model(wordlist, argv, len(data.dict_target))
    batch_n = (len(train_data) - 1) / args.batch + 1
    print "batch_n:", batch_n

    optimizer = OptimizerList[args.optimizer](model.params, args.lr, args.lr_word_vector)
    details = {'loss': [], 'loss_train': [], 'loss_dev': [], 'loss_test': [], \
               'acc_train': [], 'acc_dev': [], 'acc_test': [], 'loss_l2': []}
    print "details:", details

    for e in range(args.epoch):
        now = {}
        print "epoch####################################################:", e
        random.shuffle(train_data)
        now['loss'], now['loss_l2'] = train(model, train_data, optimizer, e, args.batch, batch_n)
        now['loss_train'], now['acc_train'] = test(model, train_data, args.grained)
        now['loss_test'], now['acc_test'] = test(model, test_data, args.grained)
        print now
        for key, value in now.items():
            details[key].append(value)
