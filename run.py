import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import rnn
import utils.helpers as utils

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='./data/corpus-short.csv',
                    help='input data')
parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers of model')
parser.add_argument('--rnn_input', type=int, default=128, help='')
parser.add_argument('--hidden_dim', type=int, default=64, help='')
parser.add_argument('--rnn_output', type=int, default=2048, help='')
parser.add_argument('--num_epochs', type=int, default=50, help='')
parser.add_argument('--learning_rate', type=float, default=00001, help='')
parser.add_argument('--dropout', type=float, default=0.0, help='')
parser.add_argument('--ret_num', type=int, default=5, help='')
parser.add_argument('--train_proportion', type=float, default=0.8, help='')
parser.add_argument('--k_fold', type=bool, default=False, help='')
parser.add_argument('--k', type=int, default=5, help='')

opt = parser.parse_args()
os.makedirs('models', exist_ok=True)
os.makedirs('images', exist_ok=True)
os.makedirs('images/cross_val', exist_ok=True)

"""
num_layers = 1
rnn_input = 128
hidden_dim = 64
rnn_output = 2048
num_epochs = 50
learning_rate = 0.0001
dropout = 0
ret_num = 5 #number of objects for retrieval task

#train/test split without cross validation
train_proportion = 0.8

#Cross validation
k_fold = False #True
k = 5
"""

#different data sizes
data_size = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

DEBUG = True #True
SAVE_MODEL = False #True
PLOT_FIG = False #True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def train(model, train_set, optimizer, plot_loss):
    model.train()
    sum_loss = 0.0
    for verb, obj, sentence, affordances in train_set:
        optimizer.zero_grad()
        affordances = torch.from_numpy(affordances).to(device).float().unsqueeze(0)
        sentence = Tensor(sentence).unsqueeze(0)
        output = model(sentence)
        loss = F.binary_cross_entropy_with_logits(output, affordances)
        loss.backward()
        sum_loss += abs(loss.item())
        optimizer.step()
    print('Train loss:', sum_loss/len(train_set))
    plot_loss.append(sum_loss/len(train_set))


def eval(model, test_set, plot_loss):
    model.eval()
    with torch.no_grad():
        sum_loss = 0.0
        for verb, obj, sentence, affordances in test_set:
            affordances = torch.from_numpy(affordances).to(device).float().unsqueeze(0)
            sentence = Tensor(sentence).unsqueeze(0)
            output = model(sentence)
            loss = F.binary_cross_entropy_with_logits(output, affordances)
            sum_loss += abs(loss.item())
        print('Test loss:', sum_loss/len(test_set))
        plot_loss.append(sum_loss/len(test_set))


def genRet(verb, test_set, objs):
    while True:
        sample = random.choice(test_set)
        #only sample objects used for different task
        if sample[0] != verb:
            for i, obj in enumerate(objs):
                #ensure that the set has unique objects
                if sample[1] is obj[0]:
                    break
                elif i == (len(objs) - 1):
                    objs.append([sample[1], sample[3]]) #object name and affordance vector
                    break
            if len(objs) >= opt.ret_num:
                return objs


def ret(model, test_set, id2word, ret_acc1, ret_acc2):
    model.eval()
    correct, correct2 = 0.0, 0.0
    with torch.no_grad():
        for verb, ob, sentence, affordances in test_set:
            s = ''
            for i in sentence:
                s += id2word[i] + ' '
            sentence = Tensor(sentence).unsqueeze(0)
            ret_objs = genRet(verb, test_set, [[ob, affordances]])
            pre_sort, losses = [], []
            output = model(sentence)
            for obj_name, obj in ret_objs:
                affordances = torch.from_numpy(obj).to(device).float().unsqueeze(0)
                loss = F.binary_cross_entropy_with_logits(output, affordances)
                pre_sort.append(loss.item())
                losses.append(loss.item())
            losses.sort()
            if losses[0] == pre_sort[0]:
                correct += 1
                correct2 += 1
                result = 'FIRST'
            elif losses[1] == pre_sort[0]:
                correct2 += 1
                result = 'SECOND'
            else:
                result = 'BOTH WRONG'
            if DEBUG:
                print()
                print(result)
            l = []
            for i, lt in enumerate(ret_objs):
                obj_name, aff = lt
                l.append([obj_name, aff, pre_sort[i]])
            top1, top2 = pre_sort.index(losses[0]), pre_sort.index(losses[1])
            t1, t2 = ret_objs[top1][0], ret_objs[top2][0]
            if DEBUG:
                print(s)
                print(output)
                print(l)
                print(t1,',', t2)
        print('RET_ACC Top1: {} Top2: {}'.format(correct/len(test_set), correct2/len(test_set)))
        ret_acc1.append(correct/len(test_set))
        ret_acc2.append(correct2/len(test_set))


def genTest(verb, test_set, objs):
    while True:
        sample = random.choice(test_set)
        #only sample objects used for different task
        if sample[0] != verb:
            for i, obj in enumerate(objs):
                if sample[1] is obj[0]:
                    break
                elif i == (len(objs) - 1):
                    objs.append([sample[1], sample[3]])
                    break
            if len(objs) >= opt.ret_num:
                return objs


def test(model, test_set, word2id, test_acc1, test_acc2):
    model.eval()
    correct, correct2 = 0.0, 0.0
    with torch.no_grad():
        for verb, ob, sentence, affordances in test_set:
            s = 'give me something to ' + verb
            sentence = []
            for word in s.split():
                sentence.append(word2id[word])
            sentence = Tensor(sentence).unsqueeze(0)
            ret_objs = genRet(verb, test_set, [[ob, affordances]])
            losses = []
            output = model(sentence)
            for obj_name, obj in ret_objs:
                affordances = torch.from_numpy(obj).to(device).float().unsqueeze(0)
                loss = F.binary_cross_entropy_with_logits(output, affordances)
                losses.append(loss.item())
            sort = sorted(losses)
            if sort[0] == losses[0]:
                correct += 1
                correct2 += 1
                result = 'FIRST'
            elif sort[1] == losses[0]:
                correct2 += 1
                result = 'SECOND'
            else:
                result = 'BOTH WRONG'
            if DEBUG:
                print()
                print(result)
            l = []
            for i, lt in enumerate(ret_objs):
                obj_name, aff = lt
                l.append([obj_name, aff, losses[i]])
            top1, top2 = losses.index(sort[0]), losses.index(sort[1])
            t1, t2 = ret_objs[top1][0], ret_objs[top2][0]
            if DEBUG:
                print(s)
                print(output)
                print(l)
                print(t1,',', t2)
        print('TEST_ACC Top1: {} Top2: {}'.format(correct/len(test_set), correct2/len(test_set)))
        test_acc1.append(correct/len(test_set))
        test_acc2.append(correct2/len(test_set))


def init_model(word2id):
    # initialize the model
    model = nn.Sequential(
    nn.Embedding(len(word2id), opt.rnn_input),
    rnn.RNNModel(opt.rnn_input, opt.rnn_output, opt.hidden_dim, opt.num_layers,
                 opt.dropout, device)).to(device)
    optimizer = optim.Adam(model.parameters(), lr = opt.learning_rate)
    return model, optimizer


def main():
    # read input data and split data for training and testing
    dt, word2id, id2word = utils.read_data(opt.input)
    if PLOT_FIG:
        plot_train, plot_eval, plot_ret1, plot_ret2, plot_test1, plot_test2 = [], [], [], [], [], []
    for dt_size in data_size:
        data = dt[:dt_size]
        print()
        print('DATA SIZE:', dt_size)
        random.shuffle(data)
        if k_fold: #cross validation
            train_data, test_data, split= [], [], []
            p = 1/k
            for i in range(k):
                s = int(len(data)*p*(i+1))
                split.append(s)
            for i, s in enumerate(split):
                if i == 0:
                    d = data[:s]
                else:
                    d = data[split[i-1]:s]
                test_data.append(d)
            for l in test_data:
                train_data.append(utils.crossval_helper(l, test_data))
        else: #normal train-test split
            split = int(len(data)*opt.train_proportion)
            train_data = data[:split]
            test_data = data[split:]

        # train and evaluate model
        if k_fold:
            plt_train, plt_eval, plt_ret1, plt_ret2, plt_test1, plt_test2 = [], [], [], [], [], []
            for i in range(k):
                print("------------------------------------------------------------------------------------")
                print("FOLD", i+1)
                model, optimizer = init_model(word2id)
                train_loss, eval_loss, ret_acc1, ret_acc2, test_acc1, test_acc2 = [], [], [], [], [], []
                eval(model, test_data[i], eval_loss)
                ret(model, test_data[i], id2word, ret_acc1, ret_acc2)
                test(model, test_data[i], word2id, test_acc1, test_acc2)
                for epoch in range(opt.num_epochs):
                    print()
                    print("EPOCH", epoch + 1)
                    random.shuffle(train_data[i])
                    train(model, train_data[i], optimizer, train_loss)
                    eval(model, test_data[i], eval_loss)
                    ret(model, test_data[i], id2word, ret_acc1, ret_acc2)
                    test(model, test_data[i], word2id, test_acc1, test_acc2)
                    if SAVE_MODEL:
                        torch.save(model.state_dict(),
                                   './models/nlmodel_datasize'+str(dt_size)+'_fold'+str(i+1)+'_epoch'+str(epoch+1)+'.pt')
                if PLOT_FIG:
                    plt_train.append((i+1, train_loss))
                    plt_eval.append((i+1, eval_loss))
                    plt_ret1.append((i+1, ret_acc1))
                    plt_ret2.append((i+1, ret_acc2))
                    plt_test1.append((i+1, test_acc1))
                    plt_test2.append((i+1, test_acc2))
            #plot losses & accuracies
            if PLOT_FIG:
                utils.plot(plt_train,'Train Loss', 'Fold number','./images/cross_val/train_loss_datasize'+str(dt_size)+'.png')
                utils.plot(plt_eval, 'Eval Loss', 'Fold number', './images/cross_val/eval_loss_datasize'+str(dt_size)+'.png')
                utils.plot(plt_ret1, 'Top1 Retrieval Accuracy', 'Fold number', './images/cross_val/ret_acc1_datasize'+str(dt_size)+'.png')
                utils.plot(plt_ret2, 'Top2 Retrieval Accuracy', 'Fold number', './images/cross_val/ret_acc2_datasize'+str(dt_size)+'.png')
                utils.plot(plt_test1, 'Top1 Test Accuracy', 'Fold number', './images/cross_val/test_acc1_datasize'+str(dt_size)+'.png')
                utils.plot(plt_test2, 'Top2 Test Accuracy', 'Fold number', './images/cross_val/test_acc2_datasize'+str(dt_size)+'.png')
        else:
            model, optimizer = init_model(word2id)
            train_loss, eval_loss, ret_acc1, ret_acc2, test_acc1, test_acc2 = [], [], [], [], [], []
            eval(model, test_data, eval_loss)
            ret(model, test_data, id2word, ret_acc1, ret_acc2)
            test(model, test_data, word2id, test_acc1, test_acc2)
            for epoch in range(opt.num_epochs):
                print()
                print('EPOCH', epoch + 1)
                random.shuffle(train_data)
                train(model, train_data, optimizer, train_loss)
                eval(model, test_data, eval_loss)
                ret(model, test_data, id2word, ret_acc1, ret_acc2)
                test(model, test_data, word2id, test_acc1, test_acc2)
                if SAVE_MODEL:
                    torch.save(model.state_dict(), './models/nlmodel_datasize'+str(dt_size)+'_epoch'+str(epoch+1)+'.pt')
            if PLOT_FIG:
                plot_train.append((dt_size, train_loss))
                plot_eval.append((dt_size, eval_loss))
                plot_ret1.append((dt_size, ret_acc1))
                plot_ret2.append((dt_size, ret_acc2))
                plot_test1.append((dt_size, test_acc1))
                plot_test2.append((dt_size, test_acc2))
    #plot losses & accuracies
    if not k_fold and PLOT_FIG:
        utils.plot(plot_train, 'Train Loss', 'Data size', './images/train_loss_no_postagging.png')
        utils.plot(plot_eval, 'Eval Loss', 'Data size', './images/eval_loss_no_postagging.png')
        utils.plot(plot_ret1, 'Top1 Retrieval Accuracy', 'Data size', './images/ret_acc1_no_postagging.png')
        utils.plot(plot_ret2, 'Top2 Retrieval Accuracy', 'Data size', './images/ret_acc2_no_postagging.png')
        utils.plot(plot_test1, 'Top1 Test Accuracy', 'Data size', './images/test_acc1_no_postagging.png')
        utils.plot(plot_test2, 'Top2 Test Accuracy', 'Data size', './images/test_acc2_no_postagging.png')

if __name__ == '__main__':
    main()
