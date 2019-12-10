import csv
import random
import numpy as np
import matplotlib.pyplot as plt

def read_data(train, test):
    word2id = {"something":0, "blicket":1, "the":2, "a":3}
    id2word = {0:"something", 1:"blicket", 2:"the", 3:"a"}
    with open(train, 'r') as train_file:
        with open(test, 'r') as test_file:
            train_data = list(csv.reader(train_file))
            test_data = list(csv.reader(test_file))
            train_dt, test_dt = [], []
            for row in train_data:
                affordances = np.fromstring(row[3][1:-1], dtype=float, sep=',')
                sentence = []
                s = row[2].split()
                for word in s:
                    if word not in word2id: # build word vocab
                        word2id[word] = len(word2id)
                        id2word[word2id[word]] = word
                    sentence.append(word2id[word]) # map each word to its ID number
                train_dt.append([row[0], row[1], sentence, affordances, row[4]])
            for row in test_data:
                affordances = np.fromstring(row[3][1:-1], dtype=float, sep=',')
                sentence = []
                s = row[2].split()
                for word in s:
                    if word not in word2id:
                        word2id[word] = len(word2id)
                        id2word[word2id[word]] = word
                    sentence.append(word2id[word])
                test_dt.append([row[0], row[1], sentence, affordances, row[4]])
    return (train_dt, test_dt, word2id, id2word)


def crossval_helper(l, total_list):
    lt = []
    for t in total_list:
        if t is not l:
            lt += t
    return lt


def gen_examples(train_data):
    train_pos = [(row[0], row[1], row[2], row[3], row[4], 1.0) for row in train_data]
    train_neg = []
    for row in train_pos:
        while True:
            neg_sample = random.choice(train_pos)
            if (not neg_sample[0] == row[0]) and (not neg_sample[1] == row[1]):
                train_neg += [(row[0], row[1], row[2], neg_sample[3], neg_sample[4], -1.0)]
                break
    data = train_pos + train_neg
    return data


def plot(losses, y_label, legend_label, file_name):
    plt.figure()
    for num, loss in losses:
        line_label = legend_label + ' ' + str(num)
        plt.plot(loss, label=line_label)
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(file_name)
