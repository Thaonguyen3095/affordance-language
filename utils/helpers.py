import csv
import numpy as np
import matplotlib.pyplot as plt

def read_data(fileName):
    word2id = {"something":0, "blicket":1}
    id2word = {0:"something", 1:"blicket"}
    with open(fileName, 'r') as dataFile:
        datarows = list(csv.reader(dataFile))
        dt = []
        for row in datarows:
            affordances = np.fromstring(row[3][1:-1], dtype=float, sep=',')
            sentence = []
            s = row[2].split()
            for word in s:
                if word not in word2id: # build word vocab
                    word2id[word] = len(word2id)
                    id2word[word2id[word]] = word
                sentence.append(word2id[word]) # map each word to its ID number
            dt.append([row[0], row[1], sentence, affordances])
    return (dt, word2id, id2word)


def crossval_helper(l, total_list):
    lt = []
    for t in total_list:
        if t is not l:
            lt += t
    return lt


def plot(losses, y_label, legend_label, file_name):
    plt.figure()
    for num, loss in losses:
        line_label = legend_label + ' ' + str(num)
        plt.plot(loss, label=line_label)
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(file_name)
