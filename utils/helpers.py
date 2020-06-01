import csv
import random
import numpy as np
import matplotlib.pyplot as plt

# Helper functions

'''
Process the input data (build word vocab, map each word in the language command
to its ID number, etc.)
Each output datapoint has the form [verb, object, command, embedding, image]
where 'verb' and 'object' are strings, 'command' is a list of word IDs,
'embedding' is a numpy array representing the image/object,
and 'image' is a string containing the image file name
'''
def read_data(train, test):
    word2id = {'UNK':0}
    id2word = {0:'UNK'}
    with open(train, 'r') as train_file:
        with open(test, 'r') as test_file:
            train_data = list(csv.reader(train_file))
            test_data = list(csv.reader(test_file))
            train_dt, test_dt = [], []
            for row in train_data:
                affordances = np.fromstring(row[3][1:-1], dtype=float, sep=',')
                sentence = []
                s = row[2].lower().split()
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


'''
Generate positive and negative examples from the data.
Each output datapoint has the form [verb, object, command, embedding, image, value],
with 'value' being 1.0 for a positive example and -1.0 for a negative example
'''
def gen_examples(data, vo_dict):
    pos = [(row[0], row[1], row[2], row[3], row[4], 1.0) for row in data]
    neg = []
    for row in pos:
        while True:
            neg_sample = random.choice(pos)
            #only sample examples with objects that cannot be paired
            #with the current verb to use as a negative example
            if (neg_sample[1] not in vo_dict[row[0]]):
                neg += [(row[0], neg_sample[1], row[2], neg_sample[3], neg_sample[4], -1.0)]
                break
    all_data = pos + neg
    return all_data, pos


# Generates lists for cross validation
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
