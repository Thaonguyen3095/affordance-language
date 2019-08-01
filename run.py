import os
import argparse
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mdn
import rnn

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='corpus.csv', help='input data')
opt = parser.parse_args()
os.makedirs('models', exist_ok=True)

num_layers = 1
rnn_input = 24
hidden_dim = 12
rnn_output = 8
mdn_input = 8
mdn_output = 5
num_gaussians = 20
num_epochs = 50
learning_rate = 0.0001
train_proportion = 0.9

device = torch.device("cpu")
Tensor = torch.LongTensor

def train(model, train_set, optimizer):
    model.train()
    sum_loss = 0.0
    for sentence, affordances in train_set:
        optimizer.zero_grad()
        sentence = Tensor(sentence).unsqueeze(0)
        affordances = torch.from_numpy(affordances).to(device).float().unsqueeze(0)
        pi, sigma, mu = model(sentence)
        loss = mdn.mdn_loss(pi, sigma, mu, affordances)
        loss.backward()
        sum_loss += loss.item()
        optimizer.step()
    print("Train loss: ", sum_loss/len(train_set))

def eval(model, test_set):
    model.eval()
    with torch.no_grad():
        sum_loss = 0.0
        for sentence, affordances in test_set:
            sentence = Tensor(sentence).unsqueeze(0)
            affordances = torch.from_numpy(affordances).to(device).float().unsqueeze(0)
            pi, sigma, mu = model(sentence)
            loss = mdn.mdn_loss(pi, sigma, mu, affordances)
            sum_loss += loss.item()
        print("Test loss: ", sum_loss/len(test_set))

def main():
    # read input data
    word2id = {}
    with open(opt.input, 'r') as dataFile:
        datarows = list(csv.reader(dataFile))
        data = []
        for row in datarows:
            affordances = np.fromstring(row[3], dtype=float, sep=' ')
            sentence = []
            s = row[2].split()
            for word in s:
                if word not in word2id: # build word vocab
                    word2id[word] = len(word2id)
                sentence.append(word2id[word]) # map each word to its ID number
            data.append([sentence, affordances])

    # split data for training and validation
    random.shuffle(data)
    split = int(len(data)*train_proportion)
    train_data = data[:split]
    test_data = data[split:]

    # initialize the model
    model = nn.Sequential(
    nn.Embedding(len(word2id), rnn_input),
    rnn.RNNModel(rnn_input,rnn_output,hidden_dim,num_layers),
    nn.Tanh(),
    mdn.MDN(mdn_input, mdn_output, num_gaussians)
    )
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    # train and evaluate model
    eval(model, test_data)
    for epoch in range(num_epochs):
        print()
        print("EPOCH ", epoch + 1)
        random.shuffle(train_data)
        train(model, train_data, optimizer)
        eval(model, test_data)
        torch.save(model.state_dict(), './models/nlmodel'+str(epoch+1)+'.pt')

if __name__ == "__main__":
    main()
