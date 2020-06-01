import torch
import torch.nn as nn

# Language model to encode natural language commands

class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers,
                 drop_out, device):
        super(RNNModel, self).__init__()
        #defining some parameters
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        #defining the layers
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers,
                          batch_first=True, dropout = drop_out)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        #initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)
        #passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        #reshaping the outputs such that it can be fit into the fully connected layer
        hidden = hidden.contiguous().view(-1, self.hidden_dim)
        hidden = self.fc(hidden)
        #return last hidden layer instead of out, which is all the hidden layers
        return hidden

    def init_hidden(self, batch_size):
        #generate the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size,
                             self.hidden_dim).to(self.device)
        return hidden
