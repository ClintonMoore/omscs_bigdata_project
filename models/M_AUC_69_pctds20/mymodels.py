import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MyVariableRNNOLD(nn.Module):
    def __init__(self, dim_input):
        super(MyVariableRNN, self).__init__()
        # You may use the input argument 'dim_input', which is basically the number of features
        self.d0 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=dim_input, out_features=6)
        self.d1 = nn.Dropout(p=0.5)
        self.rnn = nn.GRU(input_size=6, hidden_size=10, num_layers=1, batch_first=True)
        self.d2 = nn.Dropout(p=0.5)
        self.fco1 = nn.Linear(in_features=10, out_features=10)
        self.d3 = nn.Dropout(p=0.5)
        self.fco2 = nn.Linear(in_features=10, out_features=2)

    def forward(self, input_tuple):
        seqs, lengths = input_tuple
        #seqs = self.d0(seqs)
        seqs = self.fc1(seqs)
        seqs = self.d1(seqs)
        # seqs = self.tanh (seqs)
        seqs, _ = self.rnn(seqs)
        #seqs = self.d2(seqs)
        seqs = self.fco1(seqs)
        #seqs = self.d3(seqs)
        seqs = F.relu(seqs)
        output = self.fco2(seqs[:, -1, :])
        return output


class MyVariableRNN(nn.Module):
    def __init__(self, dim_input):
        super(MyVariableRNN, self).__init__()
        self.hidden_size = 4
        self.fc_in_num_perceptrons = 75
        self.d1 = nn.Dropout(p=0.5)
        self.fc_in2_num_perceptrons = 10
        self.d2 = nn.Dropout(p=0.5)

        self.fc_in = nn.Linear(in_features=dim_input, out_features=self.fc_in_num_perceptrons)
        self.fc_in2 = nn.Linear(in_features=self.fc_in_num_perceptrons, out_features=self.fc_in2_num_perceptrons)
        self.tanh = nn.Tanh()
        self.gru1 = nn.GRU(input_size=self.fc_in2_num_perceptrons , hidden_size=self.hidden_size, batch_first=True, num_layers=1)
        self.fc_h= nn.Linear(in_features=self.hidden_size, out_features=2*self.hidden_size)
        self.fc_out = nn.Linear(in_features=2*self.hidden_size, out_features=2)



    def forward(self, input_tuple):

        batch_size = input_tuple[0].shape[0]

        fc_enc_embedding = F.relu(self.fc_in(input_tuple[0]))
        fc_enc_embedding = F.relu(self.fc_in2(self.d1(fc_enc_embedding)))

        packed_fc_embedd = pack_padded_sequence(self.d2(fc_enc_embedding), input_tuple[1], batch_first=True)

        packed_gru1_out, hidden1 = self.gru1( packed_fc_embedd)#, self.h1)

        y = torch.squeeze(F.relu(self.fc_h(hidden1)))
        y = torch.squeeze(self.fc_out(y))

        return y

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.hidden_size)
        return Variable(hidden)


