import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
 
class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		# You may use the input argument 'dim_input', which is basically the number of features
		self.fc1 = nn.Linear(in_features=dim_input, out_features=32)  
		self.tanh = nn.Tanh()
		self.rnn = nn.GRU(input_size=32, hidden_size=16, num_layers=2, batch_first=True)
		self.fc2 = nn.Linear(in_features=16, out_features=2)  

	def forward(self, input_tuple):

		seqs, lengths = input_tuple   
		seqs = self.fc1(seqs) 
		seqs = self.tanh (seqs)     
		seqs, _ =  self.rnn(seqs)    
		output = self.fc2(seqs[:, -1, :])
		return output
