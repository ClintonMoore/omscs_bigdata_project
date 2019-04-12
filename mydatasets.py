import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset

def calculate_num_features(seqs):
	"""
	:param seqs:
	:return: the calculated number of features
	"""
	return len (seqs[0][0])


class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels, num_features):
		"""
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
			labels (list): list of labels (int)
			num_features (int): number of total features available
		"""

		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")

		self.labels = labels 
		seq =[]
		for admission in seqs: 
			arr = np.zeros((len(admission), num_features) )
			i = 0 
			for items in admission :      
				j = 0
				for item in items:
					arr[i][j] = item 
					j+=1 
				i+= 1
			seq.append (sparse.csr_matrix(arr)) 

		self.seqs =seq


	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		# returns will be wrapped as List of Tensor(s) by DataLoader
		return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
	where N is minibatch size, seq is a (Sparse)FloatTensor, and label is a LongTensor

	:returns
		seqs (FloatTensor) - 3D of batch_size X max_length X num_features
		lengths (LongTensor) - 1D of batch_size
		labels (LongTensor) - 1D of batch_size
	""" 

	max_length = batch[0][0].shape[0]
	num_features= batch[0][0].shape[1]
	l1 = []
	l2 = []
	l3 = []
	for i in range (len(batch)):   
		l1.append(batch[i][0].toarray())
		l2.append(batch[i][0].shape[0])
		l3.append(int (batch[i][1]) )

	seqs_tensor = torch.FloatTensor(l1)
	lengths_tensor = torch.LongTensor(l2)
	labels_tensor = torch.LongTensor(l3)

	return (seqs_tensor, lengths_tensor), labels_tensor
