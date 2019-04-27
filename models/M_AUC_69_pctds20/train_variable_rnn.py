import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from local_configuration import *
from sklearn.metrics import roc_curve, auc

from utils import *
from plots import plot_learning_curves, plot_confusion_matrix, plot_learning_curves_roc
from mydatasets import calculate_num_features, VisitSequenceWithLabelDataset, visit_collate_fn
from mymodels import MyVariableRNN
from sklearn.metrics import recall_score, roc_auc_score, precision_score, f1_score
import numpy as np

from local_configuration import *

# Set a correct path to the data files that you preprocessed
LABEL_PREFIX = "temporal_only"  #temporal_and_static
PATH_TRAIN_SEQS = os.path.join(PATH_OUTPUT, LABEL_PREFIX + ".hadm.seqs")
PATH_TRAIN_LABELS = os.path.join(PATH_OUTPUT, LABEL_PREFIX + ".hadm.labels")

NUM_EPOCHS = 200
BATCH_SIZE = 100
USE_CUDA = False  # Set 'True' if you want to use GPU
NUM_WORKERS = 0

# Data loading
print('===> Loading entire datasets')
seqs = pickle.load(open(PATH_TRAIN_SEQS, 'rb'))
labels = pickle.load(open(PATH_TRAIN_LABELS, 'rb')) 

num_features = calculate_num_features(seqs)
print(num_features)

dataset = VisitSequenceWithLabelDataset(seqs, labels, num_features)

#Split Dataset
train_size = int(0.2 * len(dataset))
validation_size = train_size #int(len(dataset) - train_size)
test_size = int(len(dataset) - (2*validation_size))#0 #len(dataset) - (train_size +validation_size )
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)
# batch_size for the test set should be 1 to avoid sorting each mini-batch which breaks the connection with patient IDs
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)


model = MyVariableRNN(num_features)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())# lr =0.00001
#optimizer = optim.Adam(model.parameters(), lr =0.0001,  eps=1e-08, amsgrad=False)
#optimizer = optim.ASGD(model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
#optimizer = optim.Adamax(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
model.to(device)
criterion.to(device)

best_val_auc = 0.0
train_losses, train_accuracies, train_rocs = [], [], []
valid_losses, valid_accuracies, valid_rocs = [], [], []
test_losses, test_accuracies, test_rocs = [], [], []

for epoch in range(NUM_EPOCHS):
	train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
	valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)


	_, _, train_results = evaluate(model, device, train_loader, criterion)

	train_results = np.array(train_results)
	valid_results = np.array(valid_results)

	train_actual = train_results[:, :1]
	train_pred = train_results[:, 1:]
	#print("TrainActl: ", train_actual)
	#print("TrainPred: ", train_pred)
	train_roc = get_roc_auc(train_actual,  train_pred)
	print("Train Roc_auc: " + str(train_roc))

	valid_actual = valid_results[:, :1]
	valid_pred = valid_results[:, 1:]
	#print("ValidActl: ", valid_actual)
	#print("ValidPred: ", valid_pred)
	valid_roc = get_roc_auc(valid_actual,  valid_pred)
	print("Valid Roc_auc: " + str(valid_roc))

	train_losses.append(train_loss)
	valid_losses.append(valid_loss)

	train_accuracies.append(train_accuracy)
	valid_accuracies.append(valid_accuracy)

	train_rocs.append(train_roc)
	valid_rocs.append(valid_roc)

	is_best = valid_roc > best_val_auc  # let's keep the model that has the best accuracy, but you can also use another metric.
	if is_best:
		best_val_auc = valid_roc
		torch.save(model, os.path.join(PATH_OUTPUT, "MyVariableRNN.pth"))

best_model = torch.load(os.path.join(PATH_OUTPUT, "MyVariableRNN.pth"))

plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)

plot_learning_curves_roc(train_rocs, valid_rocs, filename='learning_curves_roc.png')

train_loss, train_accuracy, train_results = evaluate(model, device, train_loader, criterion)
plot_confusion_matrix(train_results, ["Alive", "Dead"])

valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)
plot_confusion_matrix(valid_results, ["Alive", "Dead"])

valid_results = np.array(valid_results)
actual = valid_results[:, :1]
pred = valid_results[:, 1:]
fpr, tpr, _ = roc_curve(actual, pred)
roc_auc = auc(fpr, tpr)
print("Roc_auc: " + str(roc_auc))

y_true = [x[0] for x in valid_results]
y_pred = [x[1] for x in valid_results]

print('Precision Score: ' + str(precision_score(y_true, y_pred, pos_label=0)))
print('Recall Score: ' + str(recall_score(y_true, y_pred, pos_label=0)))
print('F1 Score: ' + str(f1_score(y_true, y_pred, pos_label=0)))
print('ROC AUC Score: ' + str(roc_auc_score(y_true, y_pred)))
