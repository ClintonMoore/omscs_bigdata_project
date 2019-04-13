import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from local_configuration import *

from utils import train, evaluate
from plots import plot_learning_curves, plot_confusion_matrix
from mydatasets import calculate_num_features, VisitSequenceWithLabelDataset, visit_collate_fn
from mymodels import MyVariableRNN
from sklearn.metrics import recall_score, roc_auc_score, precision_score, f1_score

# Set a correct path to the data files that you preprocessed
PATH_TRAIN_SEQS = "data/processed/hadm.seqs"
PATH_TRAIN_LABELS = "data/processed/hadm.labels" 

NUM_EPOCHS = 1
BATCH_SIZE = 32
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
train_size = int(0.7 * len(dataset))
validation_size = int(0.2 * len(dataset))
test_size = len(dataset) - (train_size +validation_size )
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)
# batch_size for the test set should be 1 to avoid sorting each mini-batch which breaks the connection with patient IDs
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)


model = MyVariableRNN(num_features)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr =0.001)

device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
model.to(device)
criterion.to(device)

best_val_acc = 0.0
train_losses, train_accuracies = [], []
valid_losses, valid_accuracies = [], []
for epoch in range(NUM_EPOCHS):
    train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
    valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    train_accuracies.append(train_accuracy)
    valid_accuracies.append(valid_accuracy)

    is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
    if is_best:
        best_val_acc = valid_accuracy
        torch.save(model, os.path.join(PATH_OUTPUT, "MyVariableRNN.pth"))

best_model = torch.load(os.path.join(PATH_OUTPUT, "MyVariableRNN.pth"))

#Plot learning curves
plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)

test_loss, test_accuracy, test_results = evaluate(best_model, device, test_loader, criterion)

#Check confusion matrix on test set
class_names = ['Alive', 'Dead']
plot_confusion_matrix(test_results, class_names)

y_true = [x[0] for x in valid_results]
y_pred = [x[1] for x in valid_results]

print('Precision Score: ' + str(precision_score(y_true, y_pred, pos_label=0)))
print('Recall Score: ' + str(recall_score(y_true, y_pred, pos_label=0)))
print('F1 Score: ' + str(f1_score(y_true, y_pred, pos_label=0)))
print('ROC AUC Score: ' + str(roc_auc_score(y_true, y_pred)))
