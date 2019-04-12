import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.

	plt.figure()
	plt.plot(np.arange(len(train_losses)), train_losses, label='Train')
	plt.plot(np.arange(len(valid_losses)), valid_losses, label='Validation')
	plt.ylabel('Loss')
	plt.xlabel('epoch')
	plt.legend(loc="best")
	plt.savefig('loss.png') 

	plt.clf()
	plt.plot(np.arange(len(train_accuracies)), train_accuracies, label='Train')
	plt.plot(np.arange(len(valid_accuracies)), valid_accuracies, label='Validation')
	plt.ylabel('Accuracy')
	plt.xlabel('epoch')
	plt.legend(loc="best")
	plt.savefig('accuracy.png') 
	 
	pass

# Following the example provided in https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
 
def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
		
	y_true, y_pred = zip(*results) 
	cm=  confusion_matrix(y_true, y_pred )

	#Normalization
	cm = cm.astype('float') /cm.sum(axis=1)[:, np.newaxis]
	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	ax.figure.colorbar(im, ax=ax) 
	ax.set(xticks=np.arange(cm.shape[1]),
	       yticks=np.arange(cm.shape[0]), 
	       xticklabels=class_names, yticklabels=class_names,
	       title="Confusion Matrix", ylabel='True label', xlabel='Predicted label') 

	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
		 rotation_mode="anchor") 
	fmt = '.2f' 
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
	
	fig.savefig("matrix.png")
	pass
