import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, filename='learning_curves.png'):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.


	# Make a data frame
	df = pd.DataFrame({'x': range(1, len(train_losses)+1),
					   'Train Loss': np.asarray(train_losses),
					   'Test Loss':  np.asarray(valid_losses)})

	dfaccuracy = pd.DataFrame({'x': range(1, len(train_losses)+1),
					   'Train Accuracy': np.asarray(train_accuracies),
					   'Test Accuracy':  np.asarray(valid_accuracies)})

	# style
	plt.style.use('seaborn-darkgrid')

	# create a color palette
	palette = plt.get_cmap('Set1')

	# multiple line plot
	num = 0
	plt.subplot(1, 2, 1)
	for column in df.drop('x', axis=1):
		num += 1
		plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)

	# Add legend
	plt.legend(loc=2, ncol=2)

	# Add titles
	plt.title("Loss Curves", loc='left', fontsize=12, fontweight=0, color='orange')
	plt.xlabel("Epoch")
	plt.ylabel("Loss")




	# multiple line plot
	num = 0
	plt.subplot(1, 2, 2)
	for column in dfaccuracy.drop('x', axis=1):
		num += 1
		plt.plot(dfaccuracy['x'], dfaccuracy[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)

	# Add legend
	plt.legend(loc=2, ncol=2)

	# Add titles
	plt.title("Accuracy Curves", loc='left', fontsize=12, fontweight=0, color='orange')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	fig = plt.gcf()
	fig.set_size_inches(8.5, 3.5)
	plt.savefig(filename)
	plt.show()

	pass


def plot_confusion_matrix(results, class_names, filename='confusion_matrix.png'):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	title = 'Normalized confusion matrix'
	normalize = True

	y_true = [i[0] for i in results]
	y_pred = [i[1] for i in results]

	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	# Only use the labels that appear in the data
	classes = class_names #class_names[unique_labels(y_true, y_pred)]
	print(cm)
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	print(cm)

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	ax.figure.colorbar(im, ax=ax)
	ax.grid(False)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
		   yticks=np.arange(cm.shape[0]),
		   # ... and label them with the respective list entries
		   xticklabels=classes, yticklabels=classes,
		   title=title,
		   ylabel='True label',
		   xlabel='Predicted label')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			 rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
					ha="center", va="center",
					color="black")
	fig.tight_layout()
	fig = plt.gcf()
	fig.set_size_inches(6.5, 5)
	plt.savefig(filename)
	plt.show()
	pass



def plot_learning_curves_roc(train_rocs, test_rocs, filename='learning_curves_auc_roc.png'):
	# Make a data frame

	dfroc = pd.DataFrame({'x': range(1, len(train_rocs) + 1),
							   'Train AUC ROC Scores': np.asarray(train_rocs),
							   'Test AUC ROC Scores': np.asarray(test_rocs)})

	# style
	plt.style.use('seaborn-darkgrid')

	# create a color palette
	palette = plt.get_cmap('Set1')

	# multiple line plot
	num = 0
	plt.subplot(1, 1, 1)
	for column in dfroc.drop('x', axis=1):
		num += 1
		plt.plot(dfroc['x'], dfroc[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)

	# Add legend
	plt.legend(loc=2, ncol=2)

	# Add titles
	plt.title("AUC ROC Scores", loc='left', fontsize=12, fontweight=0, color='orange')
	plt.xlabel("Epoch")
	plt.ylabel("AUC ROC Score")

	fig = plt.gcf()
	fig.set_size_inches(5.5, 3.5)
	plt.savefig(filename)
	plt.show()