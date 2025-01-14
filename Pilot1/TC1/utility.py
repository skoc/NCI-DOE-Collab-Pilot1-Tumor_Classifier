from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, \
    precision_recall_fscore_support, matthews_corrcoef, confusion_matrix
from sklearn.preprocessing import label_binarize
import sys
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import argparse
import shutil
import glob

from scipy import interp
from itertools import cycle

def eprint(args):
    sys.stderr.write(str(args) + "\n")

# Print History in detail
# TRAINING
def print_history(history, validation):

    epochs = len(history.history['loss'])
    for epoch in range(epochs):
        if validation:
            eprint('Epoch {}/{}: accuracy : {:.3f}, loss : {:.3f}, val_accuracy : {:.3f}, val_loss : {:.3f}'.format(epoch + 1, epochs, history.history['acc'][epoch], history.history['loss'][epoch], history.history['val_acc'][epoch], history.history['val_loss'][epoch]))
        else:
            eprint(
                'Epoch {}/{}: accuracy : {:.3f}, loss : {:.3f} '.format(epoch + 1, epochs, history.history['acc'][epoch], history.history['loss'][epoch]))

# TRAINING OR TESTING
def plot_auc(y_labels, pred, parameters_dict = {}, title=''):
    # Scores
    false_positive_rate, recall, thresholds = roc_curve(y_labels, pred)
    roc_auc = auc(false_positive_rate, recall)

    # Plot
    plt.figure()
    plt.title(title + ' Receiver Operating Characteristic (ROC)')
    plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Recall')
    plt.xlabel('Fall-out (1-Specificity)')

    conf = ':'.join(map(str, parameters_dict.values())).replace('[', '').replace(']', '').replace(' ', '').replace(':', '_').replace(',', '-')
    plt.savefig(title + '_AUC_' + conf + '.png')
    plt.show()

def plot_auc_multi_class(y_labels, pred, classes, parameters_dict={}, title='', figsize=(12,12), lst_disease=None):

    # Compute ROC curve and ROC area for each class
    fpr, tpr, roc_auc = dict(), dict(), dict()
    y_labelized = label_binarize(y_labels, classes=classes)
    pred_labelized = label_binarize(pred, classes=classes)

    n_classes = len(classes)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_labelized[:, i], pred_labelized[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_labelized.ravel(), pred_labelized.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=figsize)
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
          label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
          color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
          label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
          color='navy', linestyle=':', linewidth=4)

    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))

    for i, color in zip(range(n_classes), colors):
        if lst_disease:
            label_str = 'ROC curve of class {0} (area = {1:0.2f})'.format(lst_disease[i], roc_auc[i])
        else:
            label_str = 'ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
              label=label_str)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    
    conf = ':'.join(map(str, parameters_dict.values())).replace('[', '').replace(']', '').replace(' ', '').replace(':', '_').replace(',', '-')
    plt.savefig("plot_auc_" + title + '_' + conf + '.png')
    plt.show()

# TRAINING OR TESTING
def calculate_metrics(y_labels, pred, average="macro"):

    precision, recall, fscore, _ = precision_recall_fscore_support(y_labels, pred, average=average)
    acc = accuracy_score(y_labels, pred)

    return np.round(100 * acc, 3), \
           np.round(100 * precision, 3), \
           np.round(100 * recall, 3), \
           np.round(100 * fscore, 3)

# TRAINING
def plot_loss(history, parameters_dict = {}, title='', figsize=(12, 8)):
    loss_train = list(history.history['loss'])
    loss_val = list(history.history['val_loss'])
    epochs_min= np.argmin(loss_val) + 1 # zero indexing fix

    epochs = range(1, len(loss_train) + 1)
    min_loss = min(loss_train + loss_val)

    plt.figure(figsize=figsize)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='Validation loss')
    plt.title('Training vs Validation loss')
    plt.axvline(x=epochs_min, c='red', ymax=0.99, ymin=0.01, linestyle='--')
    plt.xticks(epochs)
    plt.text(x=epochs_min - 0.8, y=min_loss - 0.15, s='epoch_best', c='red')
    plt.xlabel('Epochs')
    plt.ylabel('log(Loss)')
    plt.legend(loc='best')
    conf = ':'.join(map(str, parameters_dict.values())).replace('[', '').replace(']', '').replace(' ', '').replace(':','_').replace(',', '-')
    plt.savefig('plot_loss_' + title + '_' + conf + '.png')
    plt.show()

# TRAINING OR TESTING
def plot_predictions(validation_labels, pred, parameters_dict = {}, title='', figsize=(30, 10)):
    num_samples_to_show = np.min([len(pred), 100])
    plt.figure(figsize=figsize)
    plt.plot(range(num_samples_to_show), pred[:num_samples_to_show], 'ys', label='Predicted_value')
    plt.plot(range(num_samples_to_show), validation_labels[:num_samples_to_show], 'r*', label='Test_value')

    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.ylabel('predicted')
    plt.xlabel(' samples')
    plt.legend(loc="best")
    plt.title(title + ' - Truth vs predicted')
    conf = ':'.join(map(str, parameters_dict.values())).replace('[', '').replace(']', '').replace(' ', '').replace(':', '_').replace(',', '-')
    plt.savefig("plot_predictions_" + title + '_' + conf + '.png')
    plt.show()

# TRAINING OR TESTING
def plot_confusion_matrix(cm, class_names, parameters_dict = {}, title='', figsize=(8, 8)):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title.replace('_', ' '))
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix
    labels = np.around(cm.astype('int'))  # / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    conf = ':'.join(map(str, parameters_dict.values())).replace('[', '').replace(']', '').replace(' ', '').replace(':', '_').replace(',', '-')
    plt.savefig('plot_confusion_matrix_' + title + "_" + conf + '.png')

    # Calculate specificity and fall-out for each class
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    TN = TN.astype(float)

    # Specificity or true negative rate
    specificity = TN / (TN + FP)
    # Fall out or false positive rate
    fallout = 1 - specificity

    # Return mean value - to discuss
    return np.round(100 * specificity.mean(), 3), \
           np.round(100 * fallout.mean(), 3)