import numpy as np
import io
from sklearn import metrics
import PIL.Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import itertools
from sklearn import metrics
from mpl_toolkits.axes_grid1 import ImageGrid
import math
from tqdm import tqdm
import torch


############
## GRAPHs ##
def plot_signals(x, x_rec):

    plt.ioff()
    fig, axes = plt.subplots(x.size(0)//2, 2, figsize=(20, 10))

    for i, ax in enumerate(axes.reshape(-1)): 
        ax.plot(torch.arange(len(x[i, :])), x[i, :], color='blue')
        ax.plot(torch.arange(len(x[i, :])), x_rec[i, :], color='orange', ls='--')
        ax.title.set_text(f"signal{i}")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.clf()
    plt.close(fig)

    return image

def plot_curves(fpr, tpr, recall, precision):

    plt.ioff()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    img = axes[0].plot(fpr, tpr, color='orange')
    axes[0].plot([0, 1], [0, 1], transform=axes[0].transAxes, color='orange', ls='--')
    axes[0].title.set_text("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")

    img = axes[1].plot(recall, precision, color='blue')
    axes[1].axhline(y = 0.5, color='blue', ls='--')
    axes[1].title.set_text("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.clf()
    plt.close(fig)

    return image

def plot_confusion_matrix(tot_true_labels, tot_predicted_labels, classes, title='Confusion matrix', normalize=False, show_values=False, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = metrics.confusion_matrix(tot_true_labels, tot_predicted_labels)

    plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 10))
    img = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(img, ax=ax)
    ax.title.set_text(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks) 
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks) 
    ax.set_yticklabels(classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if show_values:
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.clf()
    plt.close(fig)

    return image


def plot_graph(x, y, xlabel, ylabel):
    """
    This function plots a curve and export it in image.
    """
    plt.ioff()
    fig, ax = plt.subplots()

    ax.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.clf()
    plt.close(fig)

    return image


def plot_histogram(hist, bins, x_tick_vertical=False, x_label="Value", y_label="Count", title='Histogram'):
    """
    This function prints and plots a histogram.
    """
    plt.ioff()
    fig, ax = plt.subplots()
    ax.hist(hist, bins=bins, alpha=0.5)

    if x_tick_vertical:
        plt.xticks(rotation='vertical')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.clf()
    plt.close(fig)

    return image


def plot_double_histogram(hist1, hist2, labels, bins, x_tick_vertical=False, x_label="Value", y_label="Count", title='Double Histogram'):
    """
    This function prints and plots two histograms on same graph.
    """
    plt.ioff()
    fig, ax = plt.subplots()
    ax.hist(hist1, bins=bins[0], alpha=0.5, label=labels[0])
    ax.hist(hist2, bins=bins[1], alpha=0.5, label=labels[1])

    if x_tick_vertical:
        plt.xticks(rotation='vertical')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='upper right')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.clf()
    plt.close(fig)

    return image


def plot_bar(labels, counts, x_tick_vertical=False, x_label="Value", y_label="Count", title='Bar plot'):

    """
    This function prints and plots a bar-plot.
    """
    plt.ioff()
    fig, ax = plt.subplots()
    ticks = range(len(counts))
    plt.bar(ticks,counts, align='center')
    plt.xticks(ticks, labels, rotation='vertical')

    if x_tick_vertical:
        plt.xticks(rotation='vertical')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.clf()
    plt.close(fig)
    
    return image


def plotImages(title, image):
    # image.shape = (CHANNEL,M,N)
    _,ax = plt.subplots()
    ax.title.set_text(title)
    ax.imshow(image.permute(1,2,0))


def saveGridImages(filename,imgs,n_colonne=10, figsize=100):
    plt.ioff()
    fig, axes = plt.subplots(nrows=math.ceil(len(imgs)/n_colonne), ncols=n_colonne, figsize=(figsize,figsize))
    for idx, image in enumerate(imgs):
        row = idx // n_colonne
        col = idx % n_colonne
        axes[row, col].axis("off")
        axes[row, col].set_title(idx)
        axes[row, col].imshow(image, cmap="gray", aspect="equal")
    for idx in range(len(imgs),math.ceil(len(imgs)/n_colonne)*n_colonne):
        row = idx // n_colonne
        col = idx % n_colonne
        axes[row, col].axis("off")
    plt.subplots_adjust(wspace=.0, hspace=.0)
    plt.savefig(filename + ".jpg")
    plt.clf()
    plt.close(fig)
    del fig


#############
## METRICs ##
def calc_prediction_agreement_rate(tot_predicted_labels, tot_predicted_labels_last):

    tot_predicted_labels = np.array(tot_predicted_labels)
    tot_predicted_labels_last = np.array(tot_predicted_labels_last)

    prediction_agreement = tot_predicted_labels_last == tot_predicted_labels
    prediction_agreement = np.count_nonzero(prediction_agreement)/len(prediction_agreement)

    return prediction_agreement, tot_predicted_labels.tolist()

def calc_prediction_agreement_rate_image_paths(tot_predicted_labels, tot_predicted_labels_last, tot_image_paths):
    tot_predicted_labels_last_split = {tot_image_paths[i]:tot_predicted_labels[i] for i in range(len(tot_image_paths))}

    predictionAgreement=0
    predictionAgreement_tot=0
    for key in tot_predicted_labels_last_split:
        if key not in tot_predicted_labels_last.keys():
            continue
        predictionAgreement_tot += 1

        if tot_predicted_labels_last_split[key] == tot_predicted_labels_last[key]:
            predictionAgreement += 1
    
    try:
        predictionAgreementRate = predictionAgreement/predictionAgreement_tot
    except ZeroDivisionError:
        predictionAgreementRate = 0.0

    return predictionAgreementRate, tot_predicted_labels_last_split

def calc_predictionAgreementRate(tot_predicted_labels, tot_predicted_labels_last, tot_image_paths=None):
    if tot_image_paths is None:
        return calc_prediction_agreement_rate(tot_predicted_labels, tot_predicted_labels_last)
    return calc_prediction_agreement_rate_image_paths(tot_predicted_labels, tot_predicted_labels_last, tot_image_paths)

def calc_FN_FP_histograms(tot_true_labels, tot_predicted_labels, tot, name, split):
    correct_labels, predicted_labels = np.array(tot_true_labels), np.array(tot_predicted_labels)

    FP_index = (((correct_labels==0).astype(int) + (predicted_labels==1).astype(int)) == 2)
    FN_index = (((correct_labels==1).astype(int) + (predicted_labels==0).astype(int)) == 2)

    tot_FP = np.array(tot)[FP_index]
    tot_FN = np.array(tot)[FN_index]

    tot_FP = tot_FP[np.isfinite(tot_FP)]
    tot_FN = tot_FN[np.isfinite(tot_FN)]

    hist_image = plot_double_histogram(tot_FP, tot_FN, ["FP","FN"], [100, 100], x_tick_vertical=False, x_label=(name + " value"), y_label="Count", title="Histograms " + name + " - "+split)

    return hist_image

def calc_predictionError_histograms(labels_in, predicted_in, soglia, name, split):
    labels, predicted = np.array(labels_in), np.array(predicted_in)
    
    labels_finite = labels[np.isfinite(labels)]
    predicted_finite = predicted[np.isfinite(labels)]

    wrong = np.abs(np.subtract(labels_finite,predicted_finite)) > soglia

    predictionError = labels_finite[wrong]

    hist_image = plot_histogram(predictionError, bins=100, x_tick_vertical=False, x_label=(name + " value"), y_label="Count", title="Histogram error prediction " + name + " - "+split)

    return hist_image


