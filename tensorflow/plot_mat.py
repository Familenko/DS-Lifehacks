import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def plot_mat(model, X_valid, y_valid):
    np.seterr(divide='ignore')

    y_pred_valid = model.predict(X_valid)
    y_pred_valid = np.where(y_pred_valid > 0.5, 1, 0)
    cm = confusion_matrix(y_valid, y_pred_valid)

    print(classification_report(y_valid, y_pred_valid, zero_division=1))

    cm_procent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    try:
        classes = model.classes_
    except AttributeError:
        classes = [0, 1]

    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_procent[i, j]
            if i == j:
                s = cm.sum(axis=1)[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)

    try:
        ax = sns.heatmap(np.round(cm_procent,2), 
            annot=annot, 
            xticklabels=classes, 
            yticklabels=classes,
            cmap='Greens', fmt='')
    except TypeError:
        ax = sns.heatmap(np.round(cm_procent,2), 
            annot=annot,
            cmap='Greens', fmt='')

    ax.set(xlabel='Predict', ylabel='Actual')