from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

import config.settings as stt
import config.constants as const
import src.dataset as dset


def func():
    dataset = dset.Dataset.getInstance()
    x_train = dataset.create_train_dataset_for_authentication(const.USER_NAME)
    X = x_train.transpose(0, 2, 1).reshape(x_train.shape[0] * 2, const.BLOCK_SIZE)
    x_test, y_test = dataset.create_test_dataset_for_authentication(const.USER_NAME)

    clf = OneClassSVM(gamma='auto').fit(X)
    #y_pred = clf.predict(x_test)
    

def plot_ROC(userid, fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - user '+ str(userid))
    plt.legend(loc="lower right")
    plt.show()
    


def compute_fpr_tpr(userid, positive_scores, negative_scores, plot = False):

    fpr, tpr, _ = metrics.roc_curve(positive_scores, negative_scores)
    roc_auc = metrics.auc(fpr, tpr)
    if( plot == True ):
        plot_ROC( userid, fpr, tpr, roc_auc )

    return roc_auc


if __name__ == "__main__":
    #compute_fpr_tpr(1, np.array([0, 0, 1, 1]), np.array([0, 2, 1, 1]), True)
    func()