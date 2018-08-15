import os
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def customize_cross_entropy_function(dataset, weights):
    """
    Generate a weighted cross entropy function based on the number
    of positive cases and number of negative cases.
    For more detail see Rajpurkar (2017).

    Args:
        dataset: a pytorch dataset
        weights: can be either 'auto' or a list
            'auto' means the negative cases will be given the positve weights
            and vice versa to balance out the gap in quantity
            list e.g., [0.4,0.3] gives user a handler to customize it.

    Return: A customized loss function
    """
    if weights == 'auto':
        try:
            labels = dataset.labels
        except AttributeError: #more than 1 dataset
            labels = [d.labels for d in self.dataset.datasets]

        positives = sum(torch.tensor(labels).squeeze_())
        negatives = len(labels) - positives

        w_plus = negatives.float()/len(labels) #cannot use long type
        w_minus = positives.float()/len(labels)
        weights = [w_minus, w_plus]

    elif len(weights) == 2:
        pass

    else:
        raise ValueError('weights must be either auto or a vector of length 2')

    logging.info(f'the ratio of positive cases is {weights[0]}')
    logging.info(f'the ratio of negative cases is {weights[1]}')

    weights_tensor = torch.tensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight = weights_tensor)

    return criterion


def f1_calculator_for_confusion_matrix(cf_matrix):

    """
    Args:
    cf_matrix:(numpy array): a 2 x 2 confusion matrix

    Returns (float): f1 score

    Note: Use sklearn.metrics.f1_score if the input data comes with 0s and 1s.
    This function only works with confusion matrix.
    """

    a = cf_matrix[1,1]
    b = cf_matrix[0,1]
    c = cf_matrix[1,0]

    p = a/(a+c)
    r = a/(a+b)
    logging.info('test')
    return 2*p*r/(p+ r)

def log_parser(log, keyword_1, keyword_2, smooth = 0, early_stop = 0):
    '''
    To parse the result copied and pasted from console.

    Input Args:
        log (string): The log file of tranining

        keyword_1(string): Choose from training or validation and etc.
        keyword_2(string): Choose from f1, auc, acc and etc. Certain combination does not exist.
        Note: Certain combination of keyword 1 and keyword 2 does not exist.

        smooth(int): Instead of calculating and plotting the raw data,
             a moving average will be calculated based on given smooth window. Default value is 0.
        early_stop(int): Truncate the list based on given terms. Default value is 0.

    Output:
        A list of metrics over epoch
        (Optional): Graphs

    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    result = []
    log_file = open(log)
    for line in log_file:
        if keyword_1 in line.split(' ') and keyword_2 in line.split(' '):
            try:
                result.append(float(line.split(' ')[-1]))
            except ValueError:
                pass
    if early_stop:
        result = result[:-early_stop]
    if smooth:
        result = pd.Series(result).rolling(smooth).mean()
        result.dropna(inplace = True)

    assert len(result) != 0, 'the result is empty. Try a different keyword combination'
    df1 = pd.DataFrame({"epoch":range(len(result)), "value" : result})
    f, ax = plt.subplots(1, 1)
    LABEL = keyword_1 + ' ' +  keyword_2
    ax.plot(range(len(df1["epoch"])), df1["value"], 'go-',label=LABEL, alpha = 0.5)
    ax.legend()
    plt.show()

    return result

def compute_multilabel_AUCs(ground_truth, pred_prob, N_CLASSES):

    '''
    Args:
        input of groud truth and pred prob are numpy arrays.
        Each column corresponds each image, and column is the class.
    Return:
        A list of auc for each group.
    '''

    AUROCs = []
    for i in range(N_CLASSES):
        try:
            AUROCs.append(roc_auc_score(ground_truth[:, i], pred_prob[:, i]))
        except ValueError:
            AUROCs.append(0)
    return AUROCs

def stats_calculator_binary(statistics):
    assert len(statistics) >= 2, 'expecting a list of binary prediction, probs and ground truth. Or at least probs and ground truth'

    try:
        y_pred_binary, y_pred_prob, y_true = statistics
    except ValueError:
        y_pred_prob, y_true = statistics
        y_pred_binary = np.array(y_pred_prob >= 0.5, dtype = int)

    roc_auc_score_ = roc_auc_score(y_true, y_pred_prob)
    f1_socre_ = f1_score(y_true, y_pred_binary)
    accuracy = sum(y_true == y_pred_binary)/len(y_pred_binary)
    conf_matrix = confusion_matrix(y_true, y_pred_binary)

    try:
        accuracy = accuracy.item()
    except AttributeError:
        pass

    return {
            "ROC": roc_auc_score_,
            "F1": f1_socre_,
            "Acc": accuracy,
            'Confusion_Matrix':conf_matrix
            }
