import pandas as pd
import numpy as np
import torch
from torch.utils.data import sampler, Dataset
from PIL import Image
import os
import logging
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

def sampler_imbalanced(torch_dataset, ratio, sample_size = None, verbose = False):
    """
    Creates an imbalanced sampler which samples the minority class more frequently. Frequency is determined by specifying ratio in Args

    Args:
        torch_dataset (torch.dataset object): A torch dataset
        ratio (float): The target ratio for the minority group after imbalanced sampling.

    Returns: A torch sampler which can be used in creating dataloaders.

    Note:
    Currently only support binary classification. We assume the 1s are the minority group. Change the source code if your dataset is different.
    """
    weights = []
    try:
        labels = torch_dataset.labels
    except AttributeError:
        logging.warning('This dataset is made up by two smaller dataset.')
        labels = np.hstack([torch_dataset.datasets[0].labels, torch_dataset.datasets[1].labels])
    try:
        positive_cases = sum(labels)
        negative_cases = len(labels) - positive_cases
    except KeyError:
        logging.error("only supporting binary: 0, 1")
        return None

    class_1 = 1 #base class
    class_2 = (negative_cases/positive_cases)*(ratio/0.5) #minority class

    for label in labels:
        if label == 0:
            weights.append(class_1)
        elif label == 1:
            weights.append(class_2)
        else:
            raise ValueError("only supporting binary: 0, 1")
    if verbose:
        logging.info(weights)

    if sample_size:
        imbalanced_sampler = sampler.WeightedRandomSampler(weights, sample_size, replacement =True)
    else:
        imbalanced_sampler = sampler.WeightedRandomSampler( weights, len(labels) , replacement =True)
    return imbalanced_sampler

def compute_cross_entropy_weights(torch_dataset, verbose = False):
    """
    Compute the weights for cross entropy function.
    In order to compensate the imbalanced sample in the loss function.

    For more detail see Rajpurkar (2017).

    Args: Torch dataset
    Return: A float tensor of class weights
    """
    try:
        labels = torch_dataset.labels
    except AttributeError:
        logging.info('This dataset is made up by two smaller dataset.')
        labels = np.hstack([torch_dataset.datasets[0].labels, torch_dataset.datasets[1].labels])

    positives = sum(labels)
    negatives = len(labels) - positives

    try:
        #plus is negative cases or 0 (see Rajpurkar 2017)
        w_plus = negatives/len(labels)
        w_minus = positives/len(labels)

    except ValueError:
        logging.error("only supporting binary: 0, 1")
        return None

    if verbose:
        logging.info('the length of training set is {}'.format(len(labels)))
        logging.info('the ratio of positive cases is {:.4f}'.format(w_minus))
        logging.info('the ratio of negative cases is {:.4f}'.format(w_plus))

    weights_tensor = [w_minus, w_plus]
    return weights_tensor

import numpy as np
import torch

def f1_calculator_for_confusion_matrix(cf_matrix):

    """
    Args:
    cf_matrix:(numpy array): a 2 x 2 confusion matrix

    Returns (float): f1 score

    Note: Use sklearn.metrics.f1_score if the input data comes with 0s and 1s. This function only works with confusion matrix.
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

        smooth(int): Instead of calculating and plotting the raw data, a moving average will be calculated based on given smooth window. Default value is 0.

        early_stop(int): Truncate the list based on given terms. Default value is 0.

    Output:
        A list of metrics over epoch
        (Optional): Graphs

    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
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
