import os, logging
import torch, torchvision
from sklearn.metrics import f1_score, roc_auc_score, log_loss, confusion_matrix
import pandas as pd
import numpy as np
from helper_functions import stats_calculator_binary
# model_dir = "/Users/haigangliu/Desktop/ML_model_cache/DenseNet121.pth.tar"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Tester:
    """
    This class is used to calculate the summary statistics on the test set.
    The argument is a torch model and a torch dataloader
    The available metrics include f1 score, auc and accuracy.
    """

    def __init__(self, model_or_model_dir, test_loader):

        if type(model_or_model_dir) == str:
            self.model = torch.load(model_or_model_dir)
        else:
            self.model = model_or_model_dir

        self.test_loader = test_loader


    @staticmethod
    def _experiment_with_f1_threshold(y_true, y_pred_prob, threshold_candidates = None):

        if threshold_candidates is None:
            threshold_candidates = np.linspace(0, 0.5, 10)

        f1s = []
        for threshold in threshold_candidates:
            y_pred_binary = pd.Series(y_pred_prob > threshold, dtype = int)
            f1s.append(f1_score(y_true, y_pred_binary))
            threshold_candidates = threshold
        return {'f1': f1s,'threshold': threshold_candidates}

    def test_session(self,
                    verbose = True,
                     #threshold_auto = True
                     ):

        binary_guesses = []
        likelihood_prediction = []
        ground_truth = []

        for images, labels in self.test_loader:

            images = images.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(mode = False):

                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)

                likelihood_prediction.append(outputs[:,1])
                ground_truth.append(labels)
                binary_guesses.append(preds)

        gt = torch.cat(ground_truth, 0).cpu().detach().numpy()
        lp = torch.cat(likelihood_prediction, 0).cpu().detach().numpy()
        bg = torch.cat(binary_guesses, 0).cpu().detach().numpy()

        # if threshold_auto:
        #     result = Tester._experiment_with_f1_threshold(gt, lp)
        #     f1 = result['f1']; threshold = result['threshold']
        #     bg = pd.Series(lp.ravel() > threshold, dtype= int).values
        # else:
        #     f1 = None
        #     threshold = 0.5

        summary_stats = stats_calculator_binary([bg.ravel(), lp.ravel(), gt.ravel()])

        roc_score = summary_stats['ROC']
        acc = summary_stats['Acc']
        conf_matrix = summary_stats['Confusion_Matrix']
        # if f1 is None:
        f1 = summary_stats['F1']

        logging.info('the test roc score is {:.4f}'.format(roc_score))
        logging.info('the test f1 score is {:.4f}'.format(f1))
        logging.info('the test acc score is {:.4f}'.format(acc))
        logging.info('the test confusion_matrix is {}'.format(conf_matrix))

        return [bg.ravel(), lp.ravel(), gt.ravel()]

def ensemble(models_or_model_dirs, test_loader, threshold_auto = True):

    probs_list = []
    try:
        for model_or_model_dir in models_or_model_dirs:
            test_object = Tester(model_or_model_dir, test_loader)
            binary_prediction, probs, truth = test_object.test_session()
            probs_list.append(probs)
    except NameError:
        logging.info('Need to import Tester Module first')
        return None

    probs_list_df = pd.DataFrame(probs_list).transpose()
    probs_average = probs_list_df.mean(axis = 1)

    #two schemes of calculating f1. threshold_auto = True will try maximizing f1 score. threshold_auto = False will use 0.5
    if threshold_auto:
        result = Tester._experiment_with_f1_threshold(gt, lp)
        f1 = result['f1']; threshold = result['threshold']
    else:
        f1 = None; threshold = 0.5
    binary_guesses = pd.Series(probs_average > threshold, dtype= int)
    stats = stats_calculator_binary([binary_guesses, probs_average, truth])
    roc_score = stats['ROC']; acc = stats['Acc']
    conf_matrix = stats['Confusion_Matrix']

    if f1 is None: f1 = stats['F1']

    logging.info('the test roc score is {:.4f}'.format(roc_score))
    logging.info('the test f1 score is {:.4f}, and the threshold is {:.4f}'.format(f1, threshold))
    logging.info('the test acc score is {:.4f}'.format(acc))
    logging.info('the test confusion_matrix is {}'.format(conf_matrix))

    return [binary_guesses, probs_average, truth]
