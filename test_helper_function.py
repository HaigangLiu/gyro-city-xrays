import unittest
from helper_functions import sampler_imbalanced, compute_cross_entropy_weights, f1_calculator_for_confusion_matrix
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from data_utilities import DataConstructor

class TestHelperFunctions(unittest.TestCase):

    def test_sampler_imbalanced(self):

        fake_labels = np.random.choice(2,150, int)
        sampler = sampler_imbalanced(fake_labels)

        ratio_of_ones = sum(fake_labels)/len(fake_labels)
        prob_of_zero = 1 - ratio_of_ones
        # if there is 1, 1, 0. then the prob to sample 1 is set to 1/3. The expected number of sample will be 2x1/3 = 2/3. The expected number of zeros is going to be 1*2/3 = 2/3. Hence they are equally likely to be sampled.

        for i, j in zip(fake_labels, sampler.weights):
            if i == 1:
                self.assertAlmostEqual(ratio_of_ones/(1 - ratio_of_ones), 1/j.item())

    def test_f1_calculator_for_confusion_matrix(self):

        y_true = np.array([0,1,1,1,1,1,1, 0, 0])
        y_pred = np.array([1,1,1,1,1,1,1, 0, 0])
        matrix = np.array([[2, 1], [0, 6]])

        f1 = f1_calculator_for_confusion_matrix(matrix)
        f1_standard = f1_score(y_true, y_pred)
        self.assertAlmostEqual(f1, f1_standard)

    def test_compute_cross_entropy_weights(self):

        DATA_DIR = "/Users/haigangliu/ImageData/ChestXrayData/"
        info_dir = '/Users/haigangliu/ImageData/Data_Entry_2017.csv'

        image_info = pd.read_csv(info_dir).iloc[0:1000,:]
        random_labels = np.random.randint(0, 2, image_info.shape[0], int)
        image_info['labels'] = random_labels

        torch_data_set = DataConstructor(DATA_DIR, image_info)
        positive_percentage = sum(random_labels)/image_info.shape[0]
        weights = compute_cross_entropy_weights(torch_data_set)

        self.assertAlmostEqual(weights[0], positive_percentage, places = 4)
        self.assertAlmostEqual(weights[1], 1-positive_percentage, places = 4)

if __name__ == '__main__':
    unittest.main()
