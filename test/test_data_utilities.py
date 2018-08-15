import unittest
from data_utilities import DataConstructor, CustomizedDataConstructor
import pandas as pd
import numpy as np

class TestDataUtilities(unittest.TestCase):

    """
    Note that CustomizedDataConstructor has no label options, since all images will be positive. This is because CustomizedDataConstructor is a class built to handle data augmentation, and ususally only the minority class needs to be augmented.
    """

    @classmethod
    def setUpClass(self):
        self.complete_data_set_size = 112120
        self.DATA_DIR = "/Users/haigangliu/ImageData/ChestXrayData/"
        self.info_dir = '/Users/haigangliu/ImageData/Data_Entry_2017.csv'

        self.image_info = pd.read_csv(self.info_dir)
        self.random_labels = np.random.randint(0, 2, self.image_info.shape[0], int)
        self.image_info['labels'] = self.random_labels

        self.dataset_constructor_1 = DataConstructor(self.DATA_DIR, self.image_info)
        self.dataset_constructor_2 = CustomizedDataConstructor(self.DATA_DIR)


    def test_size_and_resolution_of_dataset(self):
         # for both dataset constructors
        self.assertEqual(self.image_info.shape[0], len(self.dataset_constructor_1))
        self.assertEqual(self.dataset_constructor_1[0][0].size, (1024, 1024)) #not empty and size is right!
        self.assertEqual(self.complete_data_set_size, len(self.dataset_constructor_2))
        self.assertEqual(self.dataset_constructor_2[0][0].size, (1024, 1024))

    def test_confirm_labels(self):
        samples = np.random.choice(range(112120), 20, False)
        # for both dataset constructors
        for i in samples:
            image, label = self.dataset_constructor_1[i]
            self.assertEqual(label, self.random_labels[i])

            image_2, label_2 = self.dataset_constructor_2[i]
            self.assertEqual(label_2, 1)

if __name__ == '__main__':
    unittest.main()
