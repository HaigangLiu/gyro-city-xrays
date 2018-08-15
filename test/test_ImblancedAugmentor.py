import os, unittest
from ImbalancedClassAugmentor import ImbalancedClassAugmentor
import pandas as pd
import numpy as np
from shutil import rmtree

class TestImbalancedClassAugmentor(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        #only a small amount of images will be sampled to accelarate the testing process
        random_images_for_testing = np.random.choice(range(112120), 100, False)
        self.HOME_DIR = '/Users/haigangliu/ImageData/'
        self.DATA_DIR = "/Users/haigangliu/ImageData/ChestXrayData/"
        self.info_dir = self.HOME_DIR + 'Data_Entry_2017.csv'
        self.image_info = pd.read_csv(self.info_dir).iloc[random_images_for_testing,:]
        self.output_folder_name = '_test_folder'

        #generate some fake labels for testing
        random_labels = np.random.randint(0, 2, self.image_info.shape[0], int)
        self.image_info['labels'] = random_labels

        self.augmentor = ImbalancedClassAugmentor(HOME_DIR = self.HOME_DIR, DATA_DIR = self.DATA_DIR, output_folder_name = self.output_folder_name, training_df = self.image_info, sample_size = 10)

    @classmethod
    def tearDownClass(self):
        """
        Any folder starts with '_test' will be cleaned
        """
        print('\n deleting the folder for testing')

        for new_dir in os.listdir(self.HOME_DIR):
            if new_dir.startswith('_test'):
                print(new_dir)
                rmtree(self.HOME_DIR + new_dir)

    def test_image_quantity(self):
        """
        Make sure the right amount of images are generated.
        """
        number_of_images = len([i for i in os.listdir(self.HOME_DIR + self.output_folder_name)])
        self.assertEqual(number_of_images, 10)

    def test_make_sure_the_right_category_is_augmented(self):
        """
        Make sure the images with right labels (positive) are augmented.
        """
        right_categories = self.image_info[ self.image_info.labels == 1]['Image Index']

        for image_name in os.listdir(self.HOME_DIR + self.output_folder_name):
            original_name = '_'.join(image_name.split('_')[3:5])
            self.assertIn(original_name, list(right_categories))

    def test_execption_handling_path_existed(self):
        """
        Raise AssertionError when the path is already there.
        """
        try:
            collision_test = self.HOME_DIR + '_test_collision'
            os.mkdir(collision_test)
        except FileExistsError:
            pass
        finally:
            with self.assertRaises(AssertionError):
                ImbalancedClassAugmentor(HOME_DIR = self.HOME_DIR, DATA_DIR = self.DATA_DIR, output_folder_name = '_test_collision', training_df = self.image_info, sample_size = 10)

    def test_execption_type_error(self):
        """
        Raise AssertionError when sample size is not integer.
        """
        with self.assertRaises(AssertionError):
            ImbalancedClassAugmentor(HOME_DIR = self.HOME_DIR, DATA_DIR = self.DATA_DIR, output_folder_name = "_test_folder2", training_df = self.image_info, sample_size = 10.5)

    def test_tearing_down_procedure(self):
        """
        make sure there is only additional output folder after running the program.
        """
        number_of_folders = len([i for i in os.listdir(self.HOME_DIR)])
        self.augmentor = ImbalancedClassAugmentor(HOME_DIR = self.HOME_DIR, DATA_DIR = self.DATA_DIR, output_folder_name = '_test_folder3', training_df = self.image_info, sample_size = 2)
        #should only have one more folder after this
        number_of_folders_after = number_of_folders +1
        self.assertEqual(number_of_folders, number_of_folders_after -1)

if __name__ == '__main__':
    unittest.main()
