import os
import pandas as pd
from shutil import copyfile, rmtree
import Augmentor #data augmentation pipeline
from tqdm import tqdm #progress bar
import numpy as np
from SamplingSchemes import DataSplitter
import random
import string #Generate a random name for temp folder

class ImbalancedClassAugmentor:

    '''
    Implement data augmentation for the minority group to alleviate the problem of imbalaced sampling.

        Args:
        HOME_DIR(string): home directory (the parant directory of data directory).
        DATA_DIR(string): directory where images are placed.
        output_folder_name(string): Name of the folder. A folder of the given name will be created under HOME_DIR.
        training_df (pandas dataframe): Dataframe of image names of the minority group, with a column called 'Image Index'.
        sample_size(int): number of new images to be generated.

        Returns:
        None.

    Note: A folder with new images will be created.
    '''

    def __init__(self, HOME_DIR, DATA_DIR, output_folder_name,training_df, sample_size = 10000):

        self.HOME_DIR = HOME_DIR
        self.DATA_DIR = DATA_DIR

        self.temp_dir = HOME_DIR +''.join(random.choices(string.ascii_uppercase + string.digits, k=20) )+ '/'
        self.output_dir = self.HOME_DIR + output_folder_name

        self.image_names_to_retain = training_df[training_df.labels == 1]['Image Index']
        self.sample_size = sample_size

        assert os.path.exists(self.output_dir) == False, "target folder already exists, remove it or specify a different target name"
        assert os.path.exists(self.DATA_DIR) == True, "Cannot find the x-ray images. Please make sure the directory is correct"
        assert type(sample_size) == int and sample_size > 0, "the sample size has to be a positive integer"

        self._file_copier()
        self._augmentor_initializer()
        self._tear_down()

    def _file_copier(self):
        try:
            os.mkdir(self.temp_dir)
        except FileExistsError:
            assert len(os.listdir(self.temp_dir)) == 0 , 'This dir exists and not empty.'
        finally:
            print('Start copying positive cases to a new folder... This might take while...')
            for image in tqdm(os.listdir(self.DATA_DIR)):
                if image in list(self.image_names_to_retain):
                    new_image = 'copy_' + image
                    copyfile(self.DATA_DIR + image, self.temp_dir + new_image)

    def _augmentor_initializer(self):

        augmentor_ppl = Augmentor.Pipeline(source_directory =self.temp_dir, output_directory = self.output_dir)
        augmentor_ppl.rotate( probability=1,
                              max_left_rotation=5,
                              max_right_rotation=5)
        augmentor_ppl.zoom_random(probability=0.5,
                                  percentage_area=0.9)
        augmentor_ppl.flip_top_bottom(probability=0.2)
        augmentor_ppl.sample(self.sample_size)

    def _tear_down(self):
        print('\n cleaning up ... ')
        rmtree(self.temp_dir)
        print('\n finished cleaning up ... ')
