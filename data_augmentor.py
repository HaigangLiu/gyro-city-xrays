import os
from shutil import copyfile, rmtree
import Augmentor #data augmentation pipeline
import random
import secrets #Generate a random name for temp folder
from parameter_sheet import LOG_DIR, DATA_DIR
from data_constructors import DataConstructor
import torch
class DataAugmentor:
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

    def __init__(self, ground_truth_file, which_class, sample_size = 1000):

        self.which_class = which_class
        self.file_list_to_augment = []
        for line in open(ground_truth_file, 'r'):
                items = line.split()
                image_name = items[0]
                label = items[1:]
                label = [int(i) for i in label]

                if label[self.which_class] == 1:
                    self.file_list_to_augment.append(image_name)

        self.sample_size = sample_size

        self._set_up()
        self._file_copier()
        self._augmentor_initializer()


    def _set_up(self):
        # copy all positive cases into a one folder.
        # will be delelted later
        temp = secrets.token_hex(4)
        self.temp_dir = os.path.join(LOG_DIR, temp)
        self.output_dir = os.path.join(LOG_DIR, 'augmentation_folder')

    def _file_copier(self):
        try:
            os.mkdir(self.temp_dir)
        except FileExistsError:
            assert len(os.listdir(self.temp_dir)) == 0 , 'This dir exists and not empty.'
        finally:
            print('Start copying positive cases to a new folder... This might take while...')
            for idx, image in enumerate(os.listdir(DATA_DIR)):
                if image in self.file_list_to_augment:
                    new_image = 'copy_' + image
                    from_ = os.path.join(DATA_DIR, image)
                    to_ = os.path.join( self.temp_dir, new_image)
                    copyfile(from_, to_)

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

    def concat_original_dataset(self, original_dataset):
        additional_dataset = DataConstructor(self.output_dir,
           ground_truth=self.which_class, transform=None)
        combined_dataset = torch.utils.data.ConcatDataset([additional_dataset, original_dataset])
        self._tear_down()
        return combined_dataset


if __name__ == '__main__':
    sample_usage = DataAugmentor('/Users/haigangliu/training_log/test.txt', which_class=0, sample_size=5000)
