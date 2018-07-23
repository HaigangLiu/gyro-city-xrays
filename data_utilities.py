import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
import torch

class DataConstructor(Dataset):
    '''
     Args:
        param1 (str): Directory of the images
        param2 (pandas dataframe): Dataframe that contains the information of images.
        param3 (optional): pytorch image transformation operations

    Returns:
        Pytorch Dataset.
    '''
    def __init__(self, data_dir_, image_info_df, transform = None):
        try:
            self.data_dir_ = data_dir_
            self.image_names = list(image_info_df['Image Index'])
            self.labels = list(image_info_df.labels)
            self.transform = transform

        except KeyError:
            print('Expecting a dataframe with a column called Image Index (image names) and a column called labels')

    def __getitem__(self, index):

        image_name = os.path.join(self.data_dir_ + self.image_names[index])
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.image_names)

class CustomizedDataConstructor(Dataset):
    """
    This class is only supposed to be used in data augmentation.
    Hence, all labels from this dataset is 1, the positive and imbalanced class.
    """
    def __init__(self, data_dir_, transform = None, size = None):
        self.data_dir_ = data_dir_

        self.image_names = [i for i in os.listdir(self.data_dir_) if i.endswith('png')]

        if size is not None:
            self.image_names = list(np.random.choice(self.image_names, size, replace = False))

        self.labels = np.ones(len(self.image_names), dtype = int)
        self.transform = transform

    def __getitem__(self, index):

        image_name = os.path.join(self.data_dir_, self.image_names[index])
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.image_names)

class ChestXrayDataSetMultiLabel(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):

        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)



class SpecializedContructorForCAM(Dataset):
    def __init__(self, data_dir, info_df_dr, transform=None):
        self.info_df = self._process_info_df(info_df_dr)
        self.image_names = self.info_df['Image Index']
        self.labels = self.info_df['Finding Label']
        self.transform = transform
        self.data_dir = data_dir

    def _process_info_df(self, info_df_dr):
        bbox = pd.read_csv(info_df_dr)
        image_names = bbox['Image Index']
        labels = bbox['Finding Label']
        pixels = bbox.iloc[:, 2:6]
        pixels.columns = ['x', 'y', 'w', 'h']
        return  pd.concat([image_names,labels, pixels ], axis = 1)

    def __getitem__(self, index):
        image_name = self.data_dir + self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return self.image_names[index], image, label, list(self.info_df.loc[index,['x', 'y', 'w', 'h']])

    def __len__(self):
        return len(self.image_names)
