import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision import transforms

from parameter_sheet import IMAGE_SIZE
NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

class DataConstructor(Dataset):
    '''
    Args:
        data_dir (str): Directory of the images
        ground_truth (str): Dir of ground truth, should be a txt file
        transform (optional): pytorch image transformation operations; usually
        specified later
        '''
    def __init__(self, image_folder, ground_truth, transform=None):
        image_names = []
        labels = []

        if type(ground_truth) == int:
            print(f'populate all cases with label {ground_truth}')

            image_names = []
            for image_name in os.listdir(image_folder):
                if image_name.endswith(('png', 'jpg', 'jpeg')):
                    image_names.append(os.path.join(image_folder, image_name))

            labels = [[ground_truth]]*len(image_names)

        else:
            with open(ground_truth, "r") as f:
                for line in f:
                    items = line.split()
                    image_name= items[0]
                    label = items[1:]
                    label = [int(i) for i in label]
                    image_name = os.path.join(image_folder, image_name)
                    image_names.append(image_name)
                    labels.append(label)

        self.image_names = image_names
        self.labels = labels

        if transform == None:
            self.transform = transforms.Compose([
                            transforms.Resize([256,256]),
                            transforms.RandomResizedCrop(IMAGE_SIZE),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(), NORMALIZE])
        else:
            self.transform = transforms

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

if __name__ == '__main__':

    test_loader_mode1 = DataConstructor('/Users/haigangliu/ImageData/ChestXrayData/', ground_truth =3)
    print(len(test_loader_mode1))
    print(test_loader_mode1[1])

    # test_loader_mode2 = DataConstructor('/Users/haigangliu/ImageData/ChestXrayData/', 'binary_label/test.txt')
    # print(len(test_loader_mode2))
    # print(test_loader_mode2[1])

    # test_loader_mode3 = DataConstructor('/Users/haigangliu/ImageData/ChestXrayData/', 'split/test.txt')
    # print(len(test_loader_mode2))
    # print(test_loader_mode2[1])



# 1. to do list
# unify the data constructor into a same format
# text file, with name and label
# 2. unifiy dataloaders
# class DataConstructor(Dataset):
#     '''
#      Args:
#         param1 (str): Directory of the images
#         param2 (pandas dataframe): Dataframe that contains the information of images.
#         param3 (optional): pytorch image transformation operations

#     Returns:
#         Pytorch Dataset.
#     '''
#     def __init__(self, data_dir_, image_info_df, transform = None):
#         try:
#             self.data_dir_ = data_dir_
#             self.image_names = list(image_info_df['Image Index'])
#             self.labels = list(image_info_df.labels)
#             self.transform = transform

#         except KeyError:
#             print('Expecting a dataframe with a column called Image Index (image names) and a column called labels')

#     def __getitem__(self, index):

#         image_name = os.path.join(self.data_dir_ + self.image_names[index])
#         image = Image.open(image_name).convert('RGB')
#         label = self.labels[index]

#         if self.transform is not None:
#             image = self.transform(image)

#         return image, label

#     def __len__(self):
#         return len(self.image_names)

# class SpecializedContructorForAugmentation(Dataset):
#     """
#     This class is only supposed to be used in data augmentation.
#     Hence, all labels from this dataset is 1, the positive and imbalanced class.
#     """
#     def __init__(self, data_dir_, transform = None, size = None):
#         self.data_dir_ = data_dir_

#         self.image_names = [i for i in os.listdir(self.data_dir_) if i.endswith('png')]

#         if size is not None:
#             self.image_names = list(np.random.choice(self.image_names, size, replace = False))

#         self.labels = np.ones(len(self.image_names), dtype = int)
#         self.transform = transform

#     def __getitem__(self, index):

#         image_name = os.path.join(self.data_dir_, self.image_names[index])
#         image = Image.open(image_name).convert('RGB')
#         label = self.labels[index]

#         if self.transform is not None:
#             image = self.transform(image)

#         return image, label

#     def __len__(self):
#         return len(self.image_names)

