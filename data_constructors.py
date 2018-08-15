import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision import transforms
NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

class DataConstructor(Dataset):
    '''
    Args:
        image_folder (str): Directory of the images
        ground_truth (str): Dir of ground truth, should be a txt file
            special usage:
                1. set ground_truth = -1 for data augmentation in binary
                2. set ground_truth = 0-14  for data augmentation in multiclass

        transform (optional): pytorch image transformation operations;
            usually specified later; but we have a default setting to
            cut some slack.
        '''
    def __init__(self, image_folder, ground_truth, transform=None, image_size = 224):
        image_names = []
        labels = []

        if type(ground_truth) == int:
            print(f'populate all cases with label {ground_truth}')

            image_names = []
            for image_name in os.listdir(image_folder):
                if image_name.endswith(('png', 'jpg', 'jpeg')):
                    image_names.append(os.path.join(image_folder, image_name))

            if ground_truth == -1:
                labels = [[1]]*len(image_names) #binary; boost positive cases

            elif ground_truth <=14 and ground_truth >=0: #multiclass
                empty_string = [0]*14
                empty_string[ground_truth] = 1
                labels = [empty_string]*len(image_names)

            else:
                raise ValueError('ground_truth type not understood')

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
                            transforms.RandomResizedCrop(image_size),
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
    #use case 1
    test_loader_mode1 = DataConstructor('/Users/haigangliu/ImageData/ChestXrayData/', ground_truth=3)

    print(len(test_loader_mode1))
    print(test_loader_mode1[1])

    #use case 2
    test_loader_mode2 = DataConstructor('/Users/haigangliu/ImageData/ChestXrayData/', ground_truth= 'binary_label/train.txt')
    print(test_loader_mode2[1])
