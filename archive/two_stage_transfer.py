from setup_params import *
import logging
import torchvision
from customized_models import ModelCustomizer
import torch.nn as nn

logging.info('Transfer learning stage 1: ImageNet to ChestXray')
model_customizer = ModelCustomizer('densenet121')
image_net_model = model_customizer.network_modifier(14)
for param in image_net_model.parameters():
    param.requires_grad = False
image_net_model.classifier = nn.Sequential(nn.Linear(1024, 14), nn.Sigmoid())

import os, torch, torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
from torch.optim import lr_scheduler
import logging, time

from data_utilities import ChestXrayDataSetMultiLabel
from helper_functions import compute_multilabel_AUCs
from customized_models import ModelCustomizer
from training_engine import ModelTrainingAndTesting
import setup_params

cudnn.benchmark = True
number_of_epochs = setup_params.NUMBER_OF_EPOCHS
identifier = setup_params.identifier
MODEL_CACHE_DIR = setup_params.MODEL_CACHE_DIR
model_name = setup_params.MODEL_NAME
BATCH_SIZE = setup_params.BATCH_SIZE
device = setup_params.device
DATA_DIR = setup_params.DATA_DIR
learning_rate = setup_params.LEARNING_RATE
model_cache_dir = setup_params.MODEL_CACHE_DIR

N_CLASSES = 14
TEST_IMAGE_LIST = '/Users/haigangliu/ImageData/code/labels/test_list.txt'
TRAIN_IMAGE_LIST = '/Users/haigangliu/ImageData/code/labels/train_list.txt'
VALIDATION_IMAGE_LIST = '/Users/haigangliu/ImageData/code/labels/val_list.txt'

transform_train = transforms.Compose([transforms.Resize(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
train_dataset = ChestXrayDataSetMultiLabel(DATA_DIR, TRAIN_IMAGE_LIST,transform = transform_train)
train_dataloader = DataLoader(dataset = train_dataset,
                          batch_size = BATCH_SIZE,
                          shuffle = True,
                          num_workers = 4,
                          pin_memory = True,
                          drop_last = True
                          )

transform_val = transforms.Compose([transforms.Resize(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                                                    )])

validation_dataset= ChestXrayDataSetMultiLabel(DATA_DIR, VALIDATION_IMAGE_LIST, transform = transform_val)
validation_dataloader = DataLoader(dataset = validation_dataset,
                          batch_size = BATCH_SIZE,
                          shuffle = False,
                          num_workers = 4,
                          pin_memory = True,
                          drop_last = True
                          )

training_engine = ModelTrainingAndTesting(model_identifier = identifier, model_cache_dir = model_cache_dir, model_ft = image_net_model,  number_of_classes = 14, optimizer = None)

training_history, val_history = training_engine.start_engine(number_of_epochs, train_dataloader = train_dataloader, validation_dataloader = validation_dataloader)


data_transforms = {
    'train': transforms.Compose([
              transforms.Resize([256,256]),
      #  transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize([256, 256]),
     #   transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/Users/haigangliu/ImageData/CellData/chest_xray/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size= BATCH_SIZE, shuffle=True, num_workers = 4, drop_last=True) for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

stage2_model = training_engine.model_ft

for param in stage2_model.parameters():
    param.requires_grad = False
    stage2_model.classifier = nn.Sequential(nn.Linear(1024, 2), nn.Sigmoid())

training_engine = ModelTrainingAndTesting( model_identifier = identifier, model_cache_dir = model_cache_dir, model_ft = stage2_model, number_of_classes = 2 )

training_history = training_engine.start_engine(number_of_epochs, train_dataloader = dataloaders['train'],  validation_dataloader = dataloaders['test'])

predictions_on_testset = training_engine.report_on_testset(test_dataloader = dataloaders['test'], ten_crop = False)
