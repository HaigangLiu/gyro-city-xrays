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
from customized_models_v2 import ModelCustomizer
from trainingEngine_v2 import ModelTrainingAndTesting
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
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia','Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
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
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

validation_dataset= ChestXrayDataSetMultiLabel(DATA_DIR, VALIDATION_IMAGE_LIST, transform = transform_val)
validation_dataloader = DataLoader(dataset = validation_dataset,
                          batch_size = BATCH_SIZE,
                          shuffle = False,
                          num_workers = 4,
                          pin_memory = True,
                          drop_last = True
                          )

test_dataset = ChestXrayDataSetMultiLabel(data_dir=DATA_DIR,
                                          image_list_file=TEST_IMAGE_LIST,
                                          transform=transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.TenCrop(224),
                                            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225]
                                                    )(crop) for crop in crops]))
                                          ]))

test_dataloader = DataLoader(dataset = test_dataset,
                          batch_size = BATCH_SIZE,
                          shuffle = False,
                          num_workers = 4,
                          pin_memory = True,
                          drop_last = True
                          )


# cached_m = torch.load('/Users/haigangliu/Desktop/ML_model_cache/multiclass/multi_class_.pth.tar')

training_engine = ModelTrainingAndTesting(model_identifier = identifier, model_cache_dir = model_cache_dir,  number_of_classes = 14, optimizer = None)
training_history, val_history = training_engine.start_engine(number_of_epochs, train_dataloader = train_dataloader, validation_dataloader = validation_dataloader)
predictions_on_testset = training_engine.report_on_testset(test_dataloader = test_dataloader,  ten_crop = True)
