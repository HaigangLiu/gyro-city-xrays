from setup_params import *

import numpy as np
import pandas as pd
import os, time
import logging

import torch, torchvision
from torch.optim import lr_scheduler
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data.sampler import SequentialSampler,SubsetRandomSampler

import torch.nn.init as init
import pickle
from data_utilities import DataConstructor, CustomizedDataConstructor
from ImbalancedClassAugmentor import ImbalancedClassAugmentor
from helper_functions import sampler_imbalanced, compute_cross_entropy_weights

from customized_models_v2 import ModelCustomizer
from trainingEngine_v2 import ModelTrainingAndTesting
from SamplingSchemes import DataSplitter

#torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

splitter = DataSplitter(info_dir, QUALIFIED_LABEL_LIST, age_threshold = SET_AGE_THRESHOLD)
if MANIUPULATE_TEST:
    train_df, validation_df, test_df = splitter.random_split_with_fixed_positives(POSITIVE_CASES_NUM, NEGATIVE_CASES_NUM, TRAIN_RATIO)
else:
    train_df, validation_df, test_df = splitter.random_split([TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO])

logging.info('The size of training set, validation set and test is {}, {} and {}, respectively.'.format(len(train_df), len(validation_df), len(test_df)))

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

train_dataset = DataConstructor(DATA_DIR, train_df ,
                                 transforms.Compose([
        transforms.Resize([256,256]),
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), NORMALIZE
    ]) )

if DATA_AUGMENTATION:
    #data augmentation changes on the dataset level, while imbalanced sampling
    #make changes on the sampling/dataloader part.
    logging.info('Data augmentation module has been applied, and {} more images are included in the postive category'.format(ADDITIONAL_POSITIVES))

    augmented_data_dir = HOME_DIR + AUGMENTED_FOLDER_NAME

    if not os.path.exists(augmented_data_dir):
        aug = ImbalancedClassAugmentor(HOME_DIR = HOME_DIR, DATA_DIR = DATA_DIR, output_folder_name = AUGMENTED_FOLDER_NAME, training_df = train_df, sample_size = SUBSET_ADDITIONAL_POSITIVES)

    additional_positive_dataset = CustomizedDataConstructor(augmented_data_dir, transform = transforms.Compose([
             transforms.Resize([256,256]),
             transforms.RandomResizedCrop(IMAGE_SIZE),
             transforms.RandomHorizontalFlip(),
            NORMALIZE]),
        size = ADDITIONAL_POSITIVES)

    train_dataset = torch.utils.data.ConcatDataset([additional_positive_dataset, train_dataset])
else:
    logging.info('Original dataset has been used. No data augmentation')

if IMBALANCED_SAMPLING:
    logging.info('The imbalanced sampling modules applied.')
    logging.info(f'the ratio of minority group has been cranked up to {POSITIVE_CASE_RATIO}.')

    sampler_ = sampler_imbalanced(train_dataset, verbose = False, sample_size = len(train_dataset), ratio = POSITIVE_CASE_RATIO)
    SHUFFLE = False

    if SUBSET_SAMPLING:
        logging.info('The subset sampling modules applied. The training size is {}'.format(SUBSET_SAMPLING_SIZE))
        sampler_ = sampler_imbalanced(train_dataset, verbose = False, sample_size = SUBSET_SAMPLING_SIZE, ratio = POSITIVE_CASE_RATIO)

elif SUBSET_SAMPLING:
    logging.info('Using subset sampling scheme. Only a subset of {} images are used as training set every epoch.'.format(SUBSET_SAMPLING_SIZE))

    indices = np.random.choice(len(train_dataset), SUBSET_SAMPLING_SIZE, replace = False)
    sampler_ = SubsetRandomSampler(indices)
    SHUFFLE = False

else:
    logging.info('The default sequential sampling scheme has been applied.')
    sampler_ = None #use the default one
    SHUFFLE = True

train_dataloader = DataLoader(dataset = train_dataset,
                              batch_size = BATCH_SIZE,
                              sampler = sampler_,
                              shuffle = SHUFFLE,
                              num_workers = NUM_WORKERS,
                              pin_memory = True,
                              drop_last = True)

val_dataset = DataConstructor(DATA_DIR, validation_df,
                              transforms.Compose([
          transforms.Resize(IMAGE_SIZE),
          transforms.ToTensor(),
          NORMALIZE]))

test_dataset = DataConstructor(DATA_DIR, test_df,
                                 transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        NORMALIZE]))

if SUBSET_SAMPLING:
    indices_val = np.random.choice(len(val_dataset), min(round(SUBSET_SAMPLING_SIZE*0.3), len(val_dataset)), replace = False)
    sampler_val = SubsetRandomSampler(indices_val)

    indices_test = np.random.choice(len(test_dataset), min(round(SUBSET_SAMPLING_SIZE*0.2), len(test_dataset)), replace = False)
    sampler_test = SubsetRandomSampler(indices_test)

else:
    sampler_val = SequentialSampler(val_dataset)
    sampler_test = SequentialSampler(test_dataset)

val_dataloader = DataLoader(dataset = val_dataset,
                            batch_size = BATCH_SIZE,
                            shuffle = False,
                            pin_memory = True,
                            sampler = sampler_val,
                            num_workers = NUM_WORKERS,
                            drop_last = True)

test_dataloader = DataLoader(dataset = test_dataset,
                            batch_size = BATCH_SIZE,
                            shuffle = False,
                            pin_memory = True,
                            sampler = sampler_test,
                            num_workers = NUM_WORKERS,
                            drop_last = True)

cache_test_loader_dir = MODEL_CACHE_DIR + identifier + '_test_dataloader' + '.pickle'
with open(cache_test_loader_dir, 'wb') as handle:
    pickle.dump(test_dataloader, handle, protocol=pickle.HIGHEST_PROTOCOL)
logging.info('The test dataloader has been cashed as {}'.format(cache_test_loader_dir))


ALTERNATIVE_INITIALIZATION = False
if ALTERNATIVE_INITIALIZATION:
    model_ft = model_customizer.change_initialization()
    logging.info('This architecture has adopted an alternative initialization scheme: Xavier Normal.')

model_customizer = ModelCustomizer(MODEL_NAME)
#model_ft = model_customizer.network_modifier(2).to(device)
loaded_model = torchvision.models.densenet121(pretrained = True)

for param in loaded_model.parameters():
    param.requires_grad = False
    loaded_model.classifier = nn.Sequential(nn.Linear(1024, 2), nn.Sigmoid())
    model_ft = loaded_model.to(device)

weights_tensor = compute_cross_entropy_weights(train_dataset, verbose = False)

if MANUAL_WEIGHTS_TENSOR:
    weights_tensor = [NEGATIVE_CASES_WEIGHT, POSITIVE_CLASS_WEIGHT]

weights_tensor = torch.tensor(weights_tensor).to(device)
criterion = nn.CrossEntropyLoss(weight = weights_tensor)

logging.info('Weight for negative (0) cases is {:.4f} and for positive cases (1) is {:.4f}'.format(weights_tensor[0], weights_tensor[1]))

training_engine = ModelTrainingAndTesting(number_of_classes = 2, model_identifier =identifier, model_cache_dir = MODEL_CACHE_DIR, model_ft = model_ft, criterion = criterion)
training_history = training_engine.start_engine(NUMBER_OF_EPOCHS, train_dataloader = train_dataloader,  validation_dataloader = val_dataloader)
predictions_on_testset = training_engine.report_on_testset(test_dataloader = test_dataloader, ten_crop = False)
