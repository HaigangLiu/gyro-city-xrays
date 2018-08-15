from parameter_sheet import *
import numpy as np
import pandas as pd
import os
import time
import logging
import torch, torchvision

from samplers import Sampler
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
from torch.utils.data import DataLoader

from ImbalancedClassAugmentor import DataAugmentor
from data_utilities import DataConstructor
from helper_functions import sampler_imbalanced, compute_cross_entropy_weights
from customized_models_v2 import ModelCustomizer
from trainingEngine_v2 import ModelTrainingAndTesting
from label_generator import label_generator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

train_file, val_file, test_file = label_generator(INFO_DIR, 'Image Index','Finding Labels')
train_dataset = DataConstructor(DATA_DIR, ground_truth=train_file)
val_dataset = DataConstructor(DATA_DIR, ground_truth=val_file)
test_dataset = DataConstructor(DATA_DIR, ground_truth=test_file)

DATA_AUGMENTATION = False
if DATA_AUGMENTATION:
    train_dataset = DataAugmentor(train_file, which_class=0, sample_size=50).concat_original_dataset(train_dataset)
else:
    logging.info('Original dataset has been used. No data augmentation')

#decide weights automatically
sampler_train = Sampler(train_dataset, 'subset', sample_size=75000).generate_sampler()
sampler_val = Sampler(test_dataset, 'other', sample_size=None).generate_sampler()
sampler_test = Sampler(val_dataset, 'other', sample_size=None).generate_sampler()

train_dataloader = DataLoader(dataset = train_dataset,
                              batch_size = BATCH_SIZE,
                              sampler = sampler_train,
                              #shuffle = SHUFFLE,
                              num_workers = NUM_WORKERS,
                              pin_memory = True,
                              drop_last = True)

val_dataloader = DataLoader(dataset = val_dataset,
                            batch_size = BATCH_SIZE,
                            #shuffle = False,
                            pin_memory = True,
                            sampler = sampler_val,
                            num_workers = NUM_WORKERS,
                            drop_last = True)

test_dataloader = DataLoader(dataset = test_dataset,
                             batch_size = BATCH_SIZE,
                           # shuffle = False,
                            pin_memory = True,
                            sampler = sampler_test,
                            num_workers = NUM_WORKERS,
                            drop_last = True)

model_customizer = ModelCustomizer(MODEL_NAME)
model_ft = model_customizer.network_modifier(NUM_CLASS).to(device)





# ALTERNATIVE_INITIALIZATION = False
# if ALTERNATIVE_INITIALIZATION:
#     model_ft = model_customizer.change_initialization()
#     logging.info('This architecture has adopted an alternative initialization scheme: Xavier Normal.')

# loaded_model = torchvision.models.densenet121(pretrained = True)

# for param in loaded_model.parameters():
#     param.requires_grad = False
#     loaded_model.classifier = nn.Sequential(nn.Linear(1024, 2), nn.Sigmoid())
#     model_ft = loaded_model.to(device)

# weights_tensor = compute_cross_entropy_weights(train_dataset, verbose = False)

# if MANUAL_WEIGHTS_TENSOR:
#     weights_tensor = [NEGATIVE_CASES_WEIGHT, POSITIVE_CLASS_WEIGHT]

# weights_tensor = [0.3, 0.7]
# weights_tensor = torch.tensor(weights_tensor).to(device)
# criterion = nn.CrossEntropyLoss(weight = weights_tensor)

# logging.info('Weight for negative (0) cases is {:.4f} and for positive cases (1) is {:.4f}'.format(weights_tensor[0], weights_tensor[1]))

# weights_tensor = [0.3, 0.7]

training_engine = ModelTrainingAndTesting(number_of_classes = NUM_CLASS, model_identifier ='identifier', model_cache_dir = MODEL_CACHE_DIR, model_ft = model_ft,
    # criterion = criterion
    )
training_history = training_engine.start_engine(NUMBER_OF_EPOCHS, train_dataloader = train_dataloader,  validation_dataloader = val_dataloader)

predictions_on_testset = training_engine.report_on_testset(test_dataloader = test_dataloader, ten_crop = False)
