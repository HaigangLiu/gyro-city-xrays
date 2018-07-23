from setup_params import *

from TestMode import Tester
import customized_models_v2
from SamplingSchemes import DataSplitter
from data_utilities import DataConstructor
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
import torch, torchvision
import time
import logging

logging.info('This is a transfer learning module')
logging.info('start loading the check point --- ')
cached_model = '/Users/haigangliu/Desktop/ML_model_cache/transfer_learning/TO_TRANSFER.tar'
loaded_model = torch.load(cached_model)
logging.info('finish loading the check point --- ')

#freeze the previous layers
for param in loaded_model.parameters():
    param.requires_grad = False
loaded_model.classifier = nn.Sequential(nn.Linear(1024, 2), nn.Sigmoid())

all_age = pd.read_csv(info_dir)
kids = all_age[all_age['Patient Age'] <= 5]

splitter = DataSplitter(kids, QUALIFIED_LABEL_LIST)
train_df, validation_df, test_df = splitter.random_split([0.6, 0.2, 0.2])

train_dataset = DataConstructor(DATA_DIR, train_df, transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.RandomHorizontalFlip(), transforms.ToTensor(), NORMALIZE]) )
validation_dataset, test_dataset = [DataConstructor(DATA_DIR, df,
                                 transforms.Compose([transforms.Resize(IMAGE_SIZE),transforms.ToTensor(),NORMALIZE])) for df in [validation_df, test_df]]

datasets = [train_dataset, validation_dataset, test_dataset]
train_dataloader, val_dataloader, test_dataloader = [DataLoader(dataset = dataset_, batch_size = BATCH_SIZE, shuffle = False,num_workers = NUM_WORKERS, pin_memory = True, drop_last = True) for dataset_ in datasets]

pos = train_df.labels.sum()/(len(train_df)); neg = 1-pos
weights_tensor = torch.tensor(torch.tensor([pos, neg])).to(device)

criterion = nn.CrossEntropyLoss(weight = weights_tensor)
optimizer_ft = torch.optim.Adam(loaded_model.parameters(), lr = LEARNING_RATE)

from trainingEngine import ModelTrainingAndTesting
training_engine = ModelTrainingAndTesting(model_name = model_name, model_cache_dir = MODEL_CACHE_DIR, model_ft = model_ft, optimizer_ft = optimizer_ft, criterion = criterion, weights_tensor = weights_tensor, scheduler= scheduler, device = device)
training_history = training_engine.start_engine(NUMBER_OF_EPOCHS, train_dataloader = train_dataloader,  validation_dataloader = val_dataloader)
predictions_on_testset = training_engine.report_on_test_set(test_dataloader = test_dataloader)
loaded_model_original = torch.load(cached_model)
original_test = Tester(loaded_model_original, test_dataloader)
predictions_on_testset_original = original_test.test_session()
