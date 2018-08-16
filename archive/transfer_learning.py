from setup_params import *

from TestMode import Tester
from SamplingSchemes import DataSplitter

import pandas as pd
import torch, torchvision
import customized_models

from torchnet import meter
import time
from helper_functions import sampler_imbalanced, compute_cross_entropy_weights, f1_calculator_for_confusion_matrix
from TestMode import Tester
import torch.nn as nn
from helper_functions import stats_calculator_binary

logging.info('This is a transfer learning module')

logging.info('start loading the check point --- ')
cached_model = '/Users/haigangliu/Desktop/ML_model_cache/transfer_learning/TO_TRANSFER.tar'
loaded_model = torch.load(cached_model)
logging.info('finish loading the check point --- ')

all_age = pd.read_csv(info_dir)
kids = all_age[all_age['Patient Age'] <= 5]

splitter = DataSplitter(kids, QUALIFIED_LABEL_LIST)
train_df, validation_df, test_df = splitter.random_split([0.6, 0.2, 0.2])

from data_utilities import DataConstructor
from torch.utils.data import DataLoader

train_dataset = DataConstructor(DATA_DIR, train_df, transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.RandomHorizontalFlip(), transforms.ToTensor(), NORMALIZE]) )

validation_dataset, test_dataset = [DataConstructor(DATA_DIR, df,
                                 transforms.Compose([transforms.Resize(IMAGE_SIZE),transforms.ToTensor(),NORMALIZE])) for df in [validation_df, test_df]]

datasets = [train_dataset, validation_dataset, test_dataset]

train_dataloader, val_dataloader, test_dataloader = [DataLoader(dataset = dataset_, batch_size = BATCH_SIZE, shuffle = False,num_workers = NUM_WORKERS, pin_memory = True, drop_last = True) for dataset_ in datasets]

pos = train_df.labels.sum()/(len(train_df)); neg = 1-pos
weights_tensor = torch.tensor(torch.tensor([pos, neg])).to(device)

criterion = nn.CrossEntropyLoss(weight = weights_tensor)
optimizer_ft = torch.optim.Adam(loaded_model.parameters(), lr = LEARNING_RATE)

from main import training_step, validation_step

# def training_step(model, data_loader):
#     model.train()
#     loss_cum = 0
#     confusion_matrix = meter.ConfusionMeter(2, normalized = False)

#     for images, labels in data_loader:

#         images = images.to(device)
#         labels = labels.to(device)
#         optimizer_ft.zero_grad()

#         with torch.set_grad_enabled(mode = True):

#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss_cum = loss_cum + loss.item()
#             confusion_matrix.add(outputs.data, labels)

#             loss.backward()
#             optimizer_ft.step()

#     return [model, confusion_matrix.conf, loss_cum]

# def validation_step(model, data_loader):
#     model.eval()

#     loss_cum = 0
#     confusion_matrix = meter.ConfusionMeter(2, normalized = False)

#     for images, labels in data_loader:

#         images = images.to(device)
#         labels = labels.to(device)

#         with torch.set_grad_enabled(mode = False):

#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             loss_cum = loss_cum + loss.item()
#             confusion_matrix.add(outputs.data, labels)

#     return [confusion_matrix.conf, loss_cum]

for epoch in range(100):

    since = time.time()
    # loaded_model, cf_matrix_train, loss_train = training_step(loaded_model, train_dataloader)
    model_ft, loss_train, probs_train, ground_truth_train = training_step(model_ft, train_dataloader)
    summary_train =  stats_calculator_binary([probs_train, ground_truth_train])

    logging.info('the training confusion matrix is {}'.format(summary_train['Confusion_Matrix']))
    logging.info('the training f1 score is {:.4f}'.format(summary_train['F1']))
    logging.info('the training accuracy is {:.4f}'.format(summary_train['Acc']))
    logging.info('the training loss is {:.4f}'.format(loss_train/len(ground_truth_train)))

    loss_val, probs_val, ground_truth_val = validation_step(model_ft, val_dataloader)
    summary_val =  stats_calculator_binary([probs_val, ground_truth_val])

    logging.info('the validation confusion matrix is {}'.format(summary_val['Confusion_Matrix']))
    logging.info('the validation f1 score is {:.4f}'.format(summary_val['F1']))
    logging.info('the validation accuracy is {:.4f}'.format(summary_val['Acc']))
    logging.info('the validation loss is {:.4f}'.format(loss_val/len(ground_truth_val)))

    time_elapsed = time.time() - since
    tester = Tester(model_ft, test_dataloader)
    predictions = tester.test_session(verbose = True)


    # f1_score_train = f1_calculator_for_confusion_matrix(cf_matrix_train)

    # logging.info('the training confusion matrix is {}'.format(cf_matrix_train))
    # logging.info('the training f1 score is {:.4f}'.format(f1_score_train))
    # logging.info('the training loss is {:.4f}'.format(loss_train))

    # cf_matrix_val, loss_val = validation_step(loaded_model, val_dataloader)
    # f1_score_val = f1_calculator_for_confusion_matrix(cf_matrix_val)

    # tester = Tester(loaded_model, test_dataloader)
    # predictions = tester.test_session(verbose = True)

    # logging.info('the validation confusion matrix is {}'.format(cf_matrix_val))
    # logging.info('the validation f1 score is {:.4f}'.format(f1_score_val))
    # logging.info('the validation loss is {:.4f}'.format(loss_val))

    # time_elapsed = time.time() - since
    # logging.info('Training complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
    # logging.info("-"*20)



loaded_model_original = torch.load(cached_model)
original_test = Tester(loaded_model_original, test_dataloader)
original_test.test_session()
