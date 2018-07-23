import torch
import torchvision
import os
from torchvision import transforms, datasets
from torch.optim import lr_scheduler
import torch.nn as nn
import logging
import setup_params

from customized_models_v2 import ModelCustomizer
from trainingEngine_v2 import ModelTrainingAndTesting

transfer_scheme_1 = False #from ChestXray
transfer_scheme_2 = True #from imageNet

assert transfer_scheme_1 + transfer_scheme_2 <= 1, 'Choose at most one scheme'
torch.backends.cudnn.benchmark = True
identifier = setup_params.identifier
MODEL_CACHE_DIR = setup_params.MODEL_CACHE_DIR
model_name = setup_params.MODEL_NAME
BATCH_SIZE = 16
number_of_epochs = setup_params.NUMBER_OF_EPOCHS
device = setup_params.device
DATA_DIR = setup_params.DATA_DIR
learning_rate = setup_params.LEARNING_RATE
model_cache_dir = setup_params.MODEL_CACHE_DIR

data_transforms = {
    'train': transforms.Compose([
              transforms.Resize([299,299]),
      #  transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize([299, 299]),
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

model_customizer = ModelCustomizer(model_name)
model_ft = model_customizer.network_modifier(2)
model_ft = model_ft.to(device)

if transfer_scheme_1:
    logging.info('This is a transfer learning module. From chest_xray 14 data to children data obtained in Guangdong')
    logging.info('start loading the check point --- ')
    cached_model = '/Users/haigangliu/Desktop/ML_model_cache/3densent_run_with_tuning/DenseNet121_Consolidation_longer_train.pth.tar'
    loaded_model = torch.load(cached_model)
    logging.info('finish loading the check point --- ')

    for param in loaded_model.parameters():
        param.requires_grad = False
        loaded_model.classifier = nn.Sequential(nn.Linear(2048, 2), nn.Sigmoid())
        model_ft = loaded_model

elif transfer_scheme_2:
    logging.info('This is a transfer learning module. From imageNet to children data obtained in Guangdong')
    loaded_model = torchvision.models.inception_v3(pretrained = True)

    for param in loaded_model.parameters():
        param.requires_grad = False
        loaded_model.fc = nn.Sequential(nn.Linear(2048, 2), nn.Sigmoid())
        model_ft = loaded_model

else:
    pass #use original model



training_engine = ModelTrainingAndTesting( model_identifier = identifier, model_cache_dir = model_cache_dir, model_ft = model_ft, number_of_classes = 2 )

training_history = training_engine.start_engine(number_of_epochs, train_dataloader = dataloaders['train'],  validation_dataloader = dataloaders['test'])

predictions_on_testset = training_engine.report_on_testset(test_dataloader = dataloaders['test'], ten_crop = False)

# weights adjusted.. maybe not necessary?
# total = len(image_datasets['train'])
# negative = len(os.listdir(data_dir + 'train/NORMAL'))
# positive = total - negative
# weights_tensor = [positive/total, negative/total]
# weights_tensor = torch.tensor(weights_tensor).to(device)
# criterion = nn.CrossEntropyLoss(weight = weights_tensor)



# training_engine = ModelTrainingAndTesting(model_name = identifier, model_cache_dir = model_cache_dir, model_ft = model_ft, optimizer_ft = optimizer_ft, criterion = criterion, weights_tensor = weights_tensor, scheduler= scheduler, device = device)
