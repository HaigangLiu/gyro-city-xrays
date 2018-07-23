import os
import logging
import torch
#------------- Global Setup ----------------
identifier = 'multiclass_vgg'
model_log_name = identifier + '.log'
MODEL_NAME = 'vgg16_bn'

RANDOM_SEED = 1989
HOME_DIR = '/Users/haigangliu/ImageData/' #root
DATA_DIR = "/Users/haigangliu/ImageData/ChestXrayData/"
MODEL_CACHE_DIR = "/Users/haigangliu/Desktop/ML_model_cache/"
info_dir = HOME_DIR + 'Data_Entry_2017.csv'

POSITIVE_CASE_DEFINITION = {'Pneumonia Only': ['Pneumonia'],
                            'Consolidation Only': ['Consolidation'],
                            'Pneumonia or Consolidation':['Pneumonia','Consolidation' ],
                            'Effusion Only':['Effusion'],
                            'Original Settings': ['Pneumonia', 'Consolidation', 'Effusion']}
#------------- Preprocessing Choices ----------------
SET_AGE_THRESHOLD = False
QUALIFIED_LABEL_LIST = POSITIVE_CASE_DEFINITION['Pneumonia or Consolidation']

TRAIN_RATIO = 0.6; VALIDATION_RATIO = 0.2; TEST_RATIO = 0.2
MANIUPULATE_TEST = True
POSITIVE_CASES_NUM = 50 if MANIUPULATE_TEST else None
NEGATIVE_CASES_NUM = 250 if MANIUPULATE_TEST else None

DATA_AUGMENTATION = False
ADDITIONAL_POSITIVES = 20000 if DATA_AUGMENTATION else None
SUBSET_ADDITIONAL_POSITIVES = 3000 #only use some of fake data

AUGMENTED_FOLDER_NAME = 'output_consolidation_or_pneumonia/' if DATA_AUGMENTATION else None
#------------- Sampling Schemes For DataLoader----------------
SUBSET_SAMPLING = True
SEQUENTIAL_SAMPLING = False
IMBALANCED_SAMPLING = True

if SUBSET_SAMPLING:
    SUBSET_SAMPLING_SIZE = 45000
    REDUCED_EPOCHS = 50

if IMBALANCED_SAMPLING:
    POSITIVE_CASE_RATIO = 0.15

MANUAL_WEIGHTS_TENSOR = True
if MANUAL_WEIGHTS_TENSOR:
    POSITIVE_CLASS_WEIGHT = 0.87
    NEGATIVE_CASES_WEIGHT = 0.13

#------------- Model Choices ----------------

ALTERNATIVE_INITIALIZATION = False
#------------- Training Settings ----------------
from torchvision import transforms
NUMBER_OF_EPOCHS = 50 if not SUBSET_SAMPLING else REDUCED_EPOCHS
BATCH_SIZE = 8 if MODEL_NAME not in ['resnet152','vgg16_bn'] else 2 #resnet152 is too big
LEARNING_RATE = 0.001
NUM_WORKERS = 8
NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#------------- Some manual tuning hanlders ----------------
IMAGE_SIZE = 224
# if MODEL_NAME.startswith('inception'):
#     IMAGE_SIZE = 299
# else:
#     IMAGE_SIZE = 224
#------------- Sanity Checks (Make sure seteps make senese) ----------------
assert SUBSET_SAMPLING + SEQUENTIAL_SAMPLING + IMBALANCED_SAMPLING >= 1, 'choose at least one scheme for sampling in dataloader'
assert TRAIN_RATIO + TEST_RATIO + VALIDATION_RATIO == 1, 'the ratios are supposed to add up to 1'
if DATA_AUGMENTATION and os.path.exists(HOME_DIR + AUGMENTED_FOLDER_NAME):
    print('Note that directory {} already exists.\nWe will include the existing images in the training set instead of generating the new ones again.'.format(HOME_DIR + AUGMENTED_FOLDER_NAME))
# assert not os.path.exists(MODEL_CACHE_DIR + identifier +  'pth.tar'), 'Name Collision: Use a different model name'
# assert not os.path.exists(MODEL_CACHE_DIR + model_log_name), 'Name Collision: Use a different log name'

#------------- pass the check, start logging ----------------

logging.basicConfig(level=logging.INFO,
        filename= MODEL_CACHE_DIR + model_log_name,
        format='%(asctime)-15s %(levelname)-8s %(message)s')
logging.info('The cnn architecture is {}'.format(MODEL_NAME))
# logging.info('We consider the following list as postive cases {}'.format(QUALIFIED_LABEL_LIST))
