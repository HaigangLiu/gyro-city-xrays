import os
from pathlib import Path

# -------------- Enviroment variables -------------
'''
DATA_DIR is the directory of all image files
TRAINING_SESSION_IDENTIFIER must be unique and should be informative
LOG_FOLDER_NAME is the name of the folder to be created to store log files
'''

DATA_DIR = "/Users/haigangliu/ImageData/ChestXrayData/"
TRAINING_SESSION_IDENTIFIER = 'multiclass_vgg11_new'
LOG_FOLDER_NAME = 'training_log'

HOME = str(Path.home())
try:
    os.mkdir(os.path.join(HOME, LOG_FOLDER_NAME))
    print(f'create a  training_log folder at {HOME}')
    MODEL_CACHE_DIR = LOG_DIR =  os.path.join(HOME, LOG_FOLDER_NAME)
except FileExistsError:
    MODEL_CACHE_DIR = LOG_DIR = os.path.join(HOME, LOG_FOLDER_NAME)
finally:
    print(f'logs and models will saved in training_log folder at {HOME}')

# ------------------- data step ---------------
'''
SUBSET_SAMPLING allows users to sample a small portion
of the training set every time to speed up computation
    REDUCED_SAMPLE_SIZE should be specified if SUBSET_SAMPLING is true

MANIUPULATE_TEST allows user to determine how many positive
cases to show up in the test set
    POSITIVE_CASES_NUM should be speficied if
    MANIUPULATE_TEST is true
    NEGATIVE_CASES_NUM should be speficied if
    MANIUPULATE_TEST is true
'''
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.2
IMAGE_SIZE = 224

SUBSET_SAMPLING = False
REDUCED_SAMPLE_SIZE = 10000 if SUBSET_SAMPLING else None

MANIUPULATE_TEST = True
POSITIVE_CASES_NUM = 50 if MANIUPULATE_TEST else None
NEGATIVE_CASES_NUM = 200 if MANIUPULATE_TEST else None #set to 50 if multilabel

# ------------------- Model step ---------------
'''
MODEL_NAME specifies the model choice for training. User can choose from
    resnet18, resnet34, resnet50, resnet152
    densenet121, densenet161, densenet169, densenet201
    vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn,
    inception_v3
BATCH_SIZE specifies how many images will be sent to gpu in on batch
PRETRAIN dictates whether or not CNN pretrained on ImageNet will be used
ALTERNATIVE_INITIALIZATION dictates whether or not non-default
    initialization scheme will be used '''
MODEL_NAME = 'vgg11_bn'
BATCH_SIZE = 16
NUM_WORKERS = 8
LEARNING_RATE = 0.001
PRETRAIN = True
OPTIMIZER = 'Adam'
NUMBER_OF_EPOCHS = 50
ALTERNATIVE_INITIALIZATION = False

# ------------------- which scenario ---------------
'''
GROUND_TRUTH_DIR needs an absolute path to the ground truth file
'''
BINARY = True
if BINARY:
    '''
    QUALIFIED_LABEL_LIST specifies what cases should be defined as positive
    the comprehensive label list is:
    ['Atelectasis',  'Cardiomegaly','Effusion', 'Pneumonia', 'Mass',
    'Nodule', 'Infiltrate', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis' 'Pleural_Thickening', 'Hernia']

    BINARY special option:
    IMBALANCED_SAMPLING will sample more from the minority group if true
    POSITIVE_CASE_RATIO will define to what extent minority group is boosted
    '''
    QUALIFIED_LABEL_LIST = ['Pneumonia','Consolidation', 'Effusion']
    GROUND_TRUTH_DIR = os.path.join(os.getcwd(),'binary_label/Data_Entry_2017.csv')
    IMBALANCED_SAMPLING = True
    POSITIVE_CASE_RATIO = 0.15 if IMBALANCED_SAMPLING else None

else:
    '''
    Multi-label special option:
    Li et al (2017) also published a splitted file
    set USE_DEFAULT_SPLIT = True to use this split.
    '''
    GROUND_TRUTH_DIR = os.path.join(os.getcwd(), 'multi_label/image_list.txt')
    USE_DEFAULT_SPLIT = True

# ------------------- Sanity Check  -------------------
'''
Make sure do not overwrite previous files
and train, test and val ratio add up to 1.
'''
file_log = os.path.join(LOG_DIR, TRAINING_SESSION_IDENTIFIER + '.log')
assert not os.path.exists(file_log),'TRAINING_SESSION_IDENTIFIER alreay exists.'
add_to_one = TRAIN_RATIO + TEST_RATIO + VALIDATION_RATIO
assert  add_to_one >= 0.99, 'the ratios are supposed to add up to 1'

# ------------------- Initialize the logger ---------------
'''
A global session of logging will be started and a log file
will be created in the pre-determined location.
'''

import logging
logging.basicConfig(level=logging.INFO,
                    filename= file_log,
                    format='%(asctime)-15s %(levelname)-8s %(message)s')

logging.info(f'the trained model will be cached in {MODEL_CACHE_DIR}')
logging.info(f'The cnn architecture is {MODEL_NAME}.')
logging.info(f'We consider {QUALIFIED_LABEL_LIST} as positive cases')
logging.info(f'training session will last for {NUMBER_OF_EPOCHS} epochs')
logging.info(f'the train, validate, test ratio is {TRAIN_RATIO}, {VALIDATION_RATIO}, {TEST_RATIO}')

if ALTERNATIVE_INITIALIZATION:
    logging.info('xavier normal used  in initialization')
else:
    logging.info('default initialization used')

if PRETRAIN:
    logging.info('pretrained model used  from ImageNet')
else:
    logging.info('no pretrained model used')
