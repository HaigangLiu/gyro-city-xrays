from parameter_sheet import *
import logging
import torch
from torch.utils.data import DataLoader

from samplers import Sampler
from data_augmentor import DataAugmentor
from data_constructors import DataConstructor
from helper_functions import customize_cross_entropy_function

from customized_models import ModelGenerator
from training_engine import ModelTrainingAndTesting
from label_generator import label_generator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
train_file, val_file, test_file = label_generator(INFO_DIR, 'Image Index','Finding Labels')

train_dataset = DataConstructor(DATA_DIR, ground_truth=train_file, image_size = IMAGE_SIZE)
val_dataset = DataConstructor(DATA_DIR, ground_truth=val_file, image_size = IMAGE_SIZE)
test_dataset = DataConstructor(DATA_DIR, ground_truth=test_file, image_size = IMAGE_SIZE)


if DATA_AUGMENTATION:
    train_dataset = DataAugmentor(ground_truth_file=train_file,
                                 # data_folder=DATA_DIR,
                                 # new_data_folder=LOG_DIR, #anywhere is ok
                                 which_class= AUGMENT_CLASS,
                                 sample_size=5000,
                                 ).concat_original_dataset(train_dataset)
else:
    logging.info('Original dataset has been used. No data augmentation')

sampler_train = Sampler(train_dataset, SAMPLER_TYPE, sample_size=SAMPLE_SIZE).generate_sampler()
sampler_val = Sampler(val_dataset, 'other', sample_size=None).generate_sampler()
sampler_test = Sampler(test_dataset, 'other',
 sample_size=None).generate_sampler()

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

model_ft = ModelGenerator(MODEL_NAME,
                    pretrain = PRETRAIN,
                    freeze_all_layers_before_last = TRANSFER_FROM_IMAGE_NET).\
            network_modifier(NUM_CLASS).\
            change_initialization(ALTERNATIVE_INITIALIZATION).\
            get_model().\
            to(device)

if BINARY:
    cross_entropy_loss = customize_cross_entropy_function(train_dataset, WEIGHTS_TENSOR)
else:
    cross_entropy_loss = None #use default

training_engine = ModelTrainingAndTesting(number_of_classes=NUM_CLASS,
                                          model_identifier=TRAINING_SESSION_IDENTIFIER,
                                          model_cache_dir=MODEL_CACHE_DIR,
                                          model_ft=model_ft,
                                          criterion=cross_entropy_loss
                                          )
training_history = training_engine.start_engine(NUMBER_OF_EPOCHS,
                                          train_dataloader=train_dataloader,
                                          validation_dataloader=val_dataloader)
predictions_on_testset = training_engine.report_on_testset(test_dataloader=test_dataloader,ten_crop = False)


from helper_functions import log_parser
log_parser(LOG_FILE, 'training', 'ROC', LOG_DIR)
log_parser(LOG_FILE, 'training', 'Acc', LOG_DIR)
log_parser(LOG_FILE, 'training', 'F1', LOG_DIR)

log_parser(LOG_FILE, 'validation', 'ROC', LOG_DIR)
log_parser(LOG_FILE, 'validation', 'Acc', LOG_DIR)
log_parser(LOG_FILE, 'validation', 'F1', LOG_DIR)
