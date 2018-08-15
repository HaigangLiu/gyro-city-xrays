import os
import random
import pandas as pd

try:
    from parameter_sheet import QUALIFIED_LABEL_LIST
except ImportError:
    QUALIFIED_LABEL_LIST = None #mean user chose multi-class scheme instead
from parameter_sheet import TRAIN_RATIO, VALIDATION_RATIO,TEST_RATIO
from parameter_sheet import MODEL_CACHE_DIR

class_dict = {'Atelectasis': 0,
                'Cardiomegaly': 1,
                'Effusion':2,
                'Infiltration':3,
                'Mass':4,
                'Nodule':5,
                'Pneumonia':6,
                'Pneumothorax':7,
                'Consolidation':8,
                'Edema':9,
                'Emphysema':10,
                'Fibrosis':11,
                'Pleural_Thickening':12,
                'Hernia':13
                }

def label_generator(data_info_dir, col_image_name, col_name_ground_truth,  out_dir = None):

    summary = pd.read_csv(data_info_dir)[[col_image_name,col_name_ground_truth]]
    all_disease_names = list(class_dict.keys())

    if QUALIFIED_LABEL_LIST is not None:
        def converter(list_of_diseases):
            for disease_name in QUALIFIED_LABEL_LIST:
                    null = 1 if disease_name in list_of_diseases else 0
            return null

    else:
        def converter(list_of_diseases):
            null_string = [0]*14
            for disease_name in all_disease_names:
                if disease_name in list_of_diseases:
                    insertion_idx = class_dict[disease_name]
                    null_string[insertion_idx] = 1
            return null_string

    all_ = []
    train = []; train_patients = set()
    val = []; val_patients = set()
    test = []; test_patients = set()

    for idx, diagnosis in zip(summary[col_image_name], summary[col_name_ground_truth]):

        list_of_diseases = diagnosis.split('|')
        encoded_list_diseases = converter(list_of_diseases)

        try: #if list
            encoded_list_diseases.insert(0, idx)
            entry = encoded_list_diseases[:]
        except: #if int
            entry = [idx, encoded_list_diseases]

        all_.append(entry)
        rn = random.uniform(0, 1)

        # make sure no patients overlapping
        if idx in train_patients:
            train.append(entry)
        elif idx in val_patients:
            val.append(entry)
        elif idx in test_patients:
            test.append(entry)

        else:
            if rn < TRAIN_RATIO:
                train.append(entry)
                train_patients.update(idx)

            elif rn >= TRAIN_RATIO and rn <= TRAIN_RATIO + VALIDATION_RATIO:
                val.append(entry)
                val_patients.update(idx)
            else:
                test.append(entry)
                test_patients.update(idx)

    names = ['all.txt', 'train.txt', 'val.txt', 'test.txt']
    paths = []

    for idx, form in enumerate([all_, train, val, test]):
        abs_path = os.path.join(MODEL_CACHE_DIR, names[idx])
        pd.DataFrame(form).to_csv(abs_path, header=False, sep=' ', index=False)
        print(f'the scheme of data split has been saved in {abs_path}')
        paths.append(abs_path)
    return paths[1:]

if __name__ == '__main__':
    QUALIFIED_LABEL_LIST = None
    summary = label_generator('binary_label/Data_Entry_2017.csv', 'Image Index','Finding Labels')

    QUALIFIED_LABEL_LIST = ['Pneumonia', 'Consolidation', 'Effusion']
    summary = label_generator('binary_label/Data_Entry_2017.csv', 'Image Index','Finding Labels')
    print(summary)

