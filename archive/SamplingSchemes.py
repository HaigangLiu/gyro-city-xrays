import pandas as pd
import numpy as np
import sys
import torch

class DataSplitter:
    """
    This class intends to split train and test data randomly for given ratios.
    The input is the directory of the patient summary file, the qualified label list defining what is positive and what is negative.
    Another feature is there is no patient overlap among train, test and evaluation set.
    The output is three dataframes, train, eval, and test.
    """
    def __init__(self, info_dir_or_info_df, qualified_label_list, age_threshold = False, set_seed = True):
        if set_seed:
            np.random.seed(1989)
        if type(info_dir_or_info_df) == str:
            self.image_info = pd.read_csv(info_dir_or_info_df)
        else:
            self.image_info = info_dir_or_info_df

        if age_threshold:
            self.image_info[self.image_info['Patient Age'] > 5]

        self.qualified_label_list = qualified_label_list
        self.image_info_w_new_column = DataSplitter._append_pneumonia_labels(self.image_info, self.qualified_label_list)

    @staticmethod
    def _append_pneumonia_labels(df, qualified_label_list):
        """
        Convert the labels into 0 and 1 based on the following rules:
        1. As long as labels in qualified_label_list shows up in the label, then positive
        2. Otherwise, negative.
        """
        assert isinstance(df, pd.DataFrame), "the first input needs to be a pandas dataframe"

        new_list = []
        for i in df['Finding Labels'] :
            for disease in i.split('|'):
                if disease in qualified_label_list:
                    new_list.append(1)
                    break
            else:
                new_list.append(0)
        df['labels'] =  np.array(new_list)
        return df

    @staticmethod
    def _data_split_by_patient(image_info, ratio):
        """
        A customized train test split function;
        cannot use the one in sklearn because we cannot
        split along index; we have to split along individuals

        Following this algorithm, training set is always larger than specified. For instance, if we choose id = [patient 1, patient 2, patient 2, patient 3] with np.random.choice, patient 3 might have other images under his or her name. All these additionally images will automatically be included in training set.
        """
        unique_patients = len(image_info.groupby(["Patient ID"]).count())
        train_size = round(unique_patients*ratio, 0)

        train_index = np.random.choice(image_info['Patient ID'],
                                         int(train_size),
                                         replace = False)

        training_set = image_info[image_info['Patient ID'].isin(train_index)]
        test_set = image_info[~image_info['Patient ID'].isin(train_index)]
        return [training_set, test_set]

    def random_split(self, split_ratio):
        """
        This method is pure random split, and a list of ratios are expected.
        """
        assert isinstance(split_ratio, list), 'Needs a list of three values, ratio of train, validation and test. E.g., [0.7, 0.2, 0.1]'
        assert 0.999 <sum(split_ratio) <= 1.0, 'Expect a vector whose elements sums up to 1.'

        r_train, r_validation, r_test = split_ratio
        train, val_and_test = DataSplitter._data_split_by_patient(self.image_info_w_new_column, r_train)
        validation, test = DataSplitter._data_split_by_patient(val_and_test, r_validation/(r_validation + r_test))

        return [train, validation, test]

    def random_split_with_fixed_positives(self,  number_of_patients_with_positive_label, number_of_patients_with_negative_label,
        split_ratio,
        verbose = False,
        agreement_among_labels = True):

        """
        This method is involves a little tweak, since in the test set, you are allowed to set the desired number of positive patients (the second postional argument) and negative patients (the third positional argument). You can also specify the split ratio between the training set and the eval set by passing a float number as the first positional argument.

        Note that here might be disagreement with regard to the labels. Hence, setting agreement_among_labels = True can keep only patients whose Chest X-rays readings are consistent. Otherwise, you will be picking patients with at least one positive image readings.
        """
        assert type(split_ratio) == float and split_ratio < 1 and split_ratio>0, 'Expect a float number between 0 and 1 as split_ratio'
        assert type(number_of_patients_with_positive_label) == int, 'Expect an integer as patient counts (positive cases)'
        assert type(number_of_patients_with_negative_label) == int, 'Expect an integer as patient counts (negative cases)'

        dataframe_groupby_patient = self.image_info_w_new_column.groupby('Patient ID').mean().reset_index()

        if agreement_among_labels:
            positive_patient_group = dataframe_groupby_patient[dataframe_groupby_patient.labels == 1 ]
        else:
            positive_patient_group = dataframe_groupby_patient[dataframe_groupby_patient.labels > 0]

        negative_patient_group = dataframe_groupby_patient[dataframe_groupby_patient.labels == 0]

        negative_patient_IDs = np.random.choice(negative_patient_group['Patient ID'], number_of_patients_with_negative_label, replace = False)

        try:
            positive_patient_IDs = np.random.choice(positive_patient_group['Patient ID'], number_of_patients_with_positive_label, replace = False)

        except ValueError:
            print('cannot find {} qualified patients, try to select fewer postive patients'.format(number_of_patients_with_positive_label))
            return None

        patients_IDs_test_set = np.hstack([negative_patient_IDs,positive_patient_IDs])

        others = self.image_info_w_new_column[~self.image_info_w_new_column['Patient ID'].isin(patients_IDs_test_set)]
        test = self.image_info_w_new_column[self.image_info_w_new_column['Patient ID'].isin(patients_IDs_test_set)]
        train, validation =  DataSplitter._data_split_by_patient(others, split_ratio)

        if verbose:
            print('{} postive patients will be selected and {} negative patients will be selected in the test set '.format(number_of_patients_with_positive_label, number_of_patients_with_negative_label))
            print('{} patients were selected in the test set because there might be multiple images for one patient'.format(test.shape[0]))
            print('For the rest of the dataset, we split train and validation set with a split ratio {:.4f}'.format(split_ratio))

        return [train, validation, test]

if __name__ == '__main__':

    temp = '/Users/haigangliu/ImageData/Data_Entry_2017.csv'
    splitter = DataSplitter(temp, ['Pneumonia'], age_threshold = 5)
    a, b, c = splitter.random_split([0.7, 0.2, 0.1])
    print(a.shape)
    print(b.shape)
    print(c.shape)

    splitter2 = DataSplitter(temp, ['Pneumonia', 'Consolidation'])
    a, b, c = splitter2.random_split_with_fixed_positives(100, 400, 0.7)
    print(a.shape)
    print(b.shape)
    print(c.shape)
