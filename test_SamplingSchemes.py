import unittest
import SamplingSchemes
import pandas as pd
import numpy as np

class TestSamplingSchemes(unittest.TestCase):
    def setUp(self):
        self.file_path = '/Users/haigangliu/ImageData/Data_Entry_2017.csv'
        self.patient_info_dataframe = pd.read_csv(self.file_path)
        self.MOE = 0.05

    def test_random_split_pneumonia(self):
        ratio = [0.7, 0.2, 0.1]
        splitter = SamplingSchemes.DataSplitter(self.file_path, ['Pneumonia'])
        train, evaluation, test= splitter.random_split(ratio)

        self.assertTrue((ratio[0]-self.MOE)*len(self.patient_info_dataframe) <= len(train) <= (ratio[0] + self.MOE)*len(self.patient_info_dataframe) )
        self.assertTrue((ratio[1]-self.MOE)*len(self.patient_info_dataframe) <= len(evaluation) <= (ratio[1] + self.MOE)*len(self.patient_info_dataframe) )
        self.assertTrue((ratio[2]-self.MOE)*len(self.patient_info_dataframe) <= len(test) <= (ratio[2]+self.MOE)*len(self.patient_info_dataframe) )
        self.assertEqual(len(train) + len(test) + len(evaluation),len(self.patient_info_dataframe) )

    def test_random_split_more_than_one_disease(self):

        ratio = [0.75, 0.15, 0.1]
        splitter = SamplingSchemes.DataSplitter(self.file_path, ['Pneumonia', 'Consolidation'])
        train, evaluation, test= splitter.random_split(ratio)

        self.assertTrue((ratio[0]-self.MOE)*len(self.patient_info_dataframe) <= len(train) <= (ratio[0] + self.MOE)*len(self.patient_info_dataframe) )
        self.assertTrue((ratio[1]-self.MOE)*len(self.patient_info_dataframe) <= len(evaluation) <= (ratio[1] + self.MOE)*len(self.patient_info_dataframe) )
        self.assertTrue((ratio[2]-self.MOE)*len(self.patient_info_dataframe) <= len(test) <= (ratio[2]+self.MOE)*len(self.patient_info_dataframe) )
        self.assertEqual(len(train) + len(test) + len(evaluation),len(self.patient_info_dataframe) )

    def test_random_split_with_fixed_positives(self):
        ratio = 0.67; postive_cases = 200; negative_cases = 120
        splitter = SamplingSchemes.DataSplitter(self.file_path, ['Consolidation'])
        train, evaluation, test = splitter.random_split_with_fixed_positives(postive_cases, negative_cases, ratio)

        self.assertEqual(len(np.unique(test['Patient ID'])), postive_cases + negative_cases)
        self.assertTrue((ratio-self.MOE)*len(self.patient_info_dataframe) <= len(train) <= (ratio + self.MOE)*len(self.patient_info_dataframe))

    def test_random_split_with_fixed_positives_more_than_one_disease(self):
        ratio = 0.67; postive_cases = 20; negative_cases = 120
        splitter = SamplingSchemes.DataSplitter(self.file_path, ['Pneumonia','Consolidation'])
        train, evaluation, test = splitter.random_split_with_fixed_positives(postive_cases, negative_cases, ratio)

        self.assertEqual(len(np.unique(test['Patient ID'])), postive_cases + negative_cases)
        self.assertTrue((ratio-self.MOE)*len(self.patient_info_dataframe) <= len(train) <= (ratio + self.MOE)*len(self.patient_info_dataframe))

    def test_assertion_error_not_sum_to_one(self):
        ratio = [0.8, 0.2, 0.1]
        splitter = SamplingSchemes.DataSplitter(self.file_path, ['Pneumonia'])
        self.assertRaises(AssertionError, splitter.random_split, ratio)

    def test_random_split_with_fixed_positives_exception(self):
        ratio = 0.67; postive_cases = 6000; negative_cases = 120
        splitter = SamplingSchemes.DataSplitter(self.file_path, ['Pneumonia'])

        self.assertRaises(AssertionError, splitter.random_split_with_fixed_positives, ratio, postive_cases, negative_cases)

if __name__ == '__main__':
    unittest.main()
