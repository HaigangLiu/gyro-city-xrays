import numpy as np
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler, WeightedRandomSampler
import logging
import torch

class Sampler:
    '''
    generate sampler based on give sampler types:
    possible types include 'imbalance, subset, both, default'

    Args:
        dataset (pytorch data): dataset from which we sample
        sampler_type (string): 'imbalance, subset, both, default'
            'imbalance': will give a boost to minority group. Must pass in
                {1:100, 0:1}, thus the 1 group will be 100 times likely to be sampled
            'subset': a subset of dataset will be sampled. thus sample size
                must be speficied. Must be smaller than 112120
            'sample_size': needed if sampler_type = subset or both
            'prob_dict': a dictionary needed by sampler_type = imbalance or both
    '''
    def __init__(self, dataset, sampler_type, sample_size=None, prob_dict=None):

        __, sample_label = dataset[0]
        self.binary = len(sample_label) == 1
        self.dataset = dataset
        self.sampler_type = sampler_type
        self.sample_size = sample_size

        try: #one dataset
            self.labels_ = dataset.labels
        except AttributeError: # data augmentation
            self.labels_ = [d.labels for d in self.dataset.datasets]

        if self.prob_dict is None:
            positives = sum(torch.tensor(self.labels_).squeeze_())
            negatives = len(self.labels_) - positives
            w_plus = negatives.float()/len(self.labels_)
            w_minus = positives.float()/len(self.labels_)
            self.prob_dict = {1:w_plus, 0: w_minus}
        else:
            self.prob_dict = prob_dict

    def generate_sampler(self):
        #sanity check
        if self.sampler_type in ['both', 'imbalance'] and not self.binary:
            raise TypeError(f'scheme: {self.sampler_type} not work for multiclass')

        if self.sampler_type in ['both', 'subset'] and self.sample_size is None:
                raise ValueError('Must specify the sample size.')

        #dispatch calls
        if self.sampler_type == 'subset':
            logging.info('used a subset sampling scheme')
            return self._subset_sampler()
        elif self.sampler_type == 'both':
            logging.info('used a subset and imbalanced sampling scheme')
            return self._both_sampler()
        elif self.sampler_type == 'imbalance':
            logging.info('used an imbalance sampling scheme')
            return self._imbalanced_sampler()
        else:
            logging.info('used a sequential sampling scheme')
            return self._default_sampler()

    def _subset_sampler(self):
        indices = np.random.choice(len(self.dataset), self.sample_size, replace=False)
        return SubsetRandomSampler(indices)

    def _default_sampler(self):
        return SequentialSampler(self.dataset)

    def _both_sampler(self):
        weights = [self.prob_dict[label[0]] for label in self.labels_]
        return WeightedRandomSampler(weights,
                                    num_samples=self.sample_size,
                                    replacement=True)

    def _imbalanced_sampler(self):
        weights = [prob_dict[label[0]] for label in self.labels_]
        return WeightedRandomSampler(weights,
                                        num_samples=self.sample_size,
                                        replacement=True)
