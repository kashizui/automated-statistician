"""
datasets.py

Author: Stephen Koo

This module provides wrappers around loading and pre-processing datasets.

"""
from sklearn import (
    datasets as skdatasets,
    preprocessing,
)
__docformat__ = "restructuredtext en"

# Use 70-30
TRAINING_SIZE_RATIO = 0.7


class Dataset(object):
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.sample_size = self.data.shape[0]
        self._normalize()
        self._partition()

    def _partition(self):
        """
        Split the data into training/testing sets and stores as instance attributes.

        Returns: None

        """
        training_size = self.sample_size * TRAINING_SIZE_RATIO
        self.train_data = self.data[training_size:]
        self.test_data = self.data[:-training_size]
        self.train_target = self.target[training_size:]
        self.test_target = self.target[:-training_size]

    def _normalize(self):
        """
        Centers and scales the data and targets in-place to have unit variance.

        Returns: None

        """
        preprocessing.scale(self.data, with_mean=True, with_std=True, copy=False)
        preprocessing.scale(self.target, with_mean=True, with_std=True, copy=False)

    @staticmethod
    def from_file(path):
        """

        Args:
            path: string path to (csv?) file

        Returns: a new Dataset instance

        """
        with open(path, 'rb') as f:
            # TODO
            # return Dataset(x, y)
            pass


# Preload some datasets here
_diabetes = skdatasets.load_diabetes()
diabetes = Dataset(_diabetes.data, _diabetes.target)
