"""
datasets.py

Author: Stephen Koo

This module provides wrappers around loading and pre-processing datasets.

"""
import numpy as np
from sklearn import (
    datasets as skdatasets,
    preprocessing,
)
__docformat__ = "restructuredtext en"

# Use 70-30
TRAINING_SIZE_RATIO = 0.7


class Dataset(object):
    def __init__(self, data, target, scale_predictors=True, scale_targets=False):
        """
        uses same format for data and target as sklearn
        """
        self.data = data
        self.target = target
        self.sample_size = self.data.shape[0]
        self._normalize(scale_predictors, scale_targets)
        self._partition()

    def _partition(self):
        """
        Split the data into training/testing sets and stores as instance attributes.

        Returns: None

        """
        training_size = self.sample_size * TRAINING_SIZE_RATIO

        # shuffle data and targets
        rng_state = np.random.get_state()
        np.random.shuffle(self.data)
        np.random.set_state(rng_state)
        np.random.shuffle(self.target)

        # partition and save
        self.train_data = self.data[training_size:]
        self.test_data = self.data[:-training_size]
        self.train_target = self.target[training_size:]
        self.test_target = self.target[:-training_size]

    def _normalize(self, scale_predictors, scale_targets):
        """
        Centers and scales the data and targets in-place to have unit variance.

        Returns: None

        """
        if scale_predictors:
            preprocessing.scale(self.data, with_mean=True, with_std=True, copy=False)

        if scale_targets:
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
diabetes = Dataset(_diabetes.data, _diabetes.target, scale_targets=True)

_iris = skdatasets.load_iris()
iris = Dataset(_iris.data, _iris.target)

