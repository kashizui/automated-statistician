"""
models.py

Author: Stephen Koo

This module provides wrappers around different learning methods that can be used
in a plug-and-play fashion by the automatic statistician.


"""
import inspect
import timeit
import numpy as np
from sklearn import (
    svm,
    linear_model,
    ensemble,
    metrics,
    preprocessing,
)
__docformat__ = "restructuredtext en"


class Model(object):
    """
    Abstract class for model thingy.
    """
    # Should be overridden in subclasses
    NUM_HYPERPARAMETERS = None

    @classmethod
    def fit(cls, dataset, hyperparameters, verbose=True):
        """

        Args:
            dataset: the Dataset object to train and test
            hyperparameters: a tuple of hyperparameters of size matching output from cls.get_num_hyperparameters

        Returns: (performance, runtime in seconds)

        """
        hp = cls._unpack(hyperparameters)
        # print "%s(%s)" % (cls.__name__, ', '.join(["%s=%s" % (key, value) for key, value in hp.iteritems()]))
        tic = timeit.default_timer()
        performance = cls._fit(dataset, **hp)
        toc = timeit.default_timer()
        if verbose:
            print "%s(%s) => (%s, %s)" % (cls.__name__, ', '.join(["%s=%s" % (key, value) for key, value in hp.iteritems()]), performance, toc - tic)
        return performance, toc - tic

    @classmethod
    def _score(cls, pred_target, true_target):
        raise NotImplementedError

    @classmethod
    def _unpack(cls, hyperparameters):
        """

        Args:
            hyperparameters: a tuple of hyperparameters of size matching output from cls.get_num_hyperparameters

        Returns: dict mapping hyperparameter name to rescaled value

        """
        raise NotImplementedError

    @classmethod
    def _fit(cls, dataset, **hyperparameters):
        """

        Args:
            dataset: the Dataset object to train and test
            hyperparameters: a tuple of hyperparameters of size matching output from cls.get_num_hyperparameters

        Returns: predicted targets for test set

        """
        raise NotImplementedError


class RegressionModel(Model):
    @classmethod
    def _score(cls, pred_target, true_target):
        # return np.mean((true_target - pred_target) ** 2)
        # currently returning coefficient of determination:
        return metrics.r2_score(true_target, pred_target)


class ClassificationModel(Model):
    @classmethod
    def _score(cls, pred_target, true_target):
        # FIXME this is iffy, if a class label happens to be missing from either true_target or pred_target
        lb = preprocessing.LabelBinarizer()
        lb.fit(np.concatenate([true_target, pred_target]))
        # ROC AUC score
        return metrics.roc_auc_score(lb.transform(true_target), lb.transform(pred_target))


class SupportVectorClassifier(ClassificationModel):
    NUM_HYPERPARAMETERS = 1

    @classmethod
    def _unpack(cls, hyperparameters):
        return {
            'penalty': np.exp(16. * hyperparameters[0] - 8.),
        }

    @classmethod
    def _fit(cls, dataset, **hp):
        classifier = svm.SVC(kernel='rbf', C=hp['penalty'])

        # Train the model using the training set
        classifier.fit(dataset.train_data, dataset.train_target)

        # Return score on test set
        return cls._score(classifier.predict(dataset.test_data), dataset.test_target)


class LogitL1(ClassificationModel):
    NUM_HYPERPARAMETERS = 1

    @classmethod
    def _unpack(cls, hyperparameters):
        return {
            'penalty': np.exp(20. * hyperparameters[0] - 10.),
        }

    @classmethod
    def _fit(cls, dataset, **hp):
        classifier = linear_model.LogisticRegression(penalty='l1', C=hp['penalty'])

        # Train the model using the training set
        classifier.fit(dataset.train_data, dataset.train_target)

        # Return score on test set
        return cls._score(classifier.predict(dataset.test_data), dataset.test_target)


class LogitL2(ClassificationModel):
    NUM_HYPERPARAMETERS = 1

    @classmethod
    def _unpack(cls, hyperparameters):
        return {
            'penalty': np.exp(20. * hyperparameters[0] - 10.),
        }

    @classmethod
    def _fit(cls, dataset, **hp):
        classifier = linear_model.LogisticRegression(penalty='l2', C=hp['penalty'])

        # Train the model using the training set
        classifier.fit(dataset.train_data, dataset.train_target)

        # Return score on test set
        return cls._score(classifier.predict(dataset.test_data), dataset.test_target)


class RandomForest(RegressionModel):
    """
    Hyperparameters available are:
        number of trees

    """
    NUM_HYPERPARAMETERS = 1

    @classmethod
    def _unpack(cls, hyperparameters):
        num_trees, = hyperparameters
        return {
            "num_trees": max(int(10 ** (3 * num_trees)), 1),  # FIXME what range should this expand to? currently [0, 1] => [1, 1000]
        }

    @classmethod
    def _fit(cls, dataset, num_trees):
        regr = ensemble.RandomForestRegressor(n_estimators=num_trees)

        # Train the model using the training sets
        regr.fit(dataset.train_data, dataset.train_target)

        return cls._score(regr.predict(dataset.test_data), dataset.test_target)


class SupportVectorRegression(RegressionModel):
    """
    Hyperparameters available are:
        penalty
        width of epsilon-tube wiggle room

    """
    NUM_HYPERPARAMETERS = 1

    @classmethod
    def _unpack(cls, hyperparameters):
        penalty, = hyperparameters
        return {
            "penalty": max(10 ** (2 * penalty), 0.001),  # FIXME what range should this expand to? currently [0, 1]
        }

    @classmethod
    def _fit(cls, dataset, penalty):
        regr = svm.SVR(kernel='rbf', C=penalty)

        # Train the model using the training sets
        regr.fit(dataset.train_data, dataset.train_target)

        return cls._score(regr.predict(dataset.test_data), dataset.test_target)


class LassoRegression(RegressionModel):
    """
    Hyperparameters available are:
        alpha (or lambda) for lasso regression

    """
    NUM_HYPERPARAMETERS = 1

    @classmethod
    def _unpack(cls, hyperparameters):
        return {
            "alpha": hyperparameters[0] * 10 + 0.01  # FIXME what range should this expand to? currently [0, 10]
        }

    @classmethod
    def _fit(cls, dataset, alpha):
        regr = linear_model.Lasso(alpha=alpha)

        # Train the model using the training sets
        regr.fit(dataset.train_data, dataset.train_target)

        return cls._score(regr.predict(dataset.test_data), dataset.test_target)


class RidgeRegression(RegressionModel):
    """
    Hyperparameters available are:
        alpha (or lambda) for ridge/tikhonov regression

    """
    NUM_HYPERPARAMETERS = 1

    @classmethod
    def _unpack(cls, hyperparameters):
        return {
            "alpha": hyperparameters[0] * 10 + 0.01  # FIXME what range should this expand to? currently [0, 1]
        }

    @classmethod
    def _fit(cls, dataset, alpha):
        regr = linear_model.Ridge(alpha=alpha)

        # Train the model using the training sets
        regr.fit(dataset.train_data, dataset.train_target)

        return cls._score(regr.predict(dataset.test_data), dataset.test_target)


class BetterModel(Model):
    NUM_HYPERPARAMETERS = 1

    @classmethod
    def _unpack(cls, hyperparameters):
        return {
            "x": hyperparameters[0]
        }

    @classmethod
    def _fit(cls, dataset, x):
        return -(x - 0.5) ** 2 + 1.


class WorseModel(Model):
    NUM_HYPERPARAMETERS = 1

    @classmethod
    def _unpack(cls, hyperparameters):
        return {
            "x": hyperparameters[0]
        }

    @classmethod
    def _fit(cls, dataset, x):
        return -(x - 0.3) ** 2 + 0.5


def list_classification_models():
    return [SupportVectorClassifier, LogitL1, LogitL2]


def list_regression_models():
    return [RandomForest, SupportVectorRegression, LassoRegression, RidgeRegression]


def list_dummy_models():
    return [BetterModel, WorseModel]


if __name__ == "__main__":
    import random
    import datasets

    random.seed(1337)
    for model in list_regression_models():
        hyperparameters = tuple(random.random() for _ in range(model.NUM_HYPERPARAMETERS))
        model.fit(datasets.diabetes, hyperparameters)

    for model in list_classification_models():
        hyperparameters = tuple(random.random() for _ in range(model.NUM_HYPERPARAMETERS))
        model.fit(datasets.large_binary, hyperparameters)
