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
)
__docformat__ = "restructuredtext en"


class Model(object):
    """
    Abstract class for model thingy.
    """
    # Should be overridden in subclasses
    NUM_HYPERPARAMETERS = None

    @classmethod
    def fit(cls, dataset, hyperparameters):
        """

        Args:
            dataset: the Dataset object to train and test
            hyperparameters: a tuple of hyperparameters of size matching output from cls.get_num_hyperparameters

        Returns: (MSE, runtime in seconds)

        """
        hp = cls._unpack(hyperparameters)
        print "%s(%s)" % (cls.__name__, ', '.join(["%s=%s" % (key, value) for key, value in hp.iteritems()]))
        tic = timeit.default_timer()
        # performance = np.mean((cls._fit(dataset, **hp) - dataset.test_target) ** 2)
        performance = metrics.r2_score(dataset.test_target, cls._fit(dataset, **hp))
        toc = timeit.default_timer()
        return performance, toc - tic

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

        Returns: MSE

        """
        raise NotImplementedError


class RandomForest(Model):
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

        return regr.predict(dataset.test_data)


class SupportVectorRegression(Model):
    """
    Hyperparameters available are:
        penalty
        width of epsilon-tube wiggle room

    """
    NUM_HYPERPARAMETERS = 2

    @classmethod
    def _unpack(cls, hyperparameters):
        penalty, epsilon = hyperparameters
        return {
            "penalty": max(10 ** (2 * penalty), 0.001),  # FIXME what range should this expand to? currently [0, 1]
            "epsilon": epsilon,  # FIXME what range should this expand to? currently [0, 1]
        }

    @classmethod
    def _fit(cls, dataset, penalty, epsilon):
        print "SVR(C=%f, epsilon=%f)" % (penalty, epsilon)
        regr = svm.SVR(kernel='rbf', C=penalty, epsilon=epsilon)

        # Train the model using the training sets
        regr.fit(dataset.train_data, dataset.train_target)

        return regr.predict(dataset.test_data)


class LassoRegression(Model):
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

        return regr.predict(dataset.test_data)


class RidgeRegression(Model):
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

        return regr.predict(dataset.test_data)


def list_models():
    return [cls for cls in globals().values()
            if inspect.isclass(cls) and issubclass(cls, Model) and cls is not Model]

if __name__ == "__main__":
    import random
    from datasets import diabetes

    random.seed(1337)
    for model in list_models():
        hyperparameters = tuple(random.random() for _ in range(model.NUM_HYPERPARAMETERS))
        print "Calling %s.fit(diabetes, (%s))" % (model.__name__, ', '.join(map(str, hyperparameters)))
        print model.fit(diabetes, hyperparameters)
        print
