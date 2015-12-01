"""
autostat.py

Author: Stephen Koo

The main automatic statistician program.
Should, given a dataset, choose the best model.

"""
import numpy as np
import tensorflow as tf

import models
from bayesian_optimizer import BayesianOptimizer
from datasets import Dataset, diabetes
from gaussian_process import GaussianProcess
from kernels import SquaredExponential

__docformat__ = "restructuredtext en"


class ModelHistory(object):
    """
    Runs the given model and maintains the performance and time data
    """
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.hyperparameters = np.empty((0, model.NUM_HYPERPARAMETERS), dtype=np.float32)
        self.performance = np.empty((0, 1), dtype=np.float32)
        self.runtime = np.empty((0, 1), dtype=np.float32)

        # Settings
        batch_size = 4
        # create bayesian optimizer and gassian process
        self.kernel = SquaredExponential(n_dim=model.NUM_HYPERPARAMETERS,
                                         init_scale_range=(.5, 1),
                                         init_amp=1.)

        self.gp = GaussianProcess(n_epochs=100,
                                  batch_size=batch_size,
                                  n_dim=model.NUM_HYPERPARAMETERS,
                                  kernel=self.kernel,
                                  noise=0.1,
                                  train_noise=False,
                                  optimizer=tf.train.GradientDescentOptimizer(0.001),
                                  verbose=0)

        self.bo = BayesianOptimizer(self.gp,
                                    region=np.array([[0., 1.] for _ in xrange(model.NUM_HYPERPARAMETERS)]),
                                    iters=100,
                                    optimizer=tf.train.GradientDescentOptimizer(0.1),
                                    verbose=0)

    def sample(self):
        """sample an estimate of max performance and time based on the existing observations"""
        # build test points
        mhp = np.meshgrid(*[np.linspace(0, 1., 100) for _ in xrange(self.model.NUM_HYPERPARAMETERS)])
        hp = np.float32(np.hstack([row.ravel().reshape(-1, 1) for row in mhp]))

        # fit GP and predict
        self.gp.fit(self.hyperparameters, self.performance)
        perf_pred, var = [self.gp.sess.run(tv) for tv in self.gp.predict(hp)]

        max_i = np.argmax(perf_pred)
        return perf_pred[max_i], var[max_i]

    def run(self):
        """Actually run the model on the data with new hyperparameters and update history."""
        # Pick the set of hyperparameters that maximize the acquisition function
        if len(self.performance) == 0:
            # initialize in the center of the region
            hp_next = np.float32([[0.5 for _ in xrange(self.model.NUM_HYPERPARAMETERS)]])
        else:
            self.bo.fit(self.hyperparameters, self.performance)
            hp_next, _, _ = self.bo.select()

        # Run model
        perf, runtime = self.model.fit(self.dataset, tuple(hp_next[0]))

        # Append new results to history
        self.hyperparameters = np.vstack([self.hyperparameters, hp_next])
        self.performance = np.vstack([self.performance, np.float32(perf)])
        self.runtime = np.vstack([self.runtime, np.float32(runtime)])

    def plot(self):
        if self.model.NUM_HYPERPARAMETERS > 1:
            print np.concatenate([self.hyperparameters, self.performance, self.runtime], axis=1)
            return

        import matplotlib.pyplot as plt

        self.gp.fit(self.hyperparameters, self.performance)

        # predict performance mean function
        hp = np.float32(np.linspace(0, 1, 100).reshape(-1, 1))
        hp = np.sort(hp, axis=0)
        perf_pred, var = self.gp.predict(hp)
        perf_pred = self.gp.sess.run(perf_pred)
        var = self.gp.sess.run(var)
        ci = np.sqrt(var)*2

        # plot mean function, CI, and actual samples
        for i in xrange(self.model.NUM_HYPERPARAMETERS):
            plt.figure()
            plt.plot(hp, perf_pred)
            plt.plot(hp, perf_pred+ci, 'g--')
            plt.plot(hp, perf_pred-ci, 'g--')
            plt.scatter(self.hyperparameters, self.performance)
            plt.show()

    def virtual(self):
        return VirtualModelHistory(np.copy(self.hyperparameters),
                                   np.copy(self.performance),
                                   np.copy(self.runtime))


class VirtualModelHistory(object):
    """
    Maintains a virtual history of model runs, and provides a simple Gaussian model of the
    (max) performance and its associated runtime.
    unfinished
    """
    def __init__(self, hyperparameters, performance, runtime):
        self.hyperparameters = hyperparameters
        self.performance = performance
        self.runtime = runtime

        # fit gaussian model
        # mean and variance

    def update(self, hp, perf, runtime):
        """update virtual history with another sample"""
        # should add another data point
        pass

    def sample(self):
        pass


class AutomaticStatistician(object):
    # TODO: rollout evaluation
    # TODO: POMDP policy execution

    def __init__(self, dataset):
        self.models = [ModelHistory(model, dataset) for model in models.list_models()]

    def test(self):
        # Runs bayesian optimizer on each model 10 times
        for model in self.models:
            for _ in xrange(10):
                model.run()
            print model.sample()
            model.plot()

    def rollout(self, depth):
        pass


if __name__ == "__main__":
    autostat = AutomaticStatistician(diabetes)
    autostat.test()



