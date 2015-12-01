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
from tictoc import tic, toc

__docformat__ = "restructuredtext en"


class ModelHistory(object):
    """
    Runs the given model and maintains the performance and time data.

    Essentially maintains belief state for the given model.
    """
    def __init__(self, model, dataset, hyperparameters, performance, runtime):
        self.model = model
        self.dataset = dataset
        self.hyperparameters = hyperparameters
        self.performance = performance
        self.runtime = runtime

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

        if len(self.performance) > 0:
            self.bo.fit(self.hyperparameters, self.performance)

    def belief(self):
        """
        Get the univariate belief distribution of the performance
        TODO: runtime
        """
        if len(self.performance) == 0:
            # initialize in the center of the region
            # TODO: with noise?
            hp = np.float32([[0.5 for _ in xrange(self.model.NUM_HYPERPARAMETERS)]])
            return hp, 1., np.sqrt(np.sqrt(0.1))

        # select slice at max of acquisition function
        hp, perf, acq = self.bo.select()
        std = (acq - perf) / 2.  # assuming UCB acquisition function
        # perf, var = (self.gp.sess.run(tv) for tv in self.gp.pred ict(hp))

        return hp, perf, std

    def sample(self, size=1):
        """
        sample an estimate of best performance and time based on the existing observations
        """
        hp, perf, std = self.belief()

        return hp, np.random.normal(perf, std, size)  # TODO: sample runtime

    def update(self, hp, perf, runtime):
        """
        Update history with new data point.
        """
        self.hyperparameters = np.vstack([self.hyperparameters, np.float32(hp)])
        self.performance = np.vstack([self.performance, np.float32(perf)])
        self.runtime = np.vstack([self.runtime, np.float32(runtime)])
        # refit BO
        self.bo.fit(self.hyperparameters, self.performance)

    def run(self):
        """
        Actually run the model on the data with new hyperparameters and update history.
        """
        # Pick the set of hyperparameters that maximize the acquisition function
        hp_next, _, _ = self.belief()

        # Run model
        perf, runtime = self.model.fit(self.dataset, tuple(hp_next[0]))

        # Append new results to history
        self.update(hp_next, perf, runtime)

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

    @classmethod
    def new(cls, model, dataset):
        """
        Return a fresh instance with no datapoints.
        """
        return cls(model,
                   dataset,
                   np.empty((0, model.NUM_HYPERPARAMETERS), dtype=np.float32),
                   np.empty((0, 1), dtype=np.float32),
                   np.empty((0, 1), dtype=np.float32))

    def copy(self):
        """
        Return a (semi-deep) copy of this instance.
        """
        return ModelHistory(self.model,
                            self.dataset,
                            np.copy(self.hyperparameters),
                            np.copy(self.performance),
                            np.copy(self.runtime))


class AutomaticStatistician(object):

    def __init__(self, dataset, discount=0.9):
        self.models = [ModelHistory.new(model, dataset) for model in models.list_models()]
        self.discount = discount

    def test(self):
        # Runs bayesian optimizer on each model 10 times
        self.models = [ModelHistory.new(model, None) for model in [models.BetterModel, models.WorseModel]]

        for _ in xrange(10):
            selected_i = self.select()
            print "selecting %s" % self.models[selected_i].model.__name__
            self.models[selected_i].run()

        # for model in self.models:
        #     for _ in xrange(3):
        #         model.run()

        # print self.rollout(depth=4)
        # print self.estimated_reward(self.models, 0)
        # print self.estimated_reward(self.models, 1)
        #
        # for model in self.models:
        #     for _ in xrange(7):
        #         model.run()
        #
        # # print self.rollout(depth=4)
        #
        # print self.estimated_reward(self.models, 0)
        # print self.estimated_reward(self.models, 1)

    def reward(self, models, perf):
        """
        TODO: use our full reward function definition, based on maximizing over history, etc.
        """
        return perf

    def expected_reward(self, models, selected_i):
        """
        Compute expected reward for the given action.
        """
        selected_model = models[selected_i]
        _, perfs = selected_model.sample(200)
        return np.mean([self.reward(models, perf) for perf in perfs])

    def select(self):
        """
        Select the next action to take.
        """
        # Yields U(UpdateBelief(b, a, o)) for N sample observations
        def action_observation_values(i):
            for iters in xrange(5):  # 5 sample observations
                tic("Sample observation %d for model %d" % (iters, i))
                virtual_models = [m.copy() for m in self.models]
                hp, perf = virtual_models[i].sample()
                virtual_models[i].update(hp, perf, 0)  # FIXME
                yield self.rollout(virtual_models, 3)
                toc()

        # Equation (6.35)
        def action_value(i):
            return self.expected_reward(self.models, i) +\
                   self.discount * np.mean(list(action_observation_values(i)))

        # Return argmax of (6.35)
        return max(xrange(len(self.models)), key=action_value)

    def rollout(self, models, depth):
        """
        Rollout evaluation to estimate the value function.
        """
        value = 0.
        for d in xrange(depth):
            # a ~ \pi_0(b)
            samples = [vm.sample() for vm in models]
            selected_i = max(xrange(len(samples)), key=lambda i: samples[i][1])
            selected_model = models[selected_i]

            # s ~ b
            hp, perf = samples[selected_i]

            # (s', o, r) ~ G(s, a)
            new_perf = np.random.normal(perf, np.sqrt(np.sqrt(0.1)))

            # b' <- UpdateBelief(b, a, o)
            selected_model.update(hp, new_perf, 0)

            value += (self.discount ** d) * self.reward(models, new_perf)

        return value


if __name__ == "__main__":
    autostat = AutomaticStatistician(diabetes)
    autostat.test()



