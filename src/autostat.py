"""
autostat.py

Author: Stephen Koo

The main automatic statistician program.
Should, given a dataset, choose the best model.

"""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors import InvalidArgumentError
import models
import datasets
from bayesian_optimizer import BayesianOptimizer
from gaussian_process import GaussianProcess
from kernels import SquaredExponential
from tictoc import tic, toc

__docformat__ = "restructuredtext en"


class BeliefSlice(object):
    """
    Represents the slice of the GP, presumably where the acquisition function is maximized.
    """

    def __init__(self, hp, perf_mean, perf_std, runtime_mean, runtime_std):
        self.hp = hp
        self.perf_mean = perf_mean
        self.perf_std = perf_std
        self.runtime_mean = runtime_mean
        self.runtime_std = runtime_std


class ModelHistory(object):
    """
    Runs the given model and maintains the performance and time data.

    Essentially maintains belief state for the given model.
    """

    def __init__(self, model, dataset,
                 hyperparameters, performance, runtime,
                 bo, gp,
                 perf_sample_noise, runtime_sample_noise):
        self.model = model
        self.dataset = dataset
        self.hyperparameters = hyperparameters
        self.performance = performance
        self.runtime = runtime
        self.bo = bo
        self.gp = gp
        self.perf_sample_noise = perf_sample_noise
        self.runtime_sample_noise = runtime_sample_noise
        self.belief = self.reslice_belief()

    def reslice_belief(self):
        """
        Get the univariate belief distribution of the performance (the slice)
        """
        if len(self.performance) == 0:
            # initialize in the center of the region
            # TODO: with noise?
            hp = np.float32([[0.5 for _ in xrange(self.model.NUM_HYPERPARAMETERS)]])
            return BeliefSlice(hp,
                               1.,
                               self.perf_sample_noise,
                               0.,
                               self.runtime_sample_noise)

        # Fit GP
        try:
            self.bo.fit(self.hyperparameters, self.performance)
        except InvalidArgumentError:
            print "The offending data:"
            print self.hyperparameters
            print self.performance
            raise

        # Select slice at max of acquisition function
        hp, perf_mean, acq = self.bo.select()
        perf_mean, perf_var = self.gp.np_predict(hp)

        # Compute mean and variance of runtimes
        runtime_mean = np.mean(self.runtime)
        runtime_std = np.std(self.runtime)

        # Save in model history state
        return BeliefSlice(hp,
                           perf_mean,
                           np.max([np.sqrt(perf_var), np.float32(0.001)]),
                           runtime_mean,
                           np.max([runtime_std, np.float32(0.001)]))

    def sample(self, size=1):
        """
        sample an estimate of best performance and time based on the existing observations
        """
        return (self.belief.hp,
                np.random.normal(self.belief.perf_mean, self.belief.perf_std, size),
                np.random.normal(self.belief.runtime_mean, self.belief.runtime_std, size))

    def update(self, hp, perf, runtime):
        """
        Update history with new data point.
        """
        self.hyperparameters = np.vstack([self.hyperparameters, np.float32(hp)])
        self.performance = np.vstack([self.performance, np.float32(perf)])
        self.runtime = np.vstack([self.runtime, np.float32(runtime)])
        self.belief = self.reslice_belief()

    def run(self):
        """
        Actually run the model on the data with new hyperparameters and update history.
        """
        # Run model
        perf, runtime = self.model.fit(self.dataset, tuple(self.belief.hp[0]))

        # Append new results to history
        self.update(self.belief.hp, perf, runtime)

        return self.belief.hp, perf, runtime

    def plot(self):
        if self.model.NUM_HYPERPARAMETERS > 1:
            print np.concatenate([self.hyperparameters, self.performance, self.runtime], axis=1)
            return

        import matplotlib.pyplot as plt

        self.gp.fit(self.hyperparameters, self.performance)

        # predict performance mean function
        hp = np.float32(np.linspace(0, 1, 1000).reshape(-1, 1))
        hp = np.sort(hp, axis=0)
        perf_pred, var = self.gp.np_predict(hp)
        ci = np.sqrt(var) * 2

        # plot mean function, CI, and actual samples
        for i in xrange(self.model.NUM_HYPERPARAMETERS):
            plt.figure()
            plt.plot(hp, perf_pred)
            plt.plot(hp, perf_pred + ci, 'g--')
            plt.plot(hp, perf_pred - ci, 'g--')
            plt.scatter(self.hyperparameters, self.performance)
            plt.title(self.model.__name__)
            plt.show()

        print "%s runtime belief: mean=%.4f, std=%.4f" % (self.model.__name__,
                                                          self.belief.runtime_mean,
                                                          self.belief.runtime_std)

    @classmethod
    def new(cls, model, dataset, bo, gp, perf_sample_noise, runtime_sample_noise):
        """
        Return a fresh instance with no datapoints.
        """
        return cls(model,
                   dataset,
                   np.empty((0, model.NUM_HYPERPARAMETERS), dtype=np.float32),
                   np.empty((0, 1), dtype=np.float32),
                   np.empty((0, 1), dtype=np.float32),
                   bo,
                   gp,
                   perf_sample_noise,
                   runtime_sample_noise)

    def copy(self):
        """
        Return a (semi-deep) copy of this instance.
        """
        return ModelHistory(self.model,
                            self.dataset,
                            np.copy(self.hyperparameters),
                            np.copy(self.performance),
                            np.copy(self.runtime),
                            self.bo,
                            self.gp,
                            self.perf_sample_noise,
                            self.runtime_sample_noise)


class AutomaticStatistician(object):
    def __init__(self, discount=0.9):
        # Settings
        self.discount = discount
        self.n_queries = 20
        self.depth = 3
        self.n_sim = 1
        self.n_reward_samples = 200
        self.perf_sample_std = 0.001
        self.runtime_sample_std = 0.001
        batch_size = 10

        # create bayesian optimizer and gassian process
        self.kernel = SquaredExponential(n_dim=1,  # assuming 1 hyperparameter
                                         init_scale_range=(.1, .5),
                                         init_amp=1.)

        self.gp = GaussianProcess(n_epochs=100,
                                  batch_size=batch_size,
                                  n_dim=1,  # assuming 1 hyperparameter
                                  kernel=self.kernel,
                                  noise=0.05,
                                  train_noise=False,
                                  optimizer=tf.train.GradientDescentOptimizer(0.001),
                                  verbose=0)

        self.bo = BayesianOptimizer(self.gp,
                                    region=np.array([[0., 1.]]),  # assuming 1 hyperparameter
                                    iters=100,
                                    tries=2,
                                    optimizer=tf.train.GradientDescentOptimizer(0.1),
                                    verbose=0)

    def run(self, dataset, time_limit):
        """
        Run the automatic statistician.
        """
        histories = [ModelHistory.new(model, dataset, self.bo, self.gp,
                                      self.perf_sample_std, self.runtime_sample_std)
                     for model in models.list_classification_models()]

        # do multi-armed bandit for N iterations
        time_left = time_limit
        while time_left > 0:
            print "%.3f seconds left" % time_left
            selected = self.select(histories, time_left)

            _, _, runtime = selected.run()
            time_left -= runtime

        # plot resulting beliefs
        for m in histories:
            m.plot()

    @staticmethod
    def reward(perf, runtime, time_left):
        """
        Return immediate reward
        """
        return perf + (time_left - runtime) / time_left

    def expected_reward(self, histories, selected_i, time_left):
        """
        Compute expected reward for the given action.
        """
        selected_action = histories[selected_i]
        _, perfs, runtimes = selected_action.sample(self.n_reward_samples)
        return np.mean([self.reward(perf, runtime, time_left) for perf, runtime in zip(perfs, runtimes)])

    def select(self, histories, time_left):
        """
        Select the next action to take.
        """
        # Yields U(UpdateBelief(b, a, o)) for N sample observations
        def action_observation_values(i):
            for iters in xrange(self.n_sim):
                virtual_histories = [m.copy() for m in histories]
                hp, perf, runtime = virtual_histories[i].sample()
                virtual_histories[i].update(hp, perf, runtime)
                yield self.rollout(virtual_histories, self.depth, time_left - runtime)

        # Equation (6.35)
        def action_value(i):
            value = self.expected_reward(histories, i, time_left) + \
                    self.discount * np.mean(list(action_observation_values(i)))
            print "Q(b, %s) = %.3f" % (histories[i].model.__name__, value)
            return value

        # Return argmax of (6.35)
        return histories[max(xrange(len(histories)), key=action_value)]

    def rollout(self, histories, depth, time_left):
        """
        Rollout evaluation to estimate the value function.
        """
        value = 0.
        for d in xrange(depth):
            # a ~ \pi_0(b)
            samples = [h.sample() for h in histories]
            selected_i = max(xrange(len(samples)), key=lambda i: samples[i][1])
            selected = histories[selected_i]

            # s ~ b
            hp, perf, runtime = samples[selected_i]

            # (s', o, r) ~ G(s, a)
            new_perf = np.random.normal(perf, self.perf_sample_std)
            new_runtime = np.random.normal(runtime, self.runtime_sample_std)

            # b' <- UpdateBelief(b, a, o)
            selected.update(hp, new_perf, new_runtime)

            # Accumulate rewards with discount
            value += (self.discount ** d) * self.reward(new_perf, new_runtime, time_left)
            time_left -= new_runtime

        return value


if __name__ == "__main__":
    np.random.seed(10)
    tf.set_random_seed(10)
    autostat = AutomaticStatistician()
    autostat.run(datasets.large_binary, time_limit=10.)
