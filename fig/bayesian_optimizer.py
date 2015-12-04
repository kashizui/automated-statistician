import numpy as np
import tensorflow as tf
from src.bayesian_optimizer import BayesianOptimizer
from src.kernels import SquaredExponential
from src.gaussian_process import GaussianProcess
import matplotlib.pyplot as plt
import time
np.random.seed(10)
tf.set_random_seed(10)
# Settings
n_samples = 100
batch_size = 4
new_samples = 1000
n_dim = 1
# Set up the modules for bayesian optimizer
kernel = SquaredExponential(n_dim=n_dim,
                            init_scale_range=(.1,.5),
                            init_amp=1.)
gp = GaussianProcess(n_epochs=100,
                     batch_size=10,
                     n_dim=n_dim,
                     kernel=kernel,
                     noise=0.05,
                     train_noise=False,
                     optimizer=tf.train.GradientDescentOptimizer(0.001),
                     verbose=0)
bo = BayesianOptimizer(gp, region=np.array([[0., 1.]]),
                       iters=100,
                       tries=4,
                       optimizer=tf.train.GradientDescentOptimizer(0.1),
                       verbose=1)
# Define the latent function + noise
def observe(X):
    y = np.float32(
        1*(-(X - 0.5) ** 2 + 1) +
        np.random.normal(0,.1, [X.shape[0], 1]))
    # y = np.float32((np.sin(X.sum(1)).reshape([X.shape[0], 1]) +
    #                 np.random.normal(0,.1, [X.shape[0], 1])))
    return y
# Get data
X = np.float32(np.random.uniform(0, 1, [n_samples, n_dim]))
y = observe(X)
plt.axis((-0.1, 1.1, 0, 1.5))
# Fit the gp
bo.fit(X, y)
for i in xrange(10):
    # print "Iteration {0:3d}".format(i) + "*"*80
    t0 = time.time()
    max_acq = -np.inf
    # Inner loop to allow for gd with random initializations multiple times
    x_next, y_next, acq_next = bo.select()
    # Plot the selected point
    plt.plot([x_next[0,0], x_next[0,0]], plt.ylim(), 'r--')
    plt.scatter(x_next, y_next, c='r', linewidths=0, s=50)
    plt.scatter(x_next, acq_next, c='g', linewidths=0, s=50)
    # Observe and add point to observed data
    y_obs = observe(x_next)
    X = np.vstack((X, x_next))
    y = np.vstack((y, y_obs))
    t2 = time.time()
    # Fit again
    bo.fit(X, y)
    print "BOFitDuration: {0:.5f}".format(time.time() - t2)
    print "BOTotalDuration: {0:.5f}".format(time.time() - t0)

# Get the final posterior mean and variance for the entire domain space
X_new = np.float32(np.linspace(0, 1, new_samples).reshape(-1, 1))
X_new = np.sort(X_new, axis=0)
y_pred, var = gp.np_predict(X_new)
# Compute the confidence interval
ci = np.sqrt(var)*2
plt.plot(X_new, y_pred)
plt.plot(X_new, y_pred+ci, 'g--')
plt.plot(X_new, y_pred-ci, 'g--')
plt.scatter(X, y)
plt.show()
