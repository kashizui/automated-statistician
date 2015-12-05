import numpy as np
import tensorflow as tf
from src.bayesian_optimizer import BayesianOptimizer
from src.kernels import SquaredExponential
from src.gaussian_process import GaussianProcess
import matplotlib.pyplot as plt
import time

np.random.seed(42)
tf.set_random_seed(42)

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
                       tries=20,
                       optimizer=tf.train.GradientDescentOptimizer(0.1),
                       verbose=1)

X = np.array([1.0,
              0.000335462627903,
              0.0314978449076,
              2980.95798704]).reshape(-1,1)
X = np.log(X)/8
y = np.array([0.864695262443,
              0.5,
              0.860244469176,
              0.862691649896]).reshape(-1,1)
X_old = X
y_old = y
bo.fit(X, y)
x_next, y, z = bo.select()

x = np.linspace(-1, 1).reshape(-1,1)
y_pred, var = gp.np_predict(x)
ci = 2*np.sqrt(var)
ci = np.sqrt(var)*2
plt.axis([-1, 1, -1.5, 2.5])
plt.plot(x, y_pred, label='Posterior Mean')
plt.plot(x, y_pred+ci, 'g--', label='Posterior Confidence Interval')
plt.plot(x, y_pred-ci, 'g--')
plt.scatter(X_old, y_old)
plt.plot([x_next[0,0], x_next[0,0]], plt.ylim(), 'r--', label='Next Query')
plt.xlabel('Hyperparameter')
plt.ylabel('Performance')
plt.legend(loc='upper left')
plt.savefig('bayesian_optimization.eps', format='eps', dpi=1000)

