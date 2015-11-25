import tensorflow as tf
import numpy as np
from kernels import *
from gaussian_process import GaussianProcess

class BayesianOptimizer(object):
    def __init__(self, gp, region):
        self.gp = gp
        self.sess = self.gp.sess
        self.region = region

    def contains_point(self, x):
        return any([self.region[i, 0] <= x[i] <= self.region[i, 1]
                    for i in xrange(len(self.region))])

    def clip(self, x):
        xT = x.T
        low_idx = xT[:, 0] < self.region[:, 0]
        xT[low_idx, 0] = self.region[low_idx, 0]
        high_idx = xT[:, 0] > self.region[:, 1]
        xT[high_idx, 0] = self.region[high_idx, 1]
        return xT.T

    def fit(self, X, y):
        self.gp.fit(X, y)
    
    def select(self):
        x = tf.Variable(np.random.uniform(0, 10, [1, len(self.region)]).astype(np.float32))
        y_pred, cov = self.gp.predict(x)
        acq = y_pred + 2*tf.sqrt(cov)
        self.sess.run(tf.initialize_variables([x]))
        print self.sess.run(x), self.sess.run(acq)
        opt = tf.train.GradientDescentOptimizer(.1)
        train = opt.minimize(-acq, var_list=[x])
        for _ in xrange(100):
            self.sess.run(train)
            if not self.contains_point(self.sess.run(x)):
                break
        self.sess.run(tf.assign(x, self.clip(self.sess.run(x))))
        return self.sess.run(x), self.sess.run(y_pred), self.sess.run(acq)
            
def main_1d():
    import matplotlib.pyplot as plt
    n_samples = 10
    new_samples = 100
    n_dim = 1
    kernel = SquaredExponential(n_dim=n_dim,
                                init_scale_range=(.1,.2),
                                init_amp=1.)
    gp = GaussianProcess(n_epochs=100,
                         batch_size=10,
                         n_dim=n_dim,
                         kernel=kernel,
                         noise=0.1,
                         train_noise=False,
                         optimizer=tf.train.GradientDescentOptimizer(0.001),
                         verbose=0)
    bo = BayesianOptimizer(gp, region=np.array([[0., 10.]]))
    X = np.float32(np.random.uniform(1, 10, [n_samples, n_dim]))
    def observe(X):
        y = np.float32((np.sin(X.sum(1)).reshape([X.shape[0], 1]) +
                        np.random.normal(0,.1, [X.shape[0], 1])))
        return y
    y = observe(X)
    plt.axis((0, 10, -3, 3))
    bo.fit(X, y)
    for _ in xrange(10):
        x_next, y_next, acq_next = bo.select()
        plt.plot([x_next[0,0], x_next[0,0]], plt.ylim(), 'r--')
        plt.scatter(x_next, y_next, c='r', linewidths=0, s=50)
        plt.scatter(x_next, acq_next, c='g', linewidths=0, s=50)
        y_obs = observe(x_next)
        X = np.vstack((X, x_next))
        y = np.vstack((y, y_obs))
        bo.fit(X, y)
    X_new = np.float32(np.linspace(0, 10, new_samples).reshape(-1, 1))
    X_new = np.sort(X_new, axis=0)
    y_pred, cov = gp.predict(X_new)
    y_pred = gp.sess.run(y_pred)
    var = np.diag(gp.sess.run(cov)).reshape(y_pred.shape)
    ci = np.sqrt(var)*2
    plt.plot(X_new, y_pred)
    plt.plot(X_new, y_pred+ci, 'g--')
    plt.plot(X_new, y_pred-ci, 'g--')
    plt.scatter(X, y)
    plt.show()
    

if __name__ == "__main__":
    main_1d()
