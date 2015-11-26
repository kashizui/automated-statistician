import tensorflow as tf
import numpy as np
from kernels import *
from gaussian_process import GaussianProcess

class BayesianOptimizer(object):
    def __init__(self, gp, region):
        self.gp = gp
        self.sess = self.gp.sess
        self.region = region
        self.x = tf.Variable(tf.random_uniform([1, len(self.region)], minval=0, maxval=10))
        self.opt = tf.train.GradientDescentOptimizer(.1)

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
        self.y_pred, self.cov = self.gp.predict(self.x)
        self.acq = self.y_pred + 2*tf.sqrt(self.cov)
        self.train = self.opt.minimize(-self.acq, var_list=[self.x])

    def select(self):
        self.sess.run(tf.initialize_variables([self.x]))
        print self.sess.run(self.x), self.sess.run(self.acq)
        for _ in xrange(100):
            self.sess.run(self.train)
            if not self.contains_point(self.sess.run(self.x)):
                break
        return self.sess.run(self.x), self.sess.run(self.y_pred), self.sess.run(self.acq)
            
def main_1d():
    import matplotlib.pyplot as plt
    import time 
    n_samples = 4
    batch_size = 4
    new_samples = 100
    n_dim = 1
    kernel = SquaredExponential(n_dim=n_dim,
                                init_scale_range=(.1,.2),
                                init_amp=1.)
    gp = GaussianProcess(n_epochs=100,
                         batch_size=batch_size,
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
    for i in xrange(100):
        print "Iteration {0:3d}".format(i) + "*"*80
        t0 = time.time()
        max_acq = -np.inf
        for _ in xrange(1):
            t1 = time.time()
            x_cand, y_cand, acq_cand = bo.select()
            print "BOSelectDuration: {0:.5f}".format(time.time() - t1)
            if acq_cand > max_acq:
                x_next= x_cand
                y_next= y_cand
                acq_next = acq_cand
        plt.plot([x_next[0,0], x_next[0,0]], plt.ylim(), 'r--')
        plt.scatter(x_next, y_next, c='r', linewidths=0, s=50)
        plt.scatter(x_next, acq_next, c='g', linewidths=0, s=50)
        y_obs = observe(x_next)
        X = np.vstack((X, x_next))
        y = np.vstack((y, y_obs))
        t2 = time.time()
        bo.fit(X, y)
        print "BOFitDuration: {0:.5f}".format(time.time() - t2)
        print "BOTotalDuration: {0:.5f}".format(time.time() - t0)
    X_new = np.float32(np.linspace(0, 10, new_samples).reshape(-1, 1))
    X_new = np.sort(X_new, axis=0)
    y_pred, cov = gp.predict(X_new)
    y_pred = gp.sess.run(y_pred)
    var = gp.sess.run(cov)
    ci = np.sqrt(var)*2
    plt.plot(X_new, y_pred)
    plt.plot(X_new, y_pred+ci, 'g--')
    plt.plot(X_new, y_pred-ci, 'g--')
    plt.scatter(X, y)
    plt.show()
    

if __name__ == "__main__":
    main_1d()
