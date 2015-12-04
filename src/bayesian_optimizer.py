"""
bayesian_optimizer.py

Author: Rui Shu

This module implements bayesian optimization of a black-box
function. Internally, the bayesian optimizer utilizes a gaussian process to
construct a belief over the distribution of the latent function based on the
observed points, followed by using an ancquisition function that leverages
exploration/exploitation trade off to determine the next point to query.
"""
import tensorflow as tf
import numpy as np
__docformat__ = "restructuredtext en"


class BayesianOptimizer(object):
    """ Bayesian Optimization Class

    Performs bayesian optimization using a gaussian process that generates the
    posterior mean and variance of the latent function given observed data. The
    goal of bayesian optimization is to find the maximum of the latent function.
    """
    def __init__(self, gp, region, iters, tries,
                 optimizer=tf.train.GradientDescentOptimizer(0.1),
                 verbose=0):
        """ Initialize the parameters of the BayesianOptimizer object

        Parameters
        ----------
        gp : GaussianProcess
            A GaussianProcess object to be used for interpolating the data.
        region : np nd.array. shape = (n_dim, 2)
            Describes the cartesion box R that constraints the optimization
            task. The goal is to find \argmax_{x \in R} f(x).
        optimizer : tf.train.Gradient
            One of any gradient descent methods. Default to using basic sgd.
        iters : int 
            Number of iterations of gradient descent.
        verbose : int 
            Allows for different levels of verbosity for debugging purposes
        """
        # Save data
        self.gp = gp
        self.region = region
        # Get the session from the GP object
        self.sess = self.gp.sess
        # Define optimization algorithm
        self.opt = optimizer
        self.iters = iters
        self.n_tries = tries
        self.verbose = verbose

        # FIXME
        # x marks the query location of the next point. Currently not robustly
        # written. Must rewrite later.
        # We assume here that we've scaled the hyperparameter space to the unit hypercube.
        self.x = tf.Variable(tf.random_uniform([1, len(self.region)],
                                               minval=0, maxval=1))
        # Get the tensor variables for prediction at x
        self.y_pred, self.var = self.gp.tf_predict(self.x)
        # Define the acquisition function, which in our case is just the UCB
        self.acq = self.y_pred + 2*tf.sqrt(self.var)
        # Define the optimization task: maximizing acq w.r.t. self.x
        self.train = self.opt.minimize(-self.acq, var_list=[self.x])
        # placeholder for final
        self.x_placeholder = tf.placeholder(tf.float32, [None, None])
        self.assign_to_x = tf.assign(self.x, self.x_placeholder)
        self.initialize = tf.initialize_variables([self.x])
    def contains_point(self, x):
        """ Checks if x is contained within the optimization task region.

        Parameters
        ----------
        x : np nd.array. shape = (1, n_dims)
            Describes the point location.

        Returns
        -------
        N/A : bool
            Boolean describing whether x is contained in the space  .
        """
        return any([self.region[i, 0] <= x[0, i] <= self.region[i, 1]
                    for i in xrange(len(self.region))])

    def clip(self, x):
        """ Confine x to search space

        Parameters
        ----------
        x : np nd.array. shape = (1, n_dims)
            Describes the point location.

        Returns
        -------
        xT.T : np nd.array. shape = (1, n_dims)
            The clipped point.
        """
        xT = x.T
        # Clip x along dimensions that violate the lower bound of the region
        low_idx = xT[:, 0] < self.region[:, 0]
        xT[low_idx, 0] = self.region[low_idx, 0]
        # Clip x along dimensions that violate the upper bound of the region
        high_idx = xT[:, 0] > self.region[:, 1]
        xT[high_idx, 0] = self.region[high_idx, 1]
        return xT.T

    def fit(self, X, y):
        """ Fit the model that the Bayesian optimizer is using
        
        Parameters
        ----------
        X : np nd.array. shape = (n_samples, n_dim)
            The design matrix
        y : np nd.array. shape = (n_samples, 1)
            The response variable
        """
        X = np.float32(X)
        y = np.float32(y)
        self.gp.fit(X, y)
        self.gp_dict = {self.gp.X : self.gp.Xf,
                        self.gp.y : self.gp.yf,
                        self.gp.np_K_inv : self.gp.K_invf}
    def select(self):
        max_acq = -np.inf
        for _ in xrange(self.n_tries):
            x_cand, y_cand, acq_cand = self._select()
            # print "candidate_profile:", x_cand, y_cand, acq_cand
            if acq_cand > max_acq:
                x_next = x_cand
                y_next = y_cand
                acq_next = acq_cand
                max_acq = acq_cand
        # print x_next, y_next, acq_next
        return x_next, y_next, acq_next
            
    def _select(self):
        """ Select a point that maximizes the acquisition function. Finds this
        point via gradient descend of -acq.

        Returns
        -------
        self.sess.run(self.x) : np nd.array. shape = (1, n_dims)
            Best point.
        self.sess.run(self.y_pred) : np nd.array. shape = (1, 1)
            Posterior mean at self.x
        self.sess.run(self.acq) : np nd.array. shape = (1, 1)
            Acquisition function at self.x
        """
        # Only need to initialize self.x
        self.sess.run(self.initialize)
        if self.verbose > 0:
            print "BOInitPoint:", self.sess.run(self.x)[0]
            print "BOInitAcquisition:", self.sess.run(self.acq, feed_dict=self.gp_dict)[0,0]
        for _ in xrange(self.iters):
            self.sess.run(self.train, feed_dict=self.gp_dict)
            # Break from gradient descent if outside of optimization region
            if not self.contains_point(self.sess.run(self.x)):
                break
        # Generate final x
        self.sess.run(
            self.assign_to_x,
            feed_dict = {self.x_placeholder : self.clip(self.sess.run(self.x))}
        )
        return (self.sess.run(self.x),
                self.sess.run(self.y_pred, self.gp_dict),
                self.sess.run(self.acq, self.gp_dict))

def rui_1d():
    from kernels import SquaredExponential
    from gaussian_process import GaussianProcess
    import matplotlib.pyplot as plt

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
    plt.plot(x, y_pred)
    plt.plot(x, y_pred+ci, 'g--')
    plt.plot(x, y_pred-ci, 'g--')
    plt.scatter(X_old, y_old)
    plt.plot([x_next[0,0], x_next[0,0]], plt.ylim(), 'r--')
    plt.show()

def main_1d():
    from kernels import SquaredExponential
    from gaussian_process import GaussianProcess
    import matplotlib.pyplot as plt
    import time
    np.random.seed(10)
    tf.set_random_seed(10)
    # Settings
    n_samples = 3
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
                         noise=0.01,
                         train_noise=False,
                         optimizer=tf.train.GradientDescentOptimizer(0.001),
                         verbose=0)
    bo = BayesianOptimizer(gp, region=np.array([[0., 1.]]),
                           iters=100,
                           tries=2,
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
    for i in xrange(5):
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
    

if __name__ == "__main__":
    rui_1d()
