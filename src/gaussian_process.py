"""
gaussian_process.py

Author: Rui Shu

This module implements Gaussian process regression, leveraging TensorFlow's
automatic differentiation for length scale learning and gridless acquisition
function optimization.
"""
__docformat__ = "restructuredtext en"

import tensorflow as tf
import numpy as np
from kernels import *

@tf.RegisterGradient("MatrixDeterminant")
def _MatrixDeterminant(op, grad):
    """Gradient for MatrixDeterminant."""
    return grad * op.outputs[0] * tf.transpose(tf.matrix_inverse(op.inputs[0]))


class GaussianProcess(object):
    """ Gaussian Process Class
    
    Multivariate Guassian process regression using tensorflow. 
    """
    def __init__(self, n_epochs, batch_size, n_dim, kernel, noise=0.0,
                 train_noise=False,
                 optimizer=tf.train.GradientDescentOptimizer(0.01),
                 verbose=0):
        """ Initialize parameters for Gaussian process object
        """
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_dim = n_dim
        self.kernel = kernel
        self.noise = tf.Variable(noise, trainable=train_noise)
        self.optimizer = optimizer
        self.verbose = verbose
        # Create placeholder objects for GP observed data
        self.X = tf.placeholder(tf.float32, [None, n_dim])
        self.y = tf.placeholder(tf.float32, [None, 1])
        # Compute the covariance matrix
        self.K = (
            self.kernel.covariance(self.X)
            + tf.square(self.noise) * tf.diag(
                tf.ones(
                    tf.pack([tf.shape(self.X)[0]])
                )
            )
        )
        self.K_inv = tf.matrix_inverse(self.K)
        # Determine the gp negative log marginal likelihood
        self.cost = (
            tf.matmul(tf.matmul(tf.transpose(self.y), self.K_inv),
                      self.y)
            + tf.log(tf.matrix_determinant(self.K))
        )
        

    def fit(self, X, y):
        """ Fit the gaussian process

        Parameters
        ----------
        X : tf or np nd.array. shape = (n_samples, n_dim)
        y : tf or np nd.array. shape = (n_samples, 1)
        """
        # Train length scales via gradient descent
        train = self.optimizer.minimize(self.cost)
        # Initialize the TensorFlow session.
        self.sess = tf.Session(
            config=tf.ConfigProto(
                inter_op_parallelism_threads=1, intra_op_parallelism_threads=1
            )       
        )
        self.sess.run(tf.initialize_all_variables())
        # Perform in batches
        n_samples = X.shape[0]
        n_batches = n_samples/self.batch_size
        for i in xrange(self.n_epochs):
            # shuffle data
            shuffle = np.random.permutation(n_samples)
            X = X[shuffle]
            y = y[shuffle]
            for j in xrange(n_batches):
                # get minibatch
                mini_X = X[j*self.batch_size : (j+1)*self.batch_size]
                mini_y = y[j*self.batch_size : (j+1)*self.batch_size]
                self.sess.run(train, feed_dict={self.X: mini_X, self.y: mini_y})
        # Current hack: save the values
        self.K_invf = self.sess.run(self.K_inv, feed_dict={self.X: mini_X, self.y: mini_y})
        self.Xf = X
        self.yf = y
        
    def predict(self, X):
        # Predict mean
        K_ = self.kernel.covariance(X, self.Xf)
        K_K_invf = tf.matmul(K_, self.K_invf)
        y_pred = tf.matmul(K_K_invf, self.yf)
        var = self.kernel.covariance(X) - tf.matmul(K_K_invf, tf.transpose(K_))
        return self.sess.run(y_pred), np.diag(self.sess.run(var)).reshape([X.shape[0],1])
        
def main():        
    from scipy.spatial.distance import cdist
    import matplotlib.pyplot as plt
    # Create X and Y
    n_samples = 10
    n_dim = 1
    X = np.float32(np.random.uniform(1, 10, [n_samples, n_dim]))
    y = np.float32((np.sin(X.sum(1)).reshape([n_samples, 1]) +
         np.random.normal(0,.1, [n_samples, 1])))
    kernel = SquaredExponential(n_dim=n_dim,
                                init_scale_range=(1.,1.),
                                init_amp=1.)
    gp = GaussianProcess(n_epochs=100,
                         batch_size=n_samples,
                         n_dim=n_dim,
                         kernel=kernel,
                         noise=0.1,
                         train_noise=False,
                         optimizer=tf.train.GradientDescentOptimizer(0.01),
                         verbose=0)
    gp.fit(X, y)
    print gp.sess.run(gp.noise)
    X_new = np.float32(np.random.uniform(1, 10, [n_samples*100, n_dim]))
    X_new = np.sort(X_new, axis=0)
    y_pred, var = gp.predict(X_new)
    ci = np.sqrt(var)*2
    plt.plot(X_new, y_pred)
    plt.plot(X_new, y_pred+ci, 'g--')
    plt.plot(X_new, y_pred-ci, 'g--')
    plt.scatter(X, y)
    plt.show()
    
if __name__ == "__main__":
    main()
