"""
kernel.py
---------

Author: Rui Shu

This class implements the basic infrastructure and routines for Gaussian process
kernels using numpy and TensorFlow.
"""
__docformat__ = "resructedtext en"

import numpy as np
import tensorflow as tf

class Kernel(object):
    def __init__(self, n_dim, init_scale_range=(1e-1,2e-1), init_amp=1.):
        """ Initialize with parameters necessary for constructing the covariance
        matrix as a tf tensor.

        Parameters
        ----------
        n_dim : int 
            Number of dimensions in the parameter space for the Gaussian
            process. This defines the number of length scales to use.
        n_samples : int
            Number of samples in the covariance matrix to be computed; this
            defines the dimensions of the covariance matrix.
        """

        self.n_dim = n_dim
        # Randomly generate initial length scales for covariance matrix
        self.length_scales = tf.Variable(
            tf.random_uniform(
                (self.n_dim,),
                minval=init_scale_range[0], maxval=init_scale_range[1]
            )
        )
        # Create initial amplitude
        self.amp = tf.Variable(1.)

    def scaled_squared_distance(self, X, Y):
        """ Computes the squared distance.

        Parameters
        ----------
        X : np or tf nd.array. shape = (x_samples, n_dim)
            One of the design matrices
        Y : np or tf nd.array. shape = (y_samples, n_dim)
            One of the design matrices
        
        Returns
        -------
        NA : tf nd.array. shape = (x_samples, y_samples)
            Scaled squared distance matrix M where M[i, j] is the sq distance
            between X[i] and Y[j]
        """
        # Scale X and Y accordingly
        Xs, Ys = (tf.div(X, self.length_scales), tf.div(Y, self.length_scales))
        # Create matrix of ones
        Xo = tf.ones(tf.pack((tf.shape(X)[0], 1)))
        Yo = tf.ones(tf.pack((1, tf.shape(Y)[0])))
        # Precompute squared norms for rows of each matrix
        Xsqn = tf.reshape(tf.reduce_sum(tf.square(Xs), 1), tf.shape(Xo))
        Ysqn = tf.reshape(tf.reduce_sum(tf.square(Ys), 1), tf.shape(Yo))
        # Precompute "interaction" norm
        XYn = tf.matmul(Xs, tf.transpose(Ys))
        # Return the matrix of squared distances
        return tf.matmul(Xsqn, Yo) + tf.matmul(Xo, Ysqn) - 2*XYn
        
class SquaredExponential(Kernel):
    """ Squared Exponential Kernel Class

    Computes the squared exponential kernel matrix
    """
    def covariance(self, X, Y=None):
        """ Compute the covariance between the rows of X and Y (or X)

        Parameters
        ----------
        X : np or tf nd.array. shape = (x_samples, n_dim)
            One of the design matrices
        Y : np or tf nd.array. shape = (y_samples, n_dim)
            One of the design matrices
        
        Returns
        -------
        NA : tf nd.array. shape = (x_samples, y_samples)
            Covariance matrix K where K[i, j] computes k(X[i], Y[j]). k is the
            squared exponential kernel.
        """
        if Y is None:
            Y = X
        # Compute the scaled squared distance matrix
        sq_dist = self.scaled_squared_distance(X, Y)
        # Apply basic expontential function transformation
        return tf.square(self.amp) * tf.exp(-0.5*sq_dist)

    
if __name__ == "__main__":
    from scipy.spatial.distance import cdist
    kernel = SquaredExponential(n_dim=2,
                                init_scale_range=(1.,1.),
                                init_amp=1.)
    # Create X and Y
    X = np.float32(np.random.randint(1, 10, (4,2)))
    Y = np.float32(np.random.randint(1, 10, (3,2)))
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    print "X is"
    print X
    print "Y is"
    print Y

    # Tensorflow approach
    result = kernel.scaled_squared_distance(X, Y)
    covariance = kernel.covariance(X, Y)
    print "tensorflow scaled square distance"
    print sess.run(result)
    print "tensorflow covariance matrix"
    print sess.run(covariance)

    # SciPy approach
    ls = sess.run(kernel.length_scales)
    sq_dist = cdist(X, Y, 'euclidean')**2
    print "scipy scaled square distance"
    print sq_dist
    print "scipy covariance matrix"
    print np.exp(-.5*sq_dist)
