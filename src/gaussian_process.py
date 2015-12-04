"""
gaussian_process.py

Author: Rui Shu

This module implements Gaussian process regression, leveraging TensorFlow's
automatic differentiation for length scale learning and gridless acquisition
function optimization.
"""
import tensorflow as tf
import numpy as np
__docformat__ = "restructuredtext en"


@tf.RegisterGradient("MatrixDeterminant")
def _MatrixDeterminant(op, grad):
    """Gradient for MatrixDeterminant."""
    return grad * op.outputs[0] * tf.transpose(tf.matrix_inverse(op.inputs[0]))

class GaussianProcess(object):
    """ Gaussian Process Class
    
    Guassian process regression using tensorflow. 
    """
    def __init__(self, n_dim, kernel, n_epochs, batch_size, noise=0.0,
                 train_noise=False,
                 optimizer=tf.train.GradientDescentOptimizer(0.01),
                 verbose=0):
        """ Initialize parameters for Gaussian process object

        Parameters
        ----------
        n_epochs : int
            The number of epochs in stochastic gradient descend for fitting the
            GP kernel's length scale and amplitude.
        batch_size : int
            The number of data points to be fed during each step of stochastic
            gradient descent.
        n_dim : int
            The number of dimensions in the parameter space.
        kernel : Kernel object
            The covariance function to be used for GP regression
        noise : float32
            The noise hyperparameter for GP regression
        train_noise : bool
            Determines whether sgd will also optimize the loss function
            w.r.t. the noise hyperparameter in addition to the length scales.
        optimizer : tf.train.Gradient
            One of any gradient descent methods. Default to using basic sgd.
        verbose : int 
            Allows for different levels of verbosity for debugging purposes
        """
        # Save settings for kernel
        self.n_dim = n_dim
        self.kernel = kernel
        self.noise = tf.Variable(noise)
        # Save settings for gradient descend algorithm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.train_noise = train_noise
        self.optimizer = optimizer
        self.verbose = verbose
        # Create placeholder for observed data to be used in GP training of
        # length scales and/or noise parameter
        self.X = tf.placeholder(tf.float32, [None, n_dim])
        self.y = tf.placeholder(tf.float32, [None, 1])
        # Compute the covariance matrix and its inverse
        self.K = (
            self.kernel.covariance(self.X) + 
            tf.square(self.noise) * tf.diag(
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
        # Define the training operation for grad descend
        var_list = [self.kernel.length_scales]
        if self.train_noise:
            var_list.append(self.noise)
        self.grad = tf.gradients(self.cost, xs=var_list)
        self.train = self.optimizer.minimize(self.cost, var_list=var_list)
        # Initialize the TensorFlow session
        self.sess = tf.Session(
            config=tf.ConfigProto(
                inter_op_parallelism_threads=1, intra_op_parallelism_threads=1
            )       
        )

        # Variables for numpy prediction:
        self.np_X_ = tf.placeholder(tf.float32, [None, None])
        self.np_K_inv = tf.placeholder(tf.float32, [None, None])
        np_K_ = self.kernel.covariance(self.np_X_, self.X) 
        np_K_K_inv = tf.matmul(np_K_, self.np_K_inv)
        self.np_y_pred = tf.matmul(np_K_K_inv, self.y)        
        self.np_var = var = tf.abs(tf.reshape(
            tf.square(self.kernel.amp)
            - tf.reduce_sum(tf.mul(np_K_K_inv, np_K_), 1),
            [-1,1]
        ))
        self.initialize = tf.initialize_all_variables()

    def fit(self, X, y):
        """ Fit the gaussian process based on observed data (X, y)

        Parameters
        ----------
        X : np nd.array. shape = (n_samples, n_dim)
            The design matrix
        y : np nd.array. shape = (n_samples, 1)
            The response variable
        """
        X = np.float32(X)
        y = np.float32(y)
        # Train length scales via gradient descent
        self.sess.run(self.initialize)
        # Perform in batches
        n_samples = X.shape[0]
        n_batches = n_samples/self.batch_size
        for i in xrange(self.n_epochs):
            # Shuffle the data
            shuffle = np.random.permutation(n_samples)
            X = X[shuffle]
            y = y[shuffle]
            for j in xrange(n_batches):
                # Get minibatch
                mini_X = X[j*self.batch_size : (j+1)*self.batch_size]
                mini_y = y[j*self.batch_size : (j+1)*self.batch_size]
                # Update
                if self.verbose > 0:
                    print "Noise: {0:.3f}".format(self.sess.run(self.noise))
                    print ("Length Scale: {0:.3f}"
                           .format(self.sess.run(self.kernel.length_scales)[0]))
                    print self.sess.run(self.grad, feed_dict={self.X: mini_X,
                                                              self.y: mini_y})
                self.sess.run(self.train, feed_dict={self.X: mini_X,
                                                     self.y: mini_y})
        # Current hack: save the values
        self.K_invf = self.sess.run(self.K_inv,
                                    feed_dict={self.X: X, self.y: y})
        self.Xf = X
        self.yf = y

    def tf_predict(self, tf_X_):
        # Variables for tensor prediction:
        tf_K_ = self.kernel.covariance(tf_X_, self.X) 
        tf_K_K_inv = tf.matmul(tf_K_, self.np_K_inv)
        tf_y_pred = tf.matmul(tf_K_K_inv, self.y)        
        tf_var = var = tf.abs(tf.reshape(
            tf.square(self.kernel.amp)
            - tf.reduce_sum(tf.mul(tf_K_K_inv, tf_K_), 1),
            [-1,1]
        ))
        return tf_y_pred, tf_var
    
    def np_predict(self, X):
        """ Predict latent function evaluation and latent function variance at X

        Parameters
        ----------
        X : np or tf nd.array. shape = (n_samples, n_dim)
            The design matrix

        Returns
        -------
        y_pred : tf nd.array. shape = (n_samples, 1)
            The latent function mean evaluation
        var : tf nd.array. shape = (n_samples, 1)
            The latent function variance
        """
        return (self.sess.run(self.np_y_pred,
                         feed_dict = {self.np_X_ : X,
                                      self.X : self.Xf,
                                      self.y : self.yf,
                                      self.np_K_inv : self.K_invf}),
                self.sess.run(self.np_var,
                         feed_dict = {self.np_X_ : X,
                                      self.X : self.Xf,
                                      self.y : self.yf,
                                      self.np_K_inv : self.K_invf}))
    
def main_1d():
    """ Use Gaussian process regression to perform a 1D function regression task
    """
    from kernels import SquaredExponential
    import time
    import matplotlib.pyplot as plt
    # Settings
    n_samples = 10              # number of samples to train GP on 
    n_predict = 400             # number of samples for prediction
    n_dim = 1                   # 1D regression task
    lim = [0, 3]
    # Set seed
    tf.set_random_seed(1)
    np.random.seed(1)
    # Generate n_samples
    X = np.float32(np.random.uniform(lim[0], lim[1], [n_samples, n_dim]))
    y = np.float32((np.sin(X.sum(1)).reshape([n_samples, 1]) +
         np.random.normal(0,.1, [n_samples, 1])))
    # Create the kernel for GP
    kernel = SquaredExponential(n_dim=n_dim,
                                init_scale_range=(.1,1),
                                init_amp=1.)
    # Create the GP object
    gp = GaussianProcess(n_epochs=10,
                         batch_size=10,
                         n_dim=n_dim,
                         kernel=kernel,
                         noise=.1,
                         train_noise=False,
                         optimizer=tf.train.AdagradOptimizer(0.01),
                         verbose=0)
    t0 = time.time()
    # Train GP on training data to learn length scales
    for _ in xrange(1000):
        track_time = time.time()
        gp.fit(X, y)
        print "Fit_Speed: {0:.5f}".format(time.time() - track_time)
    print "FitDuration: {0:.5f}".format(time.time() - t0)
    print "LengthScale: {0:4.3f}".format(gp.sess.run(gp.kernel.length_scales)[0])
    print "Noise: {0:4.3f}".format(gp.sess.run(gp.noise))
    print "Cost: {0:4.3f}".format(gp.sess.run(gp.cost, feed_dict={gp.X: X, gp.y: y})[0,0])

    # Make prediction
    X_new = np.float32(np.linspace(lim[0], lim[1], n_predict).reshape(-1,1))
    X_new = np.sort(X_new, axis=0)
    for _ in xrange(1000):
        track_time = time.time()
        y_pred, var = gp.np_predict(X_new)
        print "Predict_Run_Speed: {0:.5f}".format(time.time() - track_time)
    ci = np.sqrt(var)*2
    plt.plot(X_new, y_pred)
    plt.plot(X_new, y_pred+ci, 'g--')
    plt.plot(X_new, y_pred-ci, 'g--')
    plt.scatter(X, y)
    plt.show()
    
def main_2d():        
    """ Use Gaussian process regression to perform a 2D function regression task
    """
    from kernels import SquaredExponential
    import time
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    # Set seed
    tf.set_random_seed(2)
    np.random.seed(2)
    # Settings
    n_samples = 20
    n_dim = 2
    # Get training samples
    X = np.float32(np.random.uniform(0, 10, [n_samples, n_dim]))
    y = np.float32((np.sin(np.sqrt(X).sum(1)).reshape([n_samples, 1])
                    + np.random.normal(0, .1, [n_samples, 1])))
    # Construct the kernel
    kernel = SquaredExponential(n_dim=n_dim,
                                init_scale_range=(12.,12.),
                                init_amp=1.)
    # Construct the gaussian process
    gp = GaussianProcess(n_epochs=100,
                         batch_size=3,
                         n_dim=n_dim,
                         kernel=kernel,
                         noise=0.2,
                         train_noise=False,
                         optimizer=tf.train.AdagradOptimizer(.01),
                         verbose=1)
    t0 = time.time()
    gp.fit(X, y)
    print "FitDuration: {0:.5f}".format(time.time() - t0)
    print "LengthScale: {0:4.3f}".format(gp.sess.run(gp.kernel.length_scales)[0])
    print "Noise: {0:4.3f}".format(gp.sess.run(gp.noise))
    print "Cost: {0:4.3f}".format(gp.sess.run(gp.cost, feed_dict={gp.X: X, gp.y: y})[0,0])
    # Create the mesh grid for prediction
    mx = np.arange(0, 10, 0.25)
    my = np.arange(0, 10, 0.25)
    mx, my = np.meshgrid(mx, my)
    X_new = np.float32(np.hstack((mx.ravel().reshape(-1, 1),
                                  my.ravel().reshape(-1, 1))))
    # Make prediction
    y_pred, var = gp.np_predict(X_new)
    y_pred = y_pred.reshape(mx.shape)
    ci = (np.sqrt(var)*2).reshape(mx.shape)
    # Plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    scat = ax.scatter(X[:,0], X[:,1], y)
    surf = ax.plot_surface(mx, my, y_pred, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, alpha=1)
    surf = ax.plot_surface(mx, my, y_pred-ci, rstride=1, cstride=1, cmap=cm.cool,
                           linewidth=0, antialiased=False, alpha=.5)
    surf = ax.plot_surface(mx, my, y_pred+ci, rstride=1, cstride=1, cmap=cm.cool,
                           linewidth=0, antialiased=False, alpha=.5)
    plt.show()

def rui_1d():
    """ Use Gaussian process regression to perform a 1D function regression task
    """
    from kernels import SquaredExponential
    import time
    import matplotlib.pyplot as plt
    # Settings
    n_predict = 400             # number of samples for prediction
    n_dim = 1                   # 1D regression task
    lim = [-1, 1]
    # Set seed
    tf.set_random_seed(1)
    np.random.seed(1)
    # Generate n_samples
    X = np.array([1.0, 
                  0.000335462627903, 
                  0.0314978449076, 
                  2980.95798704]).reshape(-1,1)
    X = np.log(X)/8
    y = np.array([0.864695262443,
                  0.5,
                  0.860244469176,
                  0.862691649896]).reshape(-1,1)
    # X = np.array().reshape(-1,1)
    # y = np.array([0.6, 0.7]).reshape(-1,1)

    # Create the kernel for GP
    kernel = SquaredExponential(n_dim=n_dim,
                                init_scale_range=(.01,.1),
                                init_amp=1.)
    # Create the GP object
    gp = GaussianProcess(n_epochs=10,
                         batch_size=10,
                         n_dim=n_dim,
                         kernel=kernel,
                         noise=.00,
                         train_noise=False,
                         optimizer=tf.train.GradientDescentOptimizer(0.001),
                         verbose=0)
    t0 = time.time()
    # Train GP on training data to learn length scales
    gp.fit(X, y)
    print "FitDuration: {0:.5f}".format(time.time() - t0)
    print "LengthScale: {0:4.3f}".format(gp.sess.run(gp.kernel.length_scales)[0])
    print "Noise: {0:4.3f}".format(gp.sess.run(gp.noise))
    print "Cost: {0:4.3f}".format(gp.sess.run(gp.cost, feed_dict={gp.X: X, gp.y: y})[0,0])
    # Make prediction
    X_new = np.float32(np.linspace(lim[0], lim[1], n_predict).reshape(-1,1))
    X_new = np.sort(X_new, axis=0)
    y_pred, var = gp.np_predict(X_new)
    ci = np.sqrt(var)*2
    plt.plot(X_new, y_pred)
    plt.plot(X_new, y_pred+ci, 'g--')
    plt.plot(X_new, y_pred-ci, 'g--')
    plt.scatter(X, y)
    plt.title('wtf')
    plt.show()
    
if __name__ == "__main__":
    rui_1d()
