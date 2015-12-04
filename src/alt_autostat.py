from bayesian_optimizer import *
from gaussian_process import *
from kernels import *
import time
from sklearn.datasets import make_blobs
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# Set up the modules for bayesian optimizer
kernel = SquaredExponential(n_dim=1,
                            init_scale_range=(.01,.1),
                            init_amp=1.)
gp = GaussianProcess(n_epochs=100,
                     batch_size=10,
                     n_dim=1,
                     kernel=kernel,
                     noise=0.05,
                     train_noise=False,
                     optimizer=tf.train.GradientDescentOptimizer(0.001),
                     verbose=0)
bo = BayesianOptimizer(gp, region=np.array([[-1., 1.]]),
                       iters=100,
                       tries=10,
                       optimizer=tf.train.GradientDescentOptimizer(0.1),
                       verbose=0)


def train(key, x):
    model = models[key]
    if key in ['svc', 'l1', 'l2']:
        log_hyper = x * 7.
        hyper = np.exp(log_hyper)
        model.set_params(C=hyper)
    elif key in ['knn']:
        hyper = 7 + (x + 1)/2*23
        model.set_params(radius=hyper)
    t = time.time()
    y_pred = model.fit(X_train, y_train).predict(X_test)
    perf = (y_pred == y_test).mean()
    t = time.time() - t
    print "...Result: perf={0:.5f}, time={1:.5f}".format(perf, t)
    return perf, t

def run(key, x):
    y, t = train(key, x)
    datas[key][0].append(x)
    datas[key][1].append(y)
    datas[key][2].append(t)
    return t

# Make dataset
np.random.seed(1)
data1 = make_blobs(n_samples=10000,
                   n_features=2, centers=2,
                   cluster_std=3.0,
                   center_box=(-10.0, 10.0),
                   shuffle=True,
                   random_state=1)    
data2 = make_blobs(n_samples=10000,
                   n_features=2, centers=2,
                   cluster_std=3.0,
                   center_box=(-10.0, 10.0),
                   shuffle=True,
                   random_state=2)
X = np.vstack((data1[0], data2[0]))
y = np.concatenate((data1[1], data2[1]))
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=42)

# Create dataset holder for SVC, logit1, and logit2.
# Data is tuple that contains x, y, t
datas = {'knn' : ([],[],[]),
         'l1' : ([],[],[]),
         'l2' : ([],[],[])}
models = {'knn' : RadiusNeighborsClassifier(outlier_label=0),
          'l1' : LogisticRegression(penalty='l1'),
          'l2' : LogisticRegression(penalty='l2')}
# models = {'knn' : RadiusNeighborsClassifier(outlier_label=0)}
# Train the first time
for key in models:
    x = np.random.uniform(-1, 1)
    run(key, x)
    
time_left = 60
discount = 0.9
counter = 0
while time_left > 0:
    # Figure out the best point of each model:
    r_s = -np.inf
    key_s = None
    x_s = None
    for key in models:
        x = np.array(datas[key][0]).reshape(-1,1)
        y = np.array(datas[key][1]).reshape(-1,1)
        bo.fit(x, y)
        x_n, _, _ = bo.select()
        if counter % 5 == -1:
            # Plot stuff
            x_plot = np.linspace(-1, 1, 100).reshape(-1,1)
            y_plot, var_plot = gp.np_predict(x_plot)
            ci = var_plot ** .5
            plt.plot(x_plot, y_plot)
            plt.plot(x_plot, y_plot+ci, 'g--')
            plt.plot(x_plot, y_plot-ci, 'g--')
            plt.scatter(x, y)
            plt.show()
        y, y_var = gp.np_predict(x_n)
        t, t_var = np.mean(datas[key][2]), np.std(datas[key][2]) + 1e-3
        # Thompson sampling
        y_n = np.random.normal(y, np.sqrt(y_var))
        t_n = np.random.normal(t, np.sqrt(t_var))
        r_n = y_n if (time_left - t_n) > 0 else -1
        print "{0:4s}: [{1:.3f}, {2:.3f}, {3:.3f}]. Time: {4:.3f}, R: {5:.3f}".format(key,
                                                                                      x_n[0][0],
                                                                                      y[0][0],
                                                                                      y_n,
                                                                                      t_n,
                                                                                      r_n)
        if r_n > r_s:
            r_s = r_n
            key_s = key
            x_s = x_n
    # Run the model
    print "Counter:",counter,
    print "Training: {0:s}(x={1:.3f}). Time left: {2:.3f}".format(key_s, x_s[0][0], time_left)
    counter += 1
    t = run(key_s, x_s[0][0])
    time_left -= t
