import numpy as np
from src.gaussian_process import *
from src.bayesian_optimizer import *
from src.kernels import *

# Data
from sklearn.datasets import make_classification

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split

# sklearn utils
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score

# plot
import matplotlib.pyplot as plt

# Set seed
np.random.seed(12)

# Get GP and BO
kernel = SquaredExponential(n_dim=1,
                            init_scale_range=(.1,.2),
                            init_amp=1.)
gp = GaussianProcess(n_epochs=100,
                     batch_size=10,
                     n_dim=1,
                     kernel=kernel,
                     noise=0.05,
                     train_noise=False,
                     optimizer=tf.train.GradientDescentOptimizer(0.1),
                     verbose=0)

# Generate data
data = make_classification(n_samples=1000,
                           n_features=1000,
                           n_informative=5,
                           n_redundant=2,
                           n_repeated=0,
                           n_classes=2,
                           n_clusters_per_class=2,
                           weights=None,
                           flip_y=0.1,
                           class_sep=0.5,
                           hypercube=True,
                           shift=0.0,
                           scale=1.0,
                           shuffle=True,
                           random_state=None)
# Normalize data
X = normalize(data[0])
y = data[1]

# Perform train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=42)

def l1_logit(show_plot):
    logit_results = []
    Cs = np.exp(np.linspace(-10, 10, 20))
    for C in Cs:
        y_pred = LogisticRegression(penalty='l1', C=C).fit(X_train, y_train).predict_proba(X_test)[:,1]
        mini_result =  roc_auc_score(y_test, y_pred)
        print C, "=>", mini_result
        logit_results.append(mini_result)
    if show_plot:
        plt.plot(np.log(Cs), logit_results)
        plt.xlabel("alphas")
        plt.ylabel("MSE")
        plt.show()
    return Cs, logit_results

def l2_logit(show_plot):
    logit_results = []
    Cs = np.exp(np.linspace(-10, 10, 20))
    for C in Cs:
        y_pred = LogisticRegression(C=C).fit(X_train, y_train).predict_proba(X_test)[:,1]
        mini_result =  roc_auc_score(y_test, y_pred)
        print C, "=>", mini_result
        logit_results.append(mini_result)
    if show_plot:
        plt.plot(np.log(Cs), logit_results)
        plt.xlabel("alphas")
        plt.ylabel("MSE")
        plt.show()
    return Cs, logit_results

def svc(show_plot):
    svc_results = []
    Cs = np.exp(np.linspace(-10, 10, 20))
    for C in Cs:
        y_pred = SVC(C=C, probability=True).fit(X_train, y_train).predict_proba(X_test)[:,1]
        mini_result =  roc_auc_score(y_test, y_pred)
        print C, "=>", mini_result
        svc_results.append(mini_result)
    if show_plot:
        plt.plot(np.log(Cs), svc_results)
        plt.xlabel("alphas")
        plt.ylabel("MSE")
        plt.show()
    return Cs, svc_results

l1_logit(True)
l2_logit(True)
svc(True)



# X, y = logit(show_plot=False)
# gp.fit(np.float32(X), np.float32(y))
# X_new = np.float32(np.linspace(0, 1, 100).reshape(-1,1))
# y_pred, var = gp.np_predict(X_new)

# ci = np.sqrt(var)*2
# plt.plot(X_new, y_pred)
# plt.plot(X_new, y_pred+ci, 'g--')
# plt.plot(X_new, y_pred-ci, 'g--')
# plt.scatter(X, y)
# plt.show()

