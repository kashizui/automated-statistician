from scipy.stats import truncnorm
import numpy as np
from src.gaussian_process import *
from src.bayesian_optimizer import *
from src.kernels import *

# Data
from sklearn.datasets import make_classification, load_digits, make_moons, make_blobs, make_friedman1
from sklearn.preprocessing import PolynomialFeatures
# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.cross_validation import train_test_split

# sklearn utils
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score, mean_squared_error

# plot
import matplotlib.pyplot as plt

np.random.seed(10)

lower = 0
upper = 100
mu = 50
sigma = 50

def evaluate(X):
    return np.sin(X).reshape(-1) + np.random.normal(0, .3, len(X)).reshape(-1)

X = truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(1000).reshape(-1,1)
y = evaluate(X)

x_pred = np.linspace(0, 2000, 1000).reshape(-1, 1)
y_true = evaluate(x_pred)
# test
Cs = [1e-4, 1e-3, 1e-2, 1e-1, 1,2,3,4,5,6,7,8,9,10]
ys = []
for C in Cs:
    print "evaluating", C
    y_pred = SVR(C=C).fit(X, y).predict(x_pred)
    ys.append(mean_squared_error(y_true, y_pred))
plt.plot(Cs, ys)
plt.show()
quit()
plt.plot(x_pred, y_pred, 'g', label='c=1e-2')
y_pred = SVR(C=1e-1).fit(X, y).predict(x_pred)
plt.plot(x_pred, y_pred, 'r', label='c=1e-1')
y_pred = SVR(C=1).fit(X, y).predict(x_pred)
plt.plot(x_pred, y_pred, 'b', label='c=1')
y_pred = SVR(C=1e1).fit(X, y).predict(x_pred)
plt.plot(x_pred, y_pred, 'k', label='c=1e1')
y_pred = SVR(C=1e5).fit(X, y).predict(x_pred)
plt.plot(x_pred, y_pred, 'y', label='c=1e2')

plt.scatter(X, y)
plt.legend()
plt.show()
quit()

xs = np.linspace(-8,15,20)
ys = []
for x in xs:
    print "evaluating", x
    y_pred = SVR(C=np.exp(x)).fit(X_train, y_train).predict(X_test)
    ys.append(mean_squared_error(y_test, y_pred))
plt.plot(xs, ys)
plt.show()
