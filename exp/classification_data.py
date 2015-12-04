import numpy as np
from src.gaussian_process import *
from src.bayesian_optimizer import *
from src.kernels import *

# Data
from sklearn.datasets import make_classification, load_digits, make_moons, make_blobs
from sklearn.preprocessing import PolynomialFeatures
# Models
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split

# sklearn utils
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score

# plot
import matplotlib.pyplot as plt

np.random.seed(10)

def color_plot(X, y):
    color = np.zeros((X.shape[0], 3))
    color[y == 1, 1] = 1
    plt.scatter(X[:,0], X[:,1], color=color)
    plt.show()

data1 = make_blobs(n_samples=5000,
                   n_features=2, centers=2,
                   cluster_std=3.0,
                   center_box=(-10.0, 10.0),
                   shuffle=True,
                   random_state=1)    
data2 = make_blobs(n_samples=5000,
                   n_features=2, centers=2,
                   cluster_std=3.0,
                   center_box=(-10.0, 10.0),
                   shuffle=True,
                   random_state=2)
# data = make_moons(n_samples=10000, shuffle=True, noise=0.1, random_state=None)
X = np.vstack((data1[0], data2[0]))
y = np.concatenate((data1[1], data2[1]))

# Add a crap ton of random features
# X_random = np.random.normal(0, 3, (X.shape[0], 2))
# X = np.hstack((X, X_random))

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=42)
xs = np.linspace(7, 20, 20)
ys = []
for x in xs:
    print 'evaluating:', x
    y_pred = RadiusNeighborsClassifier(radius=x, outlier_label=0).fit(X_train, y_train).predict(X_test)
    perf = (y_test == y_pred).mean()
    print perf
    ys.append(perf)
# color_plot(X_test[:,:2], y_pred)
plt.plot(xs, ys)
plt.show()
