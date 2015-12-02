import numpy as np
from src.gaussian_process import *
from src.bayesian_optimizer import *
from src.kernels import *

# Data
from sklearn.datasets import make_classification, load_digits, make_moons
from sklearn.preprocessing import PolynomialFeatures
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

np.random.seed(10)

def color_plot(X, y):
    color = np.zeros((X.shape[0], 3))
    color[y == 1, 1] = 1
    plt.scatter(X[:,0], X[:,1], color=color)
    plt.show()

data = make_moons(n_samples=10000, shuffle=True, noise=0.05, random_state=None)
X = data[0]
y = data[1]

# Add a crap ton of random features
X_random = np.random.normal(0, 1, (X.shape[0], 10))
X = np.hstack((X, X_random))

# Add poly features
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=42)
X_train = PolynomialFeatures(2).fit_transform(X_train)
orig_X_test = X_test
X_test = PolynomialFeatures(2).fit_transform(X_test)

y_pred = LogisticRegression(penalty='l1', C=1).fit(X_train, y_train).predict(X_test)
# y_pred = SVC(C=10).fit(X_train, y_train).predict(X_test)
print (y_test == y_pred).mean()
color_plot(orig_X_test[:,:2], y_pred)
