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

# Make dataset
np.random.seed(1)  # DO THIS
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

from src.autostat import AutomaticStatistician
from src.datasets import Dataset

ds = Dataset(X, y, scale_predictors=False)
autostat = AutomaticStatistician(discount=0.5, perf_weight=100.0, depth=3, n_sim=3)
autostat.run(ds, time_limit=60.)

