import numpy as np
from src.gaussian_process import *
from src.bayesian_optimizer import *
from src.kernels import *

# Data
from sklearn.datasets import make_blobs

np.random.seed(10)

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
# data = make_moons(n_samples=10000, shuffle=True, noise=0.1, random_state=None)
X = np.vstack((data1[0], data2[0]))
y = np.concatenate((data1[1], data2[1]))


from src.random_search import RandomSearcher
from src.datasets import Dataset

ds = Dataset(X, y)
rs = RandomSearcher()
rs.run(ds, time_limit=60.)

