import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# We will be using the make_blobs method
# in order to generate our own data.

clusters = [[2, 2], [7, 7], [5, 13]]

X, _ = make_blobs(n_samples=150, centers=clusters, cluster_std=0.60)

# After training the model, We store the
# coordinates for the cluster centers
ms = MeanShift()
ms.fit(X)
cluster_centers = ms.cluster_centers_

# Finally We plot the data points
# and centroids in a 3D graph.
fig = plt.figure()

ax = fig.add_subplot(111)

ax.scatter(X[:, 0], X[:, 1], marker="o")

ax.scatter(
    cluster_centers[:, 0],
    cluster_centers[:, 1],
    marker="x",
    color="red",
    s=300,
    linewidth=5,
    zorder=10,
)

plt.show()
