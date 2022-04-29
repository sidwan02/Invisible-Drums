import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# We will be using the make_blobs method
# in order to generate our own data.


# Proof of Concept
def POC():
    clusters = [[2, 2], [7, 7], [5, 13]]

    X, _ = make_blobs(n_samples=150, centers=clusters, cluster_std=0.60)

    # After training the model, We store the
    # coordinates for the cluster centers
    ms = MeanShift()
    ms.fit(X)

    cluster_centers = ms.cluster_centers_
    print(ms.labels_)

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

    # plt.show()


def thing(image, n_points=10, n_iter=100, radius=10):
    image = np.array(image)
    w, h = image.shape
    intensities = [[0] * w for _ in range(h)]

    points = np.array(
        [
            (x, y)
            for x, y in zip(
                random.sample((0, w), n_points), random.sample((0, h), n_points)
            )
        ]
    )

    cur_score = [0] * n_points

    cur_iter = 1

    while cur_iter <= n_iter:
        for i, (r, c) in enumerate(points):
            r, y = round(r), round(c)

            min_r, max_r = max(0, r - 10), min(h, r + 10)
            min_c, max_c = max(0, c - 10), min(w, c + 10)

            a = np.array(
                [
                    [surr_r, surr_c]
                    for rr in range(min_r, max_r + 1)
                    for cc in range(min_c, max_c + 1)
                ]
            )
            # 255 => white; # 0 => black
            b = 255 - image[min_r : max_r + 1, min_c : max_c + 1]

            points[i] = sum(a * np.expand_dims(b, axis=1)) / sum(b)

        cur_iter += 1
