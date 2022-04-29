import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import cv2
import copy

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


def mark_centroids(grayscale, cluster_centers, img_name_suff=""):
    grayscale_w_centroids = grayscale
    for (r, c) in cluster_centers:
        r, c = round(r), round(c)
        grayscale_w_centroids = cv2.circle(
            grayscale_w_centroids, (c, r), radius=1, color=(0, 0, 255), thickness=10
        )

    cv2.imwrite(
        f"centroid_imgs/grayscale_centroids{img_name_suff}.png", grayscale_w_centroids
    )


def mean_shift_custom(image, n_points=1, n_iter=10, radius=50):
    image = np.array(image)
    w, h = image.shape

    print(f"w: {w}, h: {h}")

    # points = np.array(
    #     [
    #         (x, y)
    #         for x, y in zip(
    #             random.sample(range(0, w), n_points),
    #             random.sample(range(0, h), n_points),
    #         )
    #     ]
    # )
    points = np.array([[300, 200]])

    mark_centroids(copy.deepcopy(image), points, img_name_suff=f"-0")

    cur_iter = 1

    while cur_iter <= n_iter:
        for i, (r, c) in enumerate(points):
            r, c = round(r), round(c)

            min_r, max_r = max(0, r - radius), min(h, r + radius)
            min_c, max_c = max(0, c - radius), min(w, c + radius)

            print(f"surr_r: [{min_r}, {max_r}]")
            print(f"surr_c: [{min_c}, {max_c}]")

            a = np.array(
                [
                    [surr_r, surr_c]
                    for surr_r in range(min_r, max_r + 1)
                    for surr_c in range(min_c, max_c + 1)
                ]
            )
            # 255 => white; # 0 => black
            # b = 255 - np.array(image[min_r : max_r + 1, min_c : max_c + 1]).flatten()

            b = 255 - np.array(
                [
                    image[surr_r, surr_c]
                    for surr_r in range(min_r, max_r + 1)
                    for surr_c in range(min_c, max_c + 1)
                ]
            )

            print(f"a.shape: {a.shape}, b.shape: {b.shape}")

            points[i] = sum(a * np.expand_dims(b, axis=1)) / sum(b)

        mark_centroids(copy.deepcopy(image), points, img_name_suff=f"-{cur_iter}")

        cur_iter += 1

    return points
