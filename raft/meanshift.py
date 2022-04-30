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
    w_centroids = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)

    for (r, c) in cluster_centers:
        r, c = round(r), round(c)
        w_centroids = cv2.circle(
            w_centroids, (c, r), radius=3, color=(0, 0, 255), thickness=-1
        )

    return w_centroids


def mean_shift_custom(
    image, n_points=50, n_iter=50, radius=50, centroids_path=None, meanshift_path=None
):
    image = np.array(image)
    h, w = image.shape

    # print(f"w: {w}, h: {h}")

    points = np.array(
        [
            (r, c)
            for r, c in zip(
                random.sample(range(0, h), n_points),
                random.sample(range(0, w), n_points),
            )
        ]
    )
    # points = np.array([[300, 600]])
    # points = np.array([[250, 350]])

    radius_orig = radius

    points_radii = [radius_orig] * n_points

    # print(f"points: {points}")

    all_frames = []
    frame0 = mark_centroids(copy.deepcopy(image), points, img_name_suff=f"-0")
    all_frames.append(frame0)

    cur_iter = 1

    while cur_iter <= n_iter:
        print(f"cur_iter: {cur_iter}")
        for i, (r, c) in enumerate(points):
            r, c = round(r), round(c)

            min_r, max_r = max(0, r - points_radii[i]), min(h - 1, r + points_radii[i])
            min_c, max_c = max(0, c - points_radii[i]), min(w - 1, c + points_radii[i])

            # print(f"surr_r: [{min_r}, {max_r}]")
            # print(f"surr_c: [{min_c}, {max_c}]")

            a = np.array(
                [
                    [surr_r, surr_c]
                    for surr_r in range(min_r, max_r + 1)
                    for surr_c in range(min_c, max_c + 1)
                ]
            )

            def surrounding_avg_intensity(radius):
                min_r, max_r = max(0, r - radius), min(h - 1, r + radius)
                min_c, max_c = max(0, c - radius), min(w - 1, c + radius)

                # 255 => white; # 0 => black
                x = np.array(image[min_r : max_r + 1, min_c : max_c + 1]).flatten()

                # b = np.e ** (-x + 4)

                b = 255 - x

                return b

            b = surrounding_avg_intensity(points_radii[i])

            # print(f"a.shape: {a.shape}, b.shape: {b.shape}")

            # def get_radius_from_avg_intensity(avg_intensity):
            #     print(f"avg_intensity: {avg_intensity}")
            #     # # radius=800 if avg 0
            #     # # radius=10 if avg 255
            #     # m = (800 - 10) / (0 - 255)
            #     # c = 800
            #     # return m * avg_intensity + c

            #     return int((8000) / (avg_intensity + 10) - 10)

            # points_radii[i] = get_radius_from_avg_intensity(np.average(b))
            # points_radii[i] = 800
            # print(f"radius: {points_radii[i]}")

            # if np.average(b) < 10:
            #     points_radii[i] *= 2
            # else:
            #     points_radii[i] = radius_orig

            # print(f"avg_intensity: {np.average(b)}")

            if sum(b) != 0:
                new_points = sum(a * np.expand_dims(b, axis=1)) / sum(b)
                # print(f"points[i]: {points[i]}")

                shift = np.sqrt(
                    (points[i][0] - new_points[0]) ** 2
                    + (points[i][1] - new_points[1]) ** 2
                )
                points[i] = new_points

                if shift < 5:
                    points_radii[i] *= 2
                else:
                    points_radii[i] = radius_orig

                local_fixed_r_intensity = np.average(surrounding_avg_intensity(20))

                if local_fixed_r_intensity > 50:
                    points_radii[i] = radius_orig
                else:
                    points_radii[i] *= 2
            else:
                points_radii[i] *= 2

            # if the sum is 0 that means all surrounding pixels are pure white 255. For now if that's the case don't update points which makes sense logically speaking

        frame = mark_centroids(
            copy.deepcopy(image), points, img_name_suff=f"-{cur_iter}"
        )
        all_frames.append(frame)

        cur_iter += 1

    if centroids_path is not None:
        out = cv2.VideoWriter(
            centroids_path, cv2.VideoWriter_fourcc(*"DIVX"), 15, (w, h)
        )

        for frame in all_frames:
            out.write(frame)
        out.release()

    if meanshift_path is not None:
        cv2.imwrite(meanshift_path, all_frames[-1])

    return [[r, c, image[r, c]] for (r, c) in points]
