from argparse import ArgumentParser, BooleanOptionalAction
from sklearn.cluster import MeanShift, MiniBatchKMeans
import cv2
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from PIL import Image
from itertools import chain
from meanshift import mean_shift_custom
import shutil


def load_image(imfile, resolution=None):
    img = Image.open(imfile)
    if resolution:
        img = img.resize(resolution, PIL.Image.ANTIALIAS)
    img = np.array(img).astype(np.uint8)
    return img


def place_centroids(img):
    img = np.array(img)

    print(f"img.shape: {img.shape}")

    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("grayscale.png", grayscale)

    # h, w = len(grayscale), len(grayscale[0])

    # intensity_weighted_pixels = []

    # def intensity_to_weight(intensity):
    #     # 255 => while; # 0 => black
    #     return (255 - intensity) // 50

    # [
    #     intensity_weighted_pixels.append(
    #         [[r, c]] * (intensity_to_weight(grayscale[r][c]))
    #     )
    #     for r in range(h)
    #     for c in range(w)
    # ]

    # # print(f"intensity_weighted_pixels: {intensity_weighted_pixels}")

    # X = np.array(list(chain(*intensity_weighted_pixels)))

    # print(f"X.shape: {X.shape}")

    # --> only for debugging purposes
    # ax.scatter(X[:, 0], X[:, 1], marker=",", alpha=0.01, color="black")

    # plt.show()

    # # plt.savefig("saved_figure.png")

    # assert 1 == 0
    # only for debugging purposes <--

    """
    # --> Attempt MeanShift

    ms = MeanShift(max_iter=0)
    ms.fit(X)
    cluster_centers = ms.cluster_centers_

    # Attempt MeanShift <--
    """

    # """
    # --> Attempt mean_shift_custom

    cluster_centers = mean_shift_custom(grayscale)

    # Attempt MeanShift <--
    # """

    """
    # --> Attempt KMeans

    clusters = MiniBatchKMeans(n_clusters=5).fit(X)
    cluster_centers = clusters.cluster_centers_

    # Attempt KMeans <--
    # """

    grayscale_w_centroids = grayscale
    for (x, y) in cluster_centers:
        x, y = round(x), round(y)
        grayscale_w_centroids = cv2.circle(
            grayscale_w_centroids, (x, y), radius=10, color=(0, 0, 255), thickness=10
        )

    cv2.imwrite("grayscale_centroids.png", grayscale_w_centroids)

    """
    # --> only for debugging purposes

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.set_xlim([0, w])
    ax.set_ylim([0, h])

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
    # only for debugging purposes <--
    """

    print(f"cluster_centers: {cluster_centers}")

    assert 0 == 1


def main(args):
    # os.system(f"python run_inference.py")
    data_path = args.path

    img_flo_path = data_path + "/FlowImages_gap-1"  # path to the dataset
    superfolder = glob.glob(os.path.join(img_flo_path, "*"))

    outroot = data_path + f"/Centroids/"

    for folder in superfolder:
        img_path1 = os.path.join(folder, "*.png")
        img_path2 = os.path.join(folder, "*.jpg")
        images = glob.glob(img_path1) + glob.glob(img_path2)

        # print(f"images: {images}")

        floout = os.path.join(outroot, folder)

        if args.clear and os.path.exists(floout):
            shutil.rmtree(floout)
        os.makedirs(floout, exist_ok=True)

        if args.clear and os.path.exists("centroid_imgs/"):
            shutil.rmtree("centroid_imgs/")
        os.makedirs("centroid_imgs/", exist_ok=True)

        images_ = sorted(images)

        for index, imfile1 in enumerate(images_):
            image1 = load_image(imfile1)
            svfile = imfile1

            flopath = os.path.join(floout, os.path.basename(svfile))

            processed_img = place_centroids(image1)

            cv2.imwrite(flopath[:-4] + ".png", processed_img)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--path", type=str, default="../data/custom")
    parser.add_argument("--clear", action=BooleanOptionalAction)

    args = parser.parse_args()

    main(args)
