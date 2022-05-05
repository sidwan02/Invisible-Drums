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
from main import run_single_iteration
import shutil


def load_image(imfile, resolution=None):
    img = Image.open(imfile)
    if resolution:
        img = img.resize(resolution, PIL.Image.ANTIALIAS)
    img = np.array(img).astype(np.uint8)
    return img


def main(args):
    # os.system(f"python run_inference.py")
    data_path = args.path

    print(f"data_path: {data_path}")

    img_flo_path = data_path + "/FlowImages_gap-1"  # path to the dataset
    superfolder = glob.glob(os.path.join(img_flo_path, "*"))

    centroids_root = data_path + f"/Centroids/"
    meanshift_root = data_path + f"/Mean-Shift/"
    blobs_root = data_path + f"/Blobs/"

    blobs_data_folders = []
    # flow_grayscale_folders = []

    print(f"superfolder: {superfolder}")

    for folder in superfolder:
        all_blobs_data = []
        # all_flow_grayscale = []
        img_path1 = os.path.join(folder, "*.png")
        img_path2 = os.path.join(folder, "*.jpg")
        images = glob.glob(img_path1) + glob.glob(img_path2)

        # print(f"images: {images}")

        f = os.path.basename(folder)
        centroids_dir_path = os.path.join(centroids_root, f)
        meanshift_dir_path = os.path.join(meanshift_root, f)
        blobs_dir_path = os.path.join(blobs_root, f)

        # print(f"centroids_dir_path: {centroids_dir_path}")
        # print(f"meanshift_dir_path: {meanshift_dir_path}")

        if args.clear and os.path.exists(centroids_dir_path):
            shutil.rmtree(centroids_dir_path)
        os.makedirs(centroids_dir_path, exist_ok=True)

        if args.clear and os.path.exists(meanshift_dir_path):
            shutil.rmtree(meanshift_dir_path)
        os.makedirs(meanshift_dir_path, exist_ok=True)

        if args.clear and os.path.exists(blobs_dir_path):
            shutil.rmtree(blobs_dir_path)
        os.makedirs(blobs_dir_path, exist_ok=True)

        images_ = sorted(images)

        for index, imfile1 in enumerate(images_):
            image1 = np.array(load_image(imfile1))
            svfile = imfile1

            centroids_path = os.path.join(centroids_dir_path, os.path.basename(svfile))
            meanshift_path = os.path.join(meanshift_dir_path, os.path.basename(svfile))
            blobs_path = os.path.join(blobs_dir_path, os.path.basename(svfile))

            grayscale = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            # cv2.imwrite("grayscale.png", grayscale)
            # all_flow_grayscale.append(grayscale)

            cluster_centers = mean_shift_custom(
                grayscale,
                centroids_path=centroids_path[:-4] + "-meanshift.avi",
                meanshift_path=meanshift_path[:-4] + "-centroids.png",
            )

            blobs_data = run_single_iteration(
                cluster_centers, grayscale, blobs_path=blobs_path[:-4] + "-blobs.png"
            )

            all_blobs_data.append(blobs_data)

        blobs_data_folders.append(all_blobs_data)
        # flow_grayscale_folders.append(all_flow_grayscale)

    np.save(data_path + "/all_blobs_data.npy", blobs_data_folders)
    # np.save(data_path + "/all_flow_grayscale.npy", flow_grayscale_folders)
    # print(cluster_centers_folders)
    # return cluster_centers_folders


# def get_centroids():
#     parser = ArgumentParser()

#     args = parser.parse_args()
#     args.path = "./data/custom"  # no .. since this is being called from run.py which is at the root of the repo
#     args.clear = True

#     print(f"in get_centroids")

#     cluster_centers_folders = main(args)
#     return cluster_centers_folders


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--path", type=str, default="./data/custom")
    parser.add_argument("--clear", action=BooleanOptionalAction)

    args = parser.parse_args()

    main(args)
