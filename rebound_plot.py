from argparse import ArgumentParser, BooleanOptionalAction
import numpy as np
import os
import shutil
import cv2
import copy
from centroids import load_image


def mark_rebound(img, rebound_loc):
    r, c = rebound_loc
    r, c = round(r), round(c)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.circle(img, (c, r), radius=15, color=(255, 0, 0), thickness=-1)

    return img


def rebound_plot(args, rebound_locs):
    rgbpath = args.path + "/JPEGImages/Anish-Slow/"  # path to the dataset

    for rebound in rebound_locs:
        image = np.array(load_image(f"{rgbpath}{rebound['frame']}.jpg"))
        w_rebound = mark_rebound(copy.deepcopy(image), rebound["loc"])

        rebound_path = f"{rebound_dir_path}{rebound['frame']}-rebound.png"

        cv2.imwrite(rebound_path, w_rebound)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--path", type=str, default="./data/custom")
    parser.add_argument("--clear", action=BooleanOptionalAction)

    args = parser.parse_args()

    rebound_dir_path = args.path + f"/Rebound/Anish-Slow/"
    if args.clear and os.path.exists(rebound_dir_path):
        shutil.rmtree(rebound_dir_path)
    os.makedirs(rebound_dir_path, exist_ok=True)

    rebound_locs = [
        {"frame": "15", "loc": [301.0, 489.0]},
        {"frame": "52", "loc": [159.28571428571428, 430.0]},
        {"frame": "76", "loc": [225.0, 455.0]},
        {"frame": "89", "loc": [358.0, 291.0]},
        {"frame": "106", "loc": [449.7368421052632, 433.3157894736842]},
        {"frame": "119", "loc": [284.0, 338.0]},
        {"frame": "140", "loc": [135.0, 506.0]},
        {"frame": "169", "loc": [209.0, 427.0]},
        {"frame": "189", "loc": [239.53846153846155, 348.53846153846155]},
        {"frame": "199", "loc": [431.54545454545456, 183.1818181818182]},
    ]
    rebound_plot(args, rebound_locs)

