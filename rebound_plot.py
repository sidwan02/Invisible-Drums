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

    rebound_locs = [{"frame": "0", "loc": [0, 0]}]
    rebound_plot(args, rebound_locs)

