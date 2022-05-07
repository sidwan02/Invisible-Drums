import cv2
import os
from argparse import ArgumentParser, BooleanOptionalAction
import shutil


def save_video_frames(args):
    # make new directory

    # os.mkdir(new_dir_name)
    print(f"args: {args}")

    if args.clear and os.path.exists(args.new_dir_name):
        shutil.rmtree(args.new_dir_name)
    os.makedirs(args.new_dir_name, exist_ok=True)

    video = cv2.VideoCapture(args.video_path)
    # read in first frame
    read_success, image = video.read()
    frame_count = 0
    while read_success:
        # save curr frame as jpg (e.g. 0.jpg, 1.jpg, etc)
        cv2.imwrite("{}/{}.jpg".format(args.new_dir_name, frame_count), image)
        # read new frame
        read_success, image = video.read()
        print("Read a new frame: ", read_success)
        frame_count += 1


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--video_path", type=str, default="./Videos/Anish-Slow.mp4")
    parser.add_argument(
        "--new_dir_name", type=str, default="./data/custom/JPEGImages/Anish-Slow/"
    )
    parser.add_argument("--clear", action=BooleanOptionalAction)

    args = parser.parse_args()

    save_video_frames(args)
