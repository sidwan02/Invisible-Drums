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

    frames_written = 0

    frame_count = 0
    while read_success:
        if frame_count % args.skip == 0:
            # save curr frame as jpg (e.g. 0.jpg, 1.jpg, etc)
            image = cv2.resize(image, (856, 480), cv2.INTER_NEAREST)

            cv2.imwrite("{}/{}.jpg".format(args.new_dir_name, frames_written), image)
            # read new frame
            print(f"Read frame {frame_count}")
            frames_written += 1
        read_success, image = video.read()
        frame_count += 1


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--video_path", type=str, default="./Videos/Anish-Slow.mp4")
    parser.add_argument(
        "--new_dir_name", type=str, default="./data/custom/JPEGImages/Anish-Slow/"
    )
    parser.add_argument("--skip", type=int, default=5)
    parser.add_argument("--clear", action=BooleanOptionalAction)

    args = parser.parse_args()

    save_video_frames(args)
