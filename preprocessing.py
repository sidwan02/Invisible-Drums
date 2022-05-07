import cv2
import os


def save_video_frames(video_path, new_dir_name):
    # make new directory
    os.mkdir(new_dir_name)

    video = cv2.VideoCapture(video_path)
    # read in first frame
    read_success, image = video.read()
    frame_count = 0
    while read_success:
        # save curr frame as jpg (e.g. 0.jpg, 1.jpg, etc)
        cv2.imwrite("{}/{}.jpg".format(new_dir_name, frame_count), image)
        # read new frame
        read_success, image = video.read()
        print("Read a new frame: ", read_success)
        frame_count += 1


save_video_frames("scorpionfish.mp4", "fishdir2")
