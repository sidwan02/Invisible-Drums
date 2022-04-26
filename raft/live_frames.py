import cv2
from predict import predict_live
from argparse import ArgumentParser

# https://github.com/ashwin-pajankar/Python-OpenCV3/blob/master/01%23%20Basics/prog18.py
def main(args):

    windowName = "Live Video Feed"
    cv2.namedWindow(windowName)
    cap = cv2.VideoCapture(0)

    prev_frame = None

    while cap.isOpened:

        ret, frame = cap.read()

        image_flow = predict_live(args, prev_frame, frame)

        cv2.imshow(windowName, frame)

        if image_flow:
            cv2.imshow("Image Flow", image_flow)

        prev_frame = frame

        # 27 => Esc
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

    cap.release()


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument('--resolution', nargs='+', type=int)
    parser.add_argument("--model", help="restore checkpoint")

    # required for RAFT
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )

    args = parser.parse_args()

    main(args)
