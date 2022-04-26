import cv2
from predict import predict_live

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
        cv2.imshow("Image Flow", image_flow)

        prev_frame = frame

        # 27 => Esc
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--resolution', nargs='+', type=int)
    parser.add_argument("--model", help="restore checkpoint")
    parser.add_argument("--path", help="dataset for prediction")
    parser.add_argument("--gap", type=int, help="gap between frames")
    parser.add_argument("--outroot", help="path for output flow as image")
    parser.add_argument("--reverse", type=int, help="video forward or backward")
    parser.add_argument("--raw_outroot", help="path for output flow as xy displacement")

    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument(
        "--alternate_corr",
        action="store_true",
        help="use efficent correlation implementation",
    )
    parser.add_argument("--clear", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    main(args)
