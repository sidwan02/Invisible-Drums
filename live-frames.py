import cv2

# https://github.com/ashwin-pajankar/Python-OpenCV3/blob/master/01%23%20Basics/prog18.py
def main():
    windowName = "Live Video Feed"
    cv2.namedWindow(windowName)
    cap = cv2.VideoCapture(0)

    while cap.isOpened:

        ret, frame = cap.read()

        output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Gray", output)
        cv2.imshow(windowName, frame)

        # 27 => Esc
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

    cap.release()


if __name__ == "__main__":
    main()
