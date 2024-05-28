import cv2.data
from utils.camera import Camera
from utils.display import init_display_video, show_video_frame, update_frame
import cv2
import matplotlib.patches as patches


def main():
    cam = Camera(0)

    def closeCameras(_):
        cam.closeCapture()

    init_display_video(closeCameras)

    classifier = cv2.CascadeClassifier()
    classifier.load("../data/haarcascades/haarcascade_frontalface_default.xml")

    while cam.isOpened():
        frame = cam.getFrame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        grey = cv2.equalizeHist(grey)

        h = grey.shape[1]
        h_20_perc = int(h * 0.2)

        for rect in classifier.detectMultiScale(grey, minSize=[h_20_perc, h_20_perc]):
            frame = cv2.rectangle(
                frame,
                (rect[0], rect[1]),
                (rect[0] + rect[2], rect[1] + rect[3]),
                (255, 0, 0),
                2,
            )

        show_video_frame(frame, "Original")
        update_frame()


if __name__ == "__main__":
    main()
