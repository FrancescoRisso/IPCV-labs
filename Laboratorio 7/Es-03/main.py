import cv2.data
from utils.camera import Camera
from utils.display import init_display_video, show_video_frame, update_frame
import cv2
import matplotlib.patches as patches


def highlight(frame, grey, classifier, fact_min, fact_max):
    h = grey.shape[1]
    min = int(h * fact_min)
    max = int(h * fact_max)

    for rect in classifier.detectMultiScale(
        grey, minSize=[min, min], maxSize=[max, max]
    ):
        frame = cv2.rectangle(
            frame,
            (rect[0], rect[1]),
            (rect[0] + rect[2], rect[1] + rect[3]),
            (255, 0, 0),
            2,
        )

    return frame


def main():
    cam = Camera(0)

    def closeCameras(_):
        cam.closeCapture()

    init_display_video(closeCameras)

    classifiers = []

    for path in [
        "../data/haarcascades/haarcascade_eye.xml",
        # "../data/haarcascades/haarcascade_smile.xml",
    ]:
        classifier = cv2.CascadeClassifier()
        classifier.load(path)
        classifiers.append(classifier)

    while cam.isOpened():
        frame = cam.getFrame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        grey = cv2.equalizeHist(grey)

        for classifier in classifiers:
            frame = highlight(frame, grey, classifier, 0.05, 0.2)

        show_video_frame(frame, "Original")
        update_frame()


if __name__ == "__main__":
    main()
