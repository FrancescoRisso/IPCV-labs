import cv2


class Camera:
    def __init__(self, index):
        self.capture = cv2.VideoCapture(index)

    def getFrame(self):
        return self.capture.read()[1]

    def isOpened(self):
        return self.capture.isOpened()

    def closeCapture(self):
        return self.capture.release()
