import cv2
import numpy as np


def dilate(img, kernel, iterations=1):
    for _ in range(iterations):
        img = cv2.dilate(img, kernel, iterations)
    return img


def erode(img, kernel, iterations=1):
    for _ in range(iterations):
        img = cv2.erode(img, kernel, iterations)
    return img


def close(img, kernel, iterations=1):
    img = dilate(img, kernel, iterations)
    return erode(img, kernel, iterations)


def open(img, kernel, iterations=1):
    img = erode(img, kernel, iterations)
    return dilate(img, kernel, iterations)


class Kernel:
    size = 0
    center = 0
    kernel = np.ones(1)

    def __init__(self, size):
        self.size = size
        self.center = int((size + 1) / 2)
        self.kernel = np.ones((size, size), np.uint8)

    def set_rectangular_centered(self, horiz, width):
        self.kernel = np.zeros((self.size, self.size), np.uint8)
        low = int(self.center - width / 2)
        hi = int(self.center + width / 2)

        if horiz:
            self.kernel[low:hi, :] = 1
        else:
            self.kernel[:, low:hi] = 1
        return self

    def get_kern(self):
        return self.kernel
