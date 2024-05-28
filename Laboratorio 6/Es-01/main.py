import cv2
from utils.display import *
from utils.threshold import *
from utils.morphological_ops import *
import numpy as np


def remove_meaningless_pxs(img):
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

    # Remove the non-yellow colors
    h_mask = threshold_band(h, 13, 25, 179, cv2.THRESH_BINARY)
    v_white_mask = threshold_high(v, 245, 255, cv2.THRESH_BINARY)
    mask = np.logical_or(h_mask > 0, v_white_mask > 0)
    v = v * mask

    # Remove what is too dark
    v_mask = threshold_high(v, 210, 255, cv2.THRESH_BINARY)
    mask = v_mask > 0
    v = v * mask

    # Return the image in RGB
    hsv = cv2.merge([h, s, v])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def compute_lines(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    thresh = 200
    tmp = cv2.Canny(img, thresh / 3, thresh)

    kern_dilate = Kernel(11).set_rectangular_centered(True, 1).get_kern()
    kern_dilate_2 = Kernel(11).set_rectangular_centered(True, 3).get_kern()
    kern_erode = Kernel(7).set_rectangular_centered(False, 1).get_kern()

    dilate_iter = 2

    # Fill lines, by enlarging horizontally
    tmp2 = close(tmp, kern_dilate, 2)

    # Enlarge horizontally even more to highlight further lines
    tmp2 = dilate(tmp2, kern_dilate_2, iterations=dilate_iter)

    # Delete with vertical lines to remove noise
    tmp2 = open(tmp2, kern_erode, 2)

    tmp2 = erode(tmp2, kern_dilate_2, iterations=dilate_iter)

    tmp = cv2.Canny(tmp2, thresh / 3, thresh)
    kern_final = Kernel(3).set_rectangular_centered(True, 1).get_kern()
    tmp = dilate(tmp, kern_final)
    display_image(tmp, greyscale=True)


## TODO Hough


def process_frame(img):
    cleaned = remove_meaningless_pxs(img)
    compute_lines(cleaned)


def process_image(fname):
    new_window(fname.replace("../test_images/", "").replace(".jpg", ""))
    img = cv2.imread(fname)
    process_frame(img)


def main():
    process_image("../test_images/challenge.jpg")
    # process_image("../test_images/solidYellowLeft.jpg")
    show_images()


if __name__ == "__main__":
    main()
