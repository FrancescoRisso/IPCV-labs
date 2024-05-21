from utils.morphological_ops import *
from utils.display import display_image, show_images, set_grid_size
import cv2
import numpy as np


def main():
    set_grid_size(2, 2)

    img = cv2.imread("../snowy-street.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    display_image(img, greyscale=True, index=1, title="Original image")
    display_image(img, greyscale=True, index=2, title="Original image")

    kern_rem = np.ones((3, 3))
    rem = open(img, kern_rem, 3)
    display_image(rem, greyscale=True, index=3, title="Removed snowflakes")

    kern_incr = np.ones((3, 3))
    incr = dilate(img, kern_incr, 3)
    display_image(incr, greyscale=True, index=4, title="Increased snowflakes")
    show_images()


if __name__ == "__main__":
    main()
