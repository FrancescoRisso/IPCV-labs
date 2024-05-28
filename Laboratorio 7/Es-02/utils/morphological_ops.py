import cv2


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
