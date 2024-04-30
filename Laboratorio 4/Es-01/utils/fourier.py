import cv2
import numpy as np


def transform(img, log=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    compl = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

    compl = np.fft.fftshift(compl)

    # mag = cv2.magnitude(compl[:, :, 0], compl[:, :, 1])
    # ph = cv2.phase(compl[:, :, 0], compl[:, :, 1])
    mag, ph = cv2.cartToPolar(compl[:, :, 0], compl[:, :, 1])

    if log:
        return (20 * np.log(mag + 1), ph)
    return (mag, ph)


def antitransform(mag, ph, delog=False):
    if delog:
        mag = np.exp(mag / 20) - 1

    re, im = cv2.polarToCart(mag, ph)
    img = cv2.merge([re, im])

    antitr = cv2.idft(img)
    return cv2.magnitude(antitr[:, :, 0], antitr[:, :, 1])
