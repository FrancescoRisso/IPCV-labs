import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_hist(frame, title, index, iscv2):
    plt.subplot(3, 3, index)
    plt.title(title)
    if iscv2:
        plt.plot(cv2.calcHist(frame, [0], None, [256], [0, 255]), "k-")
    else:
        plt.hist(frame.ravel(), 256, [0, 255], color="black")
    plt.xlim([0, 256])


def plot_img(frame, title):
    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis(False)


def display_with_hist(frame, title):
    plot_img(frame, title)

    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    plot_hist(frame_grey, "Ist. grigio (cv2)", 2, True)
    plot_hist(frame_grey, "Ist. grigio (plt)", 3, False)

    plot_hist(frame[:, :, 2], "Ist. R (cv2)", 4, True)
    plot_hist(frame[:, :, 2], "Ist. R (plt)", 7, False)

    plot_hist(frame[:, :, 1], "Ist. G (cv2)", 5, True)
    plot_hist(frame[:, :, 1], "Ist. G (plt)", 8, False)

    plot_hist(frame[:, :, 0], "Ist. B (cv2)", 6, True)
    plot_hist(frame[:, :, 0], "Ist. B (plt)", 9, False)


def main():
    frame = cv2.imread("./pastelli.jpeg")

    plt.figure(0)
    display_with_hist(frame, "Immagine originale")

    plt.figure(1)
    (r, g, b) = cv2.split(frame)
    auto_eq = cv2.merge((cv2.equalizeHist(r), cv2.equalizeHist(g), cv2.equalizeHist(b)))
    display_with_hist(auto_eq, "Equalizzazione automatica")

    plt.show()


if __name__ == "__main__":
    main()
