import cv2
import matplotlib.pyplot as plt
import numpy as np


FIGURE_ROWS = 2
FIGURE_COLS = 3


def printImg(img, subplot, title):
	plt.subplot(FIGURE_ROWS, FIGURE_COLS, subplot)
	plt.imshow(img)
	plt.title(title)
	plt.axis(False)


def main():
	img = cv2.imread("./pastelli.jpeg")

	printImg(img, 1, "Immagine originale")

	pixels = img[1700:2700, 300:5675]
	printImg(pixels, 2, "Sottoinsieme di pixel")

	printImg(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB), 3, "Pixel convertiti in RGB")

	printImg(cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV), 4, "Pixel convertiti in HSV")
	
	pixels[:, :, 2] = np.zeros((pixels.shape[0], pixels.shape[1]))
	printImg(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB), 5, "Pixel con R=0 (RGB)")
	
	pixels[:, :, 2] = 255 * np.ones((pixels.shape[0], pixels.shape[1]))
	printImg(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB), 6, "Pixel con R=255 (RGB)")

	plt.show()


if __name__ == "__main__":
	main()
