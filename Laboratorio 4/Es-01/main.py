from utils.display import set_grid_size, display_image, show_images, new_window
from utils.fourier import transform, antitransform
import cv2


LOG = True


def process_image(path):
    new_window(path.replace("../", ""))
    set_grid_size(1, 4)

    img = cv2.imread(path)

    (mag, ph) = transform(img, log=LOG)

    display_image(img, "Original", 1)
    display_image(mag, "Magnitude", 2, greyscale=True)
    display_image(ph, "Phase", 3, greyscale=True)
    display_image(antitransform(mag, ph, delog=LOG), "Reconstructed", 4, greyscale=True)


def main():
    process_image("../circle.png")
    process_image("../sinFunction.png")
    
    show_images()


if __name__ == "__main__":
    main()
