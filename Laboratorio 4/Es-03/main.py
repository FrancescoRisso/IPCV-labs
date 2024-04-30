from utils.display import set_grid_size, display_image, show_images, new_window
from utils.fourier import transform, antitransform
from utils.masks import circular_mask
import cv2


GRAPHS_PER_SETTING = 6


def process_image(path):
    radii = [1, 2, 4, 8, 16, 32, 64]

    new_window(path.replace("../", ""))
    set_grid_size(GRAPHS_PER_SETTING, len(radii))

    img = cv2.imread(path)

    (mag, ph) = transform(img, log=False)
    w, h = mag.shape

    for i, r in enumerate(radii):
        filter = circular_mask(w, h, radius=r, zeros_inside=True)
        mag_filt = mag * filter
        result = antitransform(mag_filt, ph, delog=False)

        index = i + 1

        display_image(img, "Original", index)
        display_image(mag, "Mag", index + len(radii), greyscale=True)
        display_image(ph, "Ph", index + 2 * len(radii), greyscale=True)
        display_image(filter, f"Filter (r={r})", index + 3 * len(radii), True)
        display_image(mag_filt, f"Filtered mag", index + 4 * len(radii), True)
        display_image(result, "Final", index + 5 * len(radii), greyscale=True)


def main():
    process_image("../circle.png")
    process_image("../sinFunction.png")

    show_images()


if __name__ == "__main__":
    main()
