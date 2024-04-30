import matplotlib.pyplot as plt

ROWS = 1
COLS = 1


def set_grid_size(rows: int, cols: int):
    global ROWS, COLS

    ROWS = rows
    COLS = cols


def display_image(img, title="", index=1, greyscale=False):
    plt.subplot(ROWS, COLS, index)
    plt.imshow(img, cmap="gray" if greyscale else "viridis")
    plt.title(title)
    plt.axis(False)


def show_images():
    plt.show()
