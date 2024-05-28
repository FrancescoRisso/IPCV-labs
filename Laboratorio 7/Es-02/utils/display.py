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


def new_window(title=""):
    plt.figure(title)


def init_display_video(onClose, title=""):
    plt.ion()
    fig = plt.figure()
    fig.canvas.mpl_connect("close_event", lambda event: onClose(event))


def show_video_frame(frame, title="", index=1, greyscale=False):
    display_image(frame, title, index, greyscale)
    plt.gcf().canvas.flush_events()


def update_frame():
    plt.draw()
