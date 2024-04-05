import cv2
import matplotlib.pyplot as plt
import numpy as np


def handle_close(event, capture):
    capture.release()


def getFrameWithLogo(frame, logo):

    subset = frame[-100:, -100:]

    _, logoFilledPx = cv2.threshold(logo[:, :, 3], 0, 255, cv2.THRESH_BINARY)
    logoEmptyPx = cv2.bitwise_not(logoFilledPx)

    logoWithoutEmtpyPx = cv2.bitwise_and(logo, logo, mask=logoFilledPx)
    subsetWithoutLogoFilledPx = cv2.bitwise_and(subset, subset, mask=logoEmptyPx)

    logoWithoutEmtpyPx = cv2.cvtColor(logoWithoutEmtpyPx, cv2.COLOR_BGRA2BGR)

    frame[-100:, -100:] = cv2.add(logoWithoutEmtpyPx, subsetWithoutLogoFilledPx)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main():
    # Init the camera
    capture = cv2.VideoCapture(1)

    # enable Matplotlib interactive mode in order to visualize video flows and avoid the frames to be overlapped
    plt.ion()

    logo = cv2.imread("./Logo.png", cv2.IMREAD_UNCHANGED)

    # Create a figure to be updated: at every frame the image is updated
    fig = plt.figure()
    fig.canvas.mpl_connect("close_event", lambda event: handle_close(event, capture))

    _, frame = capture.read()

    ax_img = plt.imshow(getFrameWithLogo(frame, logo))
    # ax_img = plt.imshow(frame)
    plt.axis("off")
    plt.title("Camera Capture")
    plt.show()

    while capture.isOpened():
        _, frame = capture.read()

        ax_img.set_data(getFrameWithLogo(frame, logo))
        # ax_img.set_data(frame)
        fig.canvas.draw()  # Update the figure associated to the shown plot
        fig.canvas.flush_events()  # If there is any enqueued event, it gets removed to avoid errors
        # plt.pause(1 / 30)  # 30 frames per second


if __name__ == "__main__":
    main()


# non si muove
