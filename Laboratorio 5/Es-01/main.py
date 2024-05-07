from utils.camera import Camera
from utils.display import (
    init_display_video,
    show_video_frame,
    set_grid_size,
)

import cv2


def main():
    cam = Camera(0)

    set_grid_size(2, 3)

    def closeCameras(_):
        cam.closeCapture()

    init_display_video(closeCameras, "Prova")

    while cam.isOpened():
        frame = cam.getFrame()

        canny = cv2.Canny(frame, 75, 225)

        sobel_x = cv2.convertScaleAbs(cv2.Sobel(frame, cv2.CV_16S, 1, 0, 2))
        sobel_x = cv2.cvtColor(sobel_x, cv2.COLOR_BGR2GRAY)
        sobel_y = cv2.convertScaleAbs(cv2.Sobel(frame, cv2.CV_16S, 0, 1, 2))
        sobel_y = cv2.cvtColor(sobel_y, cv2.COLOR_BGR2GRAY)
        sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

        show_video_frame(frame, "Original", index=1)
        show_video_frame(canny, "Canny(75, 225)", index=3, greyscale=True)
        show_video_frame(sobel_x, "Horizontal Sobel", index=4, greyscale=True)
        show_video_frame(sobel_y, "Vertical Sobel", index=5, greyscale=True)
        show_video_frame(sobel, "Sobel", index=6, greyscale=True)


if __name__ == "__main__":
    main()
