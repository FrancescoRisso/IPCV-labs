import numpy as np


def circular_mask(width, height, center=None, radius=None, zeros_inside=False):
    if center is None:
        center = (int(width / 2), int(height / 2))

    if radius is None:
        radius = min(center[0], center[1], width - center[0], height - center[1])

    X, Y = np.ogrid[:height, :width]
    distFromCenter = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    if zeros_inside:
        return distFromCenter >= radius
    return distFromCenter <= radius
