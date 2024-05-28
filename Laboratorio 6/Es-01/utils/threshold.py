import cv2


# 1 is on the highest values of img
def threshold_high(img, val, max_val, type):
    _, res = cv2.threshold(img, val, max_val, type)
    return res


# 1 is on the lowest values of img
def threshold_low(img, val, max_val, type):
    return max_val - threshold_high(img, val, max_val, type)


# 1 is in the middle band
def threshold_band(img, val_low, val_high, max_val, type):
    return (
        threshold_high(img, val_low, max_val, type)
        * threshold_low(img, val_high, max_val, type)
        / max_val
    )
