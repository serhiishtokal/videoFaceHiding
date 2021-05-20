import numpy as np
import cv2
import sys


class AnonymizeMethod:
    PIXELATE = 1,
    BLUR = 2,
    BlackRectangle = 3


def _pixelate_image(image, blocks=10):
    (h, w) = image.shape[:2]
    x_steps = np.linspace(0, w, blocks + 1, dtype="int")
    y_steps = np.linspace(0, h, blocks + 1, dtype="int")

    for i in range(1, len(y_steps)):
        for j in range(1, len(x_steps)):
            start_x = x_steps[j - 1]
            start_y = y_steps[i - 1]
            end_x = x_steps[j]
            end_y = y_steps[i]

            roi = image[start_y:end_y, start_x:end_x]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y),
                          (B, G, R), -1)

    # return the pixelated blurred image
    return image


def _blurImage(image):
    result = image
    if image.shape[0] != 0 and image.shape[1] != 0:
        result = cv2.GaussianBlur(image, (151, 151), 0)
    return result


def _blackImage(image):
    start_point = (0, 0)
    end_point = (image.shape[1], image.shape[0])
    color = (0, 0, 0)
    thickness = -1
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return image


def anonymize_face(image, method=AnonymizeMethod.PIXELATE):
    switcher = {
        AnonymizeMethod.PIXELATE: _pixelate_image,
        AnonymizeMethod.BLUR: _blurImage,
        AnonymizeMethod.BlackRectangle: _blackImage
    }
    anonymized_face = switcher.get(method, lambda: __invalid_method_exception())(image)
    return anonymized_face


def __invalid_method_exception():
    print("Invalid detection method:", sys.exc_info()[0])
    raise
