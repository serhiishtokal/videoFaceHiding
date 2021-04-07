import numpy as np
import cv2


def anonymize_face_pixelate(image, blocks=10):
    # divide the input image into NxN blocks
    (h, w) = image.shape[:2]
    x_steps = np.linspace(0, w, blocks + 1, dtype="int")
    y_steps = np.linspace(0, h, blocks + 1, dtype="int")

    # loop over the blocks in both the x and y direction
    for i in range(1, len(y_steps)):
        for j in range(1, len(x_steps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            start_x = x_steps[j - 1]
            start_y = y_steps[i - 1]
            end_x = x_steps[j]
            end_y = y_steps[i]

            # extract the ROI using NumPy array slicing, compute the
            # mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original image
            roi = image[start_y:end_y, start_x:end_x]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y),
                          (B, G, R), -1)

    # return the pixelated blurred image
    return image
