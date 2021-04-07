from os.path import dirname, join
import os
import sys
import cv2
from cascade_face_detector.cascade_face_detector import detect_faces_cascade


class Method:
    CASCADE = 1
    HOG = 2
    DNN = 3


def detect_faces(image, method=Method.CASCADE):
    switcher = {
        Method.CASCADE: detect_faces_cascade,
        Method.HOG: detect_faces_hog,
        Method.DNN: detect_faces_dnn
    }
    faces = switcher.get(method, lambda: __invalid_method_exception())(image)
    return faces


def __invalid_method_exception():
    print("Invalid method:", sys.exc_info()[0])
    raise


def detect_faces_hog(image):
    return []


def detect_faces_dnn(image):
    return []
