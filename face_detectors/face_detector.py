import sys

from face_detectors.cascade_face_detector.cascade_face_detector import detect_faces_cascade


class DetectionMethod:
    CASCADE = 1
    HOG = 2
    DNN = 3


def detect_faces(image, method=DetectionMethod.CASCADE):
    switcher = {
        DetectionMethod.CASCADE: detect_faces_cascade,
        DetectionMethod.HOG: _detect_faces_hog,
        DetectionMethod.DNN: _detect_faces_dnn
    }
    faces = switcher.get(method, lambda: __invalid_method_exception())(image)
    return faces


def __invalid_method_exception():
    print("Invalid detection method:", sys.exc_info()[0])
    raise


def _detect_faces_hog(image):
    return []


def _detect_faces_dnn(image):
    return []
