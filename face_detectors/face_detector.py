import sys

from face_detectors.cascade_face_detector.cascade_face_detector import detect_faces_cascade
from face_detectors.dnn_caffe_face_detector.dnn_face_detector import detect_faces_dnn_caffe


class DetectionMethod:
    CASCADE = 1
    DNN_CAFFE = 2
    DNN = 3


def detect_faces(image, method=DetectionMethod.CASCADE):
    switcher = {
        DetectionMethod.CASCADE: detect_faces_cascade,
        DetectionMethod.DNN_CAFFE: detect_faces_dnn_caffe,
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
