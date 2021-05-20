import sys
from .IFaceDetector import IFaceDetector

from .cascade_face_detector.CascadeFaceDetector import CascadeFaceDetector
from .dnn_caffe_face_detector.DnnCaffeFaceDetector import DnnCaffeFaceDetector
from .ULFG_face_detector.UlfgFaceDetector import UlfgFaceDetector


class DetectionCreator:
    @staticmethod
    def CASCADE() -> IFaceDetector:
        return CascadeFaceDetector()

    @staticmethod
    def DNN_CAFFE() -> IFaceDetector:
        return DnnCaffeFaceDetector()

    # Ultra-Light-Fast-Generic-Face-Detector
    @staticmethod
    def ULFD(input_img_width: int) -> IFaceDetector:
        return UlfgFaceDetector(input_img_width=input_img_width)


class FaceDetector:
    def __init__(self, detection_method) -> None:
        self.__detection_method = detection_method
        self.__face_detector = detection_method

    def detect_faces(self, image) -> list:
        faces = self.__face_detector.detect_faces(image)
        return faces

    @staticmethod
    def __invalid_method_exception():
        print("Invalid detection method:", sys.exc_info()[0])
        raise
