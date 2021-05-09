import sys
from enum import Enum, unique

from .IFaceDetector import IFaceDetector

from .cascade_face_detector.CascadeFaceDetector import CascadeFaceDetector
from .dnn_caffe_face_detector.DnnCaffeFaceDetector import DnnCaffeFaceDetector


@unique
class DetectionMethod(Enum):
    CASCADE = 1
    DNN_CAFFE = 2


class FaceDetector:
    def __init__(self, detection_method=DetectionMethod.DNN_CAFFE) -> None:
        self.__detection_method = detection_method
        self.face_detector = self.__init_face_detector(detection_method)

    def detect_faces(self, image):
        faces = self.face_detector.detect_faces(image)
        return faces

    def __init_face_detector(self, detection_method: DetectionMethod) -> IFaceDetector:
        switcher = {
            DetectionMethod.CASCADE: CascadeFaceDetector,
            DetectionMethod.DNN_CAFFE: DnnCaffeFaceDetector,
        }
        face_detector_class = switcher.get(detection_method, lambda: self.__invalid_method_exception())
        return face_detector_class()

    @staticmethod
    def __invalid_method_exception():
        print("Invalid detection method:", sys.exc_info()[0])
        raise
