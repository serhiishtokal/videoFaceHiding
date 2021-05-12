from interface import Interface
from numpy import ndarray


class IFaceDetector(Interface):

    def detect_faces(self, image: ndarray) -> list:
        pass
