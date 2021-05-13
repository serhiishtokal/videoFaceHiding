from .face_detectors.FaceDetector import DetectionCreator, FaceDetector
from .image_hiders.image_hiding import AnonymizeMethod, anonymize_face


class FaceAnonymizationService:
    def __init__(self, detection_method, anonymize_method=AnonymizeMethod.PIXELATE) -> None:
        self.__anonymize_method = anonymize_method
        self.__detection_method = detection_method
        self.__face_detector = FaceDetector(detection_method)
        self.__anonymizer = anonymize_face

    def get_anonymized_frame(self, frame):
        faces = self.__face_detector.detect_faces(frame)
        for (x_start, y_start, x_end, y_end) in faces:
            face_box = frame[y_start:y_end, x_start:x_end]
            face_box = self.__anonymizer(face_box, self.__anonymize_method)
            frame[y_start:y_end, x_start:x_end] = face_box
        return frame
