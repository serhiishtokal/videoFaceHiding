from .face_detectors.FaceDetector import DetectionMethod, FaceDetector
from .face_hiders.image_anonymizer import AnonymizeMethod, anonymize_face


class FaceAnonymizationService:
    def __init__(self, detection_method=DetectionMethod.DNN_CAFFE, anonymize_method=AnonymizeMethod.PIXELATE) -> None:
        self.__anonymize_method = anonymize_method
        self.__detection_method = detection_method
        self.__face_detector = FaceDetector(detection_method)
        self.__anonymizer = anonymize_face

    def get_anonymized_frame(self, frame):
        faces = self.__face_detector.detect_faces(frame)
        for (x_start, y_start, w, h) in faces:
            x_end = x_start + w
            y_end = y_start + h

            face = frame[y_start:y_end, x_start:x_end]
            face = self.__anonymizer(face, self.__anonymize_method)
            frame[y_start:y_end, x_start:x_end] = face
        return frame
