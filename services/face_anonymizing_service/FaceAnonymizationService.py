from .face_detectors.FaceDetector import DetectionCreator, FaceDetector
from .image_hiders.image_hiding import AnonymizeMethod, anonymize_face


class FaceAnonymizationService:
    def __init__(self, detection_method, anonymize_method=AnonymizeMethod.PIXELATE) -> None:
        self.__anonymize_method = anonymize_method
        self.__detection_method = detection_method
        self.__face_detector = FaceDetector(detection_method)
        self.__anonymizer = anonymize_face
        self.frames = 0
        self.anonymized_frames = 0

    def get_anonymized_frame(self, frame):
        frame = frame.copy()
        faces = self.__face_detector.detect_faces(frame)
        self.frames += 1
        if len(faces) != 0:
            self.anonymized_frames += 1
        for (x_start, y_start, x_end, y_end) in faces:
            face_box = frame[y_start:y_end, x_start:x_end]
            face_box = self.__anonymizer(face_box, self.__anonymize_method)
            frame[y_start:y_end, x_start:x_end] = face_box
        return frame
