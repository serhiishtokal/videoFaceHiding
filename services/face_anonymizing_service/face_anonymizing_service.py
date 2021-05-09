from .face_detectors.face_detector import DetectionMethod, FaceDetector
from .face_hiders.image_anonymizer import AnonymizeMethod, anonymize_face





def get_anonymized_frame(frame, detection_method: DetectionMethod, anonymize_method: AnonymizeMethod):
    face_detector=FaceDetector()
    # faces = detect_faces(frame, detection_method)
    #
    # for (x_start, y_start, w, h) in faces:
    #     x_end = x_start + w
    #     y_end = y_start + h
    #
    #     face = frame[y_start:y_end, x_start:x_end]
    #     face = anonymize_face(face, anonymize_method)
    #     frame[y_start:y_end, x_start:x_end] = face
    # return frame
    return face_detector
