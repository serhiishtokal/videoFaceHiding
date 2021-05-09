import cv2

from services.face_anonymizing_service.FaceAnonymizationService import DetectionMethod, AnonymizeMethod, \
    FaceAnonymizationService


def anonymize_stream(detection_method=DetectionMethod.DNN_CAFFE, anonymize_method=AnonymizeMethod.PIXELATE):
    face_anonymizer = FaceAnonymizationService(detection_method, anonymize_method)
    video_capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    while True:
        ret, frame = video_capture.read()
        face_anonymizer.get_anonymized_frame(frame)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    video_capture.release()


anonymize_stream(DetectionMethod.DNN_CAFFE, AnonymizeMethod.PIXELATE)
