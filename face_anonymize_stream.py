import cv2

from services.face_anonymizing_service.face_anonymizing_service import DetectionMethod, AnonymizeMethod, \
    get_anonymized_frame


def anonymize_stream(detection_method=DetectionMethod.CASCADE, anonymize_method=AnonymizeMethod.PIXELATE):
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        get_anonymized_frame(frame, detection_method, anonymize_method)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


anonymize_stream(DetectionMethod.DNN_CAFFE)
