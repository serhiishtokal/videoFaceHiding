from  time import time

import cv2

from services.face_anonymizing_service.FaceAnonymizationService import DetectionMethod, AnonymizeMethod, \
    FaceAnonymizationService

detector_type = DetectionMethod.ULFD
detector_type_str = detector_type.__name__


def anonymize_stream(detection_method=DetectionMethod.DNN_CAFFE, anonymize_method=AnonymizeMethod.PIXELATE):
    face_anonymizer = FaceAnonymizationService(detection_method, anonymize_method)
    video_capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    while True:
        start_time = time()
        ret, frame = video_capture.read()
        face_anonymizer.get_anonymized_frame(frame)
        cv2.imshow('Video', frame)
        print(time()-start_time)
        # print(f'{detector_type_str} detector inference time: ', time()-start_time)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    video_capture.release()


anonymize_stream(detector_type, AnonymizeMethod.PIXELATE)
