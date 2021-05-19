from  time import time

import cv2

from services.face_anonymizing_service.FaceAnonymizationService import DetectionCreator, AnonymizeMethod, \
    FaceAnonymizationService

detector = DetectionCreator.ULFD(input_img_width=640)
# detector = DetectionCreator.DNN_CAFFE()


def anonymize_stream(detection_method, anonymize_method=AnonymizeMethod.PIXELATE):
    face_anonymizer = FaceAnonymizationService(detection_method, anonymize_method)
    video_capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    while True:
        start_time = time()
        ret, frame = video_capture.read()
        anonymized_frame = face_anonymizer.get_anonymized_frame(frame)
        cv2.imshow('Video', anonymized_frame)
        print(time()-start_time)
        # print(f'{detector_type_str} detector inference time: ', time()-start_time)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    video_capture.release()


anonymize_stream(detector, AnonymizeMethod.PIXELATE)
