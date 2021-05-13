from os.path import join, dirname
import cv2

from services.face_anonymizing_service.FaceAnonymizationService import DetectionCreator, AnonymizeMethod, \
    FaceAnonymizationService

video_file_path = 'd:\STUDY Univer\PolitechnikaLubelska\DYPLOM\Datasets\TikTok\download.mp4'
output_directory_path = join(dirname(video_file_path), "anonymized_images")


def anonymize_video_to_images_opencv(input_video_path, output_images_directory,
                                     detection_method=DetectionCreator.DNN_CAFFE,
                                     anonymize_method=AnonymizeMethod.PIXELATE):
    face_anonymizer = FaceAnonymizationService(detection_method, anonymize_method)

    video_capture = cv2.VideoCapture(input_video_path)
    if not video_capture.isOpened():
        print("Error reading video file")
        raise
    frame_count = 0

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        anonymized_frame = face_anonymizer.get_anonymized_frame(frame)
        output_path = join(output_images_directory, "%s.jpg" % frame_count)
        cv2.imwrite(output_path, anonymized_frame)
        frame_count += 1
        print('[INFO] frames processed: ', frame_count)
    video_capture.release()
    cv2.destroyAllWindows()


anonymize_video_to_images_opencv(video_file_path, output_directory_path, DetectionCreator.DNN_CAFFE, AnonymizeMethod.PIXELATE)