from os.path import join, dirname
import cv2

from services.face_anonymizing_service.FaceAnonymizationService import DetectionCreator, AnonymizeMethod, \
    FaceAnonymizationService

video_file_path = 'd:\STUDY Univer\PolitechnikaLubelska\DYPLOM\Datasets\TikTok\download.mp4'
output_file_path = join(dirname(video_file_path), "DNN Caffe2.mp4")
detector = DetectionCreator.ULFD(input_img_width=320)


def anonymize_video(input_video_path, output_video_path, detection_method,
                    anonymize_method=AnonymizeMethod.PIXELATE):
    face_anonymizer = FaceAnonymizationService(detection_method, anonymize_method)

    video_capture = cv2.VideoCapture(input_video_path)

    if not video_capture.isOpened():
        print("Error reading video file")
        raise
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    # fourcc = cv2.VideoWriter_fourcc(*'H264')
    fourcc = int(video_capture.get(cv2.CAP_PROP_FOURCC))
    size = (frame_width, frame_height)

    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)
    frame_count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            anonymized_frame = face_anonymizer.get_anonymized_frame(frame)
            out.write(anonymized_frame)
            frame_count += 1
            print('[INFO] frames processed: ', frame_count)
        else:
            break
    print('[INFO] Frames: ', face_anonymizer.frames)
    print('[INFO] Anonymized Frames: ', face_anonymizer.anonymized_frames)
    video_capture.release()
    cv2.destroyAllWindows()


#anonymize_video(video_file_path, output_file_path, detector, AnonymizeMethod.PIXELATE)
