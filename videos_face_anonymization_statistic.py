import os
from time import time, strftime, gmtime
from glob import glob
import cv2

from services.face_anonymizing_service.FaceAnonymizationService import DetectionCreator, AnonymizeMethod, \
    FaceAnonymizationService

# detector = DetectionCreator.ULFD(input_img_width=640)
# detector = DetectionCreator.DNN_CAFFE()
# detector = DetectionCreator.CASCADE()

# anonymizationMethod = AnonymizeMethod.BLUR
# anonymizationMethod = AnonymizeMethod.PIXELATE
# anonymizationMethod = AnonymizeMethod.BlackRectangle

# face_anonymizer = FaceAnonymizationService(detector, anonymizationMethod)

input_folder = "d:\\STUDY Univer\\DYPLOM 2021\\Pollub\\SerhiiDataset\\"
videos_extension = 'mp4'
input_videos_paths = glob(input_folder + '*.' + videos_extension)


def get_statistics():
    videos_statistic_info = []
    total_fc = 0
    total_afc = 0
    for input_video_path in input_videos_paths:
        start_video = time()
        filename = os.path.basename(input_video_path)
        print(filename)

        fc, afc = anonymize_video(input_video_path)
        total_fc += fc
        total_afc += afc

        end_video = time()
        secs_by_video = end_video - start_video

        videos_statistic_info.append((filename, fc, afc, secs_by_video))
    return (total_afc, total_fc), videos_statistic_info


def anonymize_video(input_video_path):
    video_capture = cv2.VideoCapture(input_video_path)
    if not video_capture.isOpened():
        print("Error reading video file")
        raise
    frames_count = 0
    anonymized_frames_count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            print("[INFO] Can't receive frame (stream end?). Exiting ...")
            break
        faces_count = face_anonymizer.anonymize_frame(frame)
        frames_count += 1
        if faces_count > 0:
            anonymized_frames_count += 1
        # print(f'[INFO] Frame {frames_count} processed, faces: {faces_count}')

    return frames_count, anonymized_frames_count


def proceed():
    time_start_videos_proc = time()
    (t_afc, t_fc), videos_info = get_statistics()
    time_end_videos_proc = time()

    elapsed_time = strftime('%H:%M:%S', gmtime(time_end_videos_proc - time_start_videos_proc))
    total_rate = format(t_afc / t_fc, ".4f")

    # print(f'[RESULT] Elapsed total time [{elapsed_time}] total_rate [{total_rate}]')
    printVideosInfo(videos_info)


def printVideosInfo(videos_infos):
    for video_info in videos_infos:
        input_video_name, fc, afc, secs_by_video = video_info
        formatted_time = strftime('%H:%M:%S', gmtime(secs_by_video))
        formatted_rate = format(afc / fc, ".4f")

        print(
            f'[RESULT VIDEO] frames [{afc}]/[{fc}] rate=[{formatted_rate}]  time [{formatted_time}] {input_video_name}')


# ULFD
print('--ULFD + BLUR--')
detector = DetectionCreator.ULFD(input_img_width=640)
anonymizationMethod = AnonymizeMethod.BLUR
face_anonymizer = FaceAnonymizationService(detector, anonymizationMethod)
proceed()
print('--ULFD + PIXELATE--')
detector = DetectionCreator.ULFD(input_img_width=640)
anonymizationMethod = AnonymizeMethod.PIXELATE
face_anonymizer = FaceAnonymizationService(detector, anonymizationMethod)
proceed()
print('--ULFD + BlackRectangle--')
detector = DetectionCreator.ULFD(input_img_width=640)
anonymizationMethod = AnonymizeMethod.BlackRectangle
face_anonymizer = FaceAnonymizationService(detector, anonymizationMethod)
proceed()


# DNN_CAFFE
print('--DNN_CAFFE + BLUR--')
detector = detector = DetectionCreator.DNN_CAFFE()
anonymizationMethod = AnonymizeMethod.BLUR
face_anonymizer = FaceAnonymizationService(detector, anonymizationMethod)
proceed()
print('--DNN_CAFFE + PIXELATE--')
detector = detector = DetectionCreator.DNN_CAFFE()
anonymizationMethod = AnonymizeMethod.PIXELATE
face_anonymizer = FaceAnonymizationService(detector, anonymizationMethod)
proceed()
print('--DNN_CAFFE + PIXELATE--')
detector = detector = DetectionCreator.DNN_CAFFE()
anonymizationMethod = AnonymizeMethod.BlackRectangle
face_anonymizer = FaceAnonymizationService(detector, anonymizationMethod)
proceed()

# CASCADE
print('--CASCADE + BLUR--')
detector = detector = DetectionCreator.CASCADE()
anonymizationMethod = AnonymizeMethod.BLUR
face_anonymizer = FaceAnonymizationService(detector, anonymizationMethod)
proceed()
print('--CASCADE + PIXELATE--')
detector = detector = DetectionCreator.CASCADE()
anonymizationMethod = AnonymizeMethod.PIXELATE
face_anonymizer = FaceAnonymizationService(detector, anonymizationMethod)
proceed()
print('--CASCADE + BlackRectangle--')
detector = detector = DetectionCreator.CASCADE()
anonymizationMethod = AnonymizeMethod.BlackRectangle
face_anonymizer = FaceAnonymizationService(detector, anonymizationMethod)
proceed()


