from os.path import dirname, join, split
import cv2
from face_detectors.face_detector import detect_faces, DetectionMethod
from face_hiders.image_anonymizer import anonymize_face, AnonymizeMethod

video_file_path = 'd:\STUDY Univer\PolitechnikaLubelska\DYPLOM\Datasets\TikTok\download.mp4'
output_file_path = join(dirname(video_file_path), "1.mp4")


def hide_face_on_stream(detection_method=DetectionMethod.CASCADE, anonymize_method=AnonymizeMethod.PIXELATE):
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        get_anonymized_frame(frame, detection_method, anonymize_method)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def hide_face_on_video(input_video_path, output_video_path, detection_method=DetectionMethod.CASCADE,
                       anonymize_method=AnonymizeMethod.PIXELATE):
    video_capture = cv2.VideoCapture(input_video_path)

    if not video_capture.isOpened():
        print("Error reading video file")
        raise
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    size = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)
    frame_count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            anonymized_frame = get_anonymized_frame(frame, detection_method, anonymize_method)
            out.write(anonymized_frame)
            frame_count += 1
            print(frame_count)
        else:
            break


def get_anonymized_frame(frame, detection_method, anonymize_method):
    faces = detect_faces(frame, detection_method)

    for (x_start, y_start, w, h) in faces:
        x_end = x_start + w
        y_end = y_start + h

        face = frame[y_start:y_end, x_start:x_end]
        face = anonymize_face(face, anonymize_method)
        frame[y_start:y_end, x_start:x_end] = face
    return frame


hide_face_on_video(video_file_path, output_file_path)
