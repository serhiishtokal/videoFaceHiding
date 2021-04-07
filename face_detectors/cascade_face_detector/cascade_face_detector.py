from os.path import dirname, join
import os
import cv2

root_path = dirname(__file__)
cascade_front_path = join(root_path, "haarcascade_frontalface_default.xml")
cascade_profile_path = join(root_path, "haarcascade_profileface.xml")
face_front_cascade = cv2.CascadeClassifier(cascade_front_path)
face_profile_cascade = cv2.CascadeClassifier(cascade_profile_path)


def _get_faces(image, face_cascade, scale_factor=1.3, min_neighbors=3):
    gray_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_image,
        scale_factor,
        min_neighbors,
        # minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces


def detect_faces_cascade(image):
    faces_front = _get_faces(image, face_front_cascade, 1.2, 3)
    faces_profile = _get_faces(image, face_profile_cascade, 1.2, 3)
    faces = list(faces_front) + list(faces_profile)
    return faces
