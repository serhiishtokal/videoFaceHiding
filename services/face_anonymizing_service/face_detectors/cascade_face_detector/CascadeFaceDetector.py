from os.path import join
import cv2
from cv2.data import haarcascades
from interface import implements

from ..IFaceDetector import IFaceDetector


class CascadeFaceDetector(implements(IFaceDetector)):

    def __init__(self):
        self.__cascade_front_path = join(haarcascades, "haarcascade_frontalface_default.xml")
        self.__cascade_profile_path = join(haarcascades, "haarcascade_profileface.xml")
        print("[INFO] loading haarcascade frontalface detector model...")
        self.__face_front_cascade = cv2.CascadeClassifier(self.__cascade_front_path)
        print("[INFO] loading haarcascade profileface detector model...")
        self.__face_profile_cascade = cv2.CascadeClassifier(self.__cascade_profile_path)

    @staticmethod
    def __get_faces(image, face_cascade, scale_factor=1.3, min_neighbors=3):
        gray_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_image,
            scale_factor,
            min_neighbors,
            # minSize=(30,30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces

    def detect_faces(self, image) -> list:
        faces_front = self.__get_faces(image, self.__face_front_cascade, 1.2, 3)
        faces_profile = self.__get_faces(image, self.__face_profile_cascade, 1.2, 3)
        faces = list(faces_front) + list(faces_profile)
        return faces
