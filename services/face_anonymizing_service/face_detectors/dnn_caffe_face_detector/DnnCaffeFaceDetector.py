from os.path import join, dirname
import cv2
import numpy as np
from interface import implements

from ..IFaceDetector import IFaceDetector


class DnnCaffeFaceDetector(implements(IFaceDetector)):

    def __init__(self) -> None:
        self.__root_path = dirname(__file__)
        self.__prototxt_path = join(self.__root_path, "model", "deploy.prototxt")
        self.__weights_path = join(self.__root_path, "model",
                                   "res10_300x300_ssd_iter_140000.caffemodel")
        print("[INFO] loading face detector model...")
        self.__net = cv2.dnn.readNet(self.__prototxt_path, self.__weights_path)

    def detect_faces(self, image, min_confidence=0.6):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
        self.__net.setInput(blob)
        detections = self.__net.forward()
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > min_confidence:
                box = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int")
                faces.append([box[0], box[1], box[2] - box[0], box[3] - box[1]])
        return faces
