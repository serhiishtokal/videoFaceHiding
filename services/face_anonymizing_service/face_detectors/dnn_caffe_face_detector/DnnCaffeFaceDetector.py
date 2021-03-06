from os.path import join, dirname
import cv2
import numpy as np
from interface import implements

from ..IFaceDetector import IFaceDetector


class DnnCaffeFaceDetector(implements(IFaceDetector)):

    def __init__(self) -> None:
        root_path = dirname(__file__)
        prototxt_path = join(root_path, "model", "deploy.prototxt")
        weights_path = join(root_path, "model",
                                   "res10_300x300_ssd_iter_140000.caffemodel")
        self.__min_confidence=0.6
        print("[INFO] loading DNN Caffe face detector model...")
        self.__net = cv2.dnn.readNet(prototxt_path, weights_path)

    def detect_faces(self, image: np.ndarray) -> list:
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
        self.__net.setInput(blob)
        detections = self.__net.forward()
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.__min_confidence:
                box = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int")
                faces.append([box[0], box[1], box[2], box[3]])
        return faces
