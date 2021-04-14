from os.path import join, dirname
import cv2
import numpy as np

root_path = dirname(__file__)
print("[INFO] loading face detector model...")
prototxt_path = join(root_path, "model", "deploy.prototxt")
weights_path = join(root_path, "model",
                    "res10_300x300_ssd_iter_140000.caffemodel")


def detect_faces_dnn_caffe(image, min_confidence=0.6):
    net = cv2.dnn.readNet(prototxt_path, weights_path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > min_confidence:
            box = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int")
            faces.append([box[0], box[1], box[2]-box[0], box[3]-box[1]])
    return faces
