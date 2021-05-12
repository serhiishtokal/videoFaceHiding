from pathlib import Path
import cv2
import numpy as np
import torch
from numpy import ndarray
from interface import implements

from ..IFaceDetector import IFaceDetector

from .vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from .vision.ssd.config.fd_config import define_img_size
from .vision.utils.misc import Timer


# Ultra-Light-Fast-Generic-Face-Detector
class UlfgFaceDetector(implements(IFaceDetector)):

    def __init__(self) -> None:
        input_img_size = 480
        define_img_size(input_img_size)

        voc_model_path = Path(__file__).parent / "./models/voc-model-labels.txt"

        class_names = [name.strip() for name in open(voc_model_path).readlines()]
        num_classes = len(class_names)
        test_device = TestDevice.CPU
        is_test = True
        self.__candidate_size = 1000
        self.__threshold = 0.7

        model_path = Path(__file__).parent / "./models/pretrained/version-RFB-640.pth"
        print("[INFO] loading ULFG face detector net...")

        net = create_Mb_Tiny_RFB_fd(num_classes, is_test=is_test, device=test_device)
        self.__predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=self.__candidate_size,
                                                           device=test_device)
        net.load(model_path)

    def detect_faces(self, image: ndarray) -> list:
        # (h, w) = image.shape[:2]
        # faces = []
        boxes_tensor, labels, probs = self.__predictor.predict(image, self.__candidate_size / 2, self.__threshold)

        boxes_tensor_int = boxes_tensor.type(torch.IntTensor)

        # for i in range(boxes_tensor.size(0)):
        #     box = boxes_tensor[i, :].type(torch.IntTensor)
        #     faces.append([box[0], box[1], box[2]-box[0], box[3]-box[1]])


        # if not boxes_tensor.size(0) > 0:
        #     return faces
        # boxes_np = boxes_tensor.cpu().detach().numpy()
        # for i in range(boxes_np.shape[0]):
        #     detections = boxes_np[i, :]
        #     box = (detections[0:4]).astype("int")
        #     faces.append([box[0], box[1], box[2] - box[0], box[3] - box[1]])
        #     # label = f" {probs[i]:.2f}"
        #     # point1 = (box[0], box[1])
        #     # point2 = (box[2], box[3])
        #     # color = (0, 255, 0)
        #     # thickness = 4
        #     # cv2.rectangle(image, point1, point2, color, thickness)
        return boxes_tensor_int


class TestDevice:
    CPU = "cpu"
    GPU = "cuda:0"
