import os
from glob import glob
from typing import Tuple, List
from datetime import datetime


import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from tutorial.io.path_definition import get_project_dir


class YoloInferenceWrapper():

    def __init__(self, cfg_code: str, model_code: str):

        """
        initialize YOLOv3 network from opencv

        Args:
            cfg_code: filename of the cfg file
            model_code: filename of the weight file
        """

        _data, _cfg, _weights = self.__load_yolo_meta(cfg_code=cfg_code, model_code=model_code)
        self.net, self.classes, self.output_layers, self.colors, self.net_height, self.net_width = self.__load_cv2_net(_cfg=_cfg, _weights=_weights)

    def predict(self, img_file: str) -> Tuple[str, float, np.ndarray]:

        """

        Args:
            img_file: path to the image file
        Returns:

        """

        outs, height, width = self.detection(img_file, self.net, self.output_layers, self.net_height, self.net_width)
        boxes, class_scores, class_ids = self.object_filter(outs, height, width)
        detections = self.nms_filter(boxes, class_scores, class_ids)

        labels = []

        for class_id, outputs in detections.items():
            boxes = outputs['boxes']
            class_scores = outputs['class_scores']

            for box, class_score in zip(boxes, class_scores):
                labels.append((str(self.classes[class_id]), class_score, box))

        if len(labels) == 0:
            label = "Not detected"
            label_proba = 0
            box = np.array([0,0,0,0])
        else:
            labels.sort(key=lambda x: x[1], reverse=True)
            label = labels[0][0]
            label_proba = np.round(labels[0][1] * 100, 1)
            box = labels[0][2]

        return label, label_proba, box

    def __load_yolo_meta(self, cfg_code: str, model_code: str):

        '''

        Each model is uniquely specified by two labels:
        the date at which it is created (DoC) and the structure (structure)
        For more details, please check either AlexeyAB/darknet or pjreddie/darknet in github
        pjreddie/darknet no longer updates so AlexeyAB/darknet might be a better place to go

        https://github.com/AlexeyAB/darknet

        Args:
            cfg_code: filename of the cfg file
            model_code: filename of the weight file
        Returns:

        '''

        cfg_folder = f"{get_project_dir()}/tutorial/cfg"
        assert os.path.isdir(cfg_folder), f"directory {cfg_folder} does not exist"

        _data = f"{cfg_folder}/template.data"
        assert os.path.isfile(_data), f".data file {_data} does not exist"

        extension = os.path.splitext(cfg_code)[1]
        assert extension == '.cfg', "The file extension is not correct for cfg_code"

        _cfg = f"{cfg_folder}/{cfg_code}"
        assert os.path.isfile(_cfg), f".cfg file {_cfg} does not exist"

        extension = os.path.splitext(model_code)[1]
        assert extension == '.weights', "The file extension is not correct for model_code"

        model_folder = f"{get_project_dir()}/trained_models/yolov3"
        _weights = f"{model_folder}/{model_code}"
        assert os.path.isfile(_weights), f".weights file {_weights} does not exist"

        return _data, _cfg, _weights

    def __load_cv2_net(self, _cfg: str, _weights: str):

        '''
        doc: this function takes the .data, .cfg, and .weights files as inputs
        and return the network, classes of output, network layers, and dimensionality of the image input in YOLO

        Args:
            _cfg: yolo neural network architecture
            _weights: yolo neural network weights
        Returns:

        '''

        # net = cv2.dnn.readNet(_weights, _cfg)
        net = cv2.dnn.readNetFromDarknet(_cfg, _weights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        classes_names = os.path.join(os.path.dirname(_cfg), "classes.names")

        with open(classes_names, 'r') as f:
            classes = f.read().splitlines()
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        with open(_cfg, 'r') as f:
            x = f.readlines()
        height = int([a for a in x if ('height' in a)][0].rstrip().split("=")[-1])
        width = int([a for a in x if ('width' in a)][0].rstrip().split("=")[-1])

        return net, classes, output_layers, colors, height, width

    def detection(self, img_file: str, net: cv2.dnn_Net, output_layers: List, net_height: int, net_width: int):

        '''

        Args:
            img_file:
            net:
            output_layers:
            net_height:
            net_width:
        Returns:

        '''

        img_input = np.array(Image.open(img_file))
        assert img_input.shape[2] == 3, 'the image does not have the right color channels'

        height, width, _ = img_input.shape
        blob = cv2.dnn.blobFromImage(img_input, 1 / 255.0, (net_height, net_width), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        return outs, height, width

    def __get_box(self, height, width, detection) -> List[int]:

        center_x = int(detection[0] * width)
        center_y = int(detection[1] * height)
        w = int(detection[2] * width)
        h = int(detection[3] * height)
        # Rectangle coordinates
        x = int(center_x - w / 2)
        y = int(center_y - h / 2)

        box = [x, y, w, h]

        return box

    def object_filter(self, outs, height, width, obj_conf_thresh: float = 0.25):

        '''

        :param outs:
        :param height:
        :param width:
        :param obj_conf_thresh:
        :return:
        '''

        class_ids = []
        class_scores = []
        boxes = []

        for out in outs:
            for detection in out:
                obj_confidence = detection[4]
                scores = obj_confidence * detection[5:]
                class_id = np.argmax(scores)
                if obj_confidence > obj_conf_thresh:
                    # Object detected
                    box = self.__get_box(height, width, detection)
                    boxes.append(box)
                    class_scores.append(float(np.max(scores)))
                    class_ids.append(class_id)

        return np.array(boxes), np.array(class_scores), class_ids

    def nms_filter(self, boxes, class_scores, class_ids, score_threshold=0.5, nms_threshold=0.4):

        '''

        :param boxes:
        :param class_scores:
        :param class_ids:
        :param score_threshold:
        :param nms_threshold:
        :return:
        '''

        get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]

        # class by class
        unique_class_ids = np.unique(class_ids)

        detections = {}

        for class_id in unique_class_ids:

            ilocs = get_indexes(class_id, class_ids)
            cls_boxes = boxes[ilocs]
            cls_class_scores = class_scores[ilocs]

            try:
                indexes = cv2.dnn.NMSBoxes(cls_boxes.tolist(), cls_class_scores.tolist(),
                                           score_threshold=score_threshold,
                                           nms_threshold=nms_threshold).flatten()
            except:
                continue

            detections[class_id] = {}
            detections[class_id]['boxes'] = cls_boxes[indexes]
            detections[class_id]['class_scores'] = cls_class_scores[indexes]

        return detections



