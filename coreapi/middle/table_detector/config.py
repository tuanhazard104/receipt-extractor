""" File config.py
    Store configuration parameter of table detection and table line detection
"""
import os
from ...const import  *

dir_path = os.path.dirname(os.path.realpath(__file__))

yolov5_param = {
    'algo_config':
        {
            'weights': os.path.join(dir_path, 'models/yolov5s_best.pt'),
            'imgsz': (640, 640),  # inference size (height, width)
            'model_config': os.path.join(dir_path, 'models/yolov5s.yaml'),
            'device': 1,
            'conf_thres': 0.5,
            'iou_thres': 0.45,
            'agnostic_nms': False,
            'classes': None,
            'half': False,
        },
    'algo_name': DETECTOR_ALGORITHM_YOLOV5
}

yolov8_param = {
    'algo_config':
        {
            'weights': os.path.join(dir_path, 'models/yolov8s_best.pt'),
            'imgsz': (640, 640),  # inference size (height, width)
            'model_config': os.path.join(dir_path, 'models/yolov8s.yaml'),
            'device': 1,
            'conf_thres': 0.5,
            'iou_thres': 0.45,
            'agnostic_nms': False,
            'classes': None,
            'half': False,
        },
    'algo_name': DETECTOR_ALGORITHM_YOLOV8
}

col_lines_cfg = {
    "min_union_segments_over_height": 0.4, 
    "max_corrupt_segments_over_line": 0.45, 
    "max_distance_segments": 10, 
    "max_distance_line": 30
}