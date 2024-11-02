""" File detector.py
    Class TextDetector includes methods to inference text detection model
"""
import os, sys
import numpy as np
import cv2
import torch
from ...utility.utils import reformat_input, diff
from .CRAFT.contours import group_text_boxes_contour_based

if sys.version_info[0] == 2:
    from pathlib2 import Path
else:
    from pathlib import Path

# os.environ["LRU_CACHE_CAPACITY"] = "1"
BASE_PATH = os.path.dirname(__file__)
detection_models = {
    'craft' : {
        'filename': 'craft_mlt_25k.pth',
    },
    'dbnet18' : {
        'filename': 'pretrained_ic15_res18.pt',
    }
}
def group_text_box(polys, slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5, width_ths = 1.0, add_margin = 0.05, sort_output = True):
    # poly top-left, top-right, low-right, low-left
    horizontal_list, free_list,combined_list, merged_list = [],[],[],[]
    for poly in polys:
        slope_up = (poly[3]-poly[1])/np.maximum(10, (poly[2]-poly[0]))
        slope_down = (poly[5]-poly[7])/np.maximum(10, (poly[4]-poly[6]))
        if max(abs(slope_up), abs(slope_down)) < slope_ths:
            x_max = max([poly[0],poly[2],poly[4],poly[6]])
            x_min = min([poly[0],poly[2],poly[4],poly[6]])
            y_max = max([poly[1],poly[3],poly[5],poly[7]])
            y_min = min([poly[1],poly[3],poly[5],poly[7]])
            horizontal_list.append([x_min, x_max, y_min, y_max, 0.5*(y_min+y_max), y_max-y_min])
        else:
            height = np.linalg.norm([poly[6]-poly[0],poly[7]-poly[1]])
            width = np.linalg.norm([poly[2]-poly[0],poly[3]-poly[1]])

            margin = int(1.44*add_margin*min(width, height))

            theta13 = abs(np.arctan( (poly[1]-poly[5])/np.maximum(10, (poly[0]-poly[4]))))
            theta24 = abs(np.arctan( (poly[3]-poly[7])/np.maximum(10, (poly[2]-poly[6]))))
            # do I need to clip minimum, maximum value here?
            x1 = poly[0] - np.cos(theta13)*margin
            y1 = poly[1] - np.sin(theta13)*margin
            x2 = poly[2] + np.cos(theta24)*margin
            y2 = poly[3] - np.sin(theta24)*margin
            x3 = poly[4] + np.cos(theta13)*margin
            y3 = poly[5] + np.sin(theta13)*margin
            x4 = poly[6] - np.cos(theta24)*margin
            y4 = poly[7] + np.sin(theta24)*margin

            free_list.append([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
    if sort_output:
        horizontal_list = sorted(horizontal_list, key=lambda item: item[4])

    # combine box
    new_box = []
    for poly in horizontal_list:
        if len(new_box) == 0:
            b_height = [poly[5]]
            b_ycenter = [poly[4]]
            new_box.append(poly)
        else:
            # comparable height and comparable y_center level up to ths*height
            if abs(np.mean(b_ycenter) - poly[4]) < ycenter_ths*np.mean(b_height):
                b_height.append(poly[5])
                b_ycenter.append(poly[4])
                new_box.append(poly)
            else:
                b_height = [poly[5]]
                b_ycenter = [poly[4]]
                combined_list.append(new_box)
                new_box = [poly]
    combined_list.append(new_box)

    # merge list use sort again
    for boxes in combined_list:
        if len(boxes) == 1: # one box per line
            box = boxes[0]
            margin = int(add_margin*min(box[1]-box[0],box[5]))
            merged_list.append([box[0]-margin,box[1]+margin,box[2]-margin,box[3]+margin])
        else: # multiple boxes per line
            boxes = sorted(boxes, key=lambda item: item[0])

            merged_box, new_box = [],[]
            for box in boxes:
                if len(new_box) == 0:
                    b_height = [box[5]]
                    x_max = box[1]
                    new_box.append(box)
                else:
                    if (abs(np.mean(b_height) - box[5]) < height_ths*np.mean(b_height)) and ((box[0]-x_max) < width_ths *(box[3]-box[2])): # merge boxes
                        b_height.append(box[5])
                        x_max = box[1]
                        new_box.append(box)
                    else:
                        b_height = [box[5]]
                        x_max = box[1]
                        merged_box.append(new_box)
                        new_box = [box]
            if len(new_box) > 0: merged_box.append(new_box)

            for mbox in merged_box:
                if len(mbox) != 1: # adjacent box in same line
                    # do I need to add margin here?
                    x_min = min(mbox, key=lambda x: x[0])[0]
                    x_max = max(mbox, key=lambda x: x[1])[1]
                    y_min = min(mbox, key=lambda x: x[2])[2]
                    y_max = max(mbox, key=lambda x: x[3])[3]

                    box_width = x_max - x_min
                    box_height = y_max - y_min
                    margin = int(add_margin * (min(box_width, box_height)))

                    merged_list.append([x_min-margin, x_max+margin, y_min-margin, y_max+margin])
                else: # non adjacent box in same line
                    box = mbox[0]

                    box_width = box[1] - box[0]
                    box_height = box[3] - box[2]
                    margin = int(add_margin * (min(box_width, box_height)))

                    merged_list.append([box[0]-margin,box[1]+margin,box[2]-margin,box[3]+margin])
    # may need to check if box is really in image
    return merged_list, free_list

class TextDetector:
    """ Provide the model selection and methods of object classification

    Args:
        config (dict): configuration of the selected model
    """
    def __init__(self, gpu=True, 
                 model_storage_directory=None,
                 detect_network="craft",
                 quantize=True, cudnn_benchmark=False
                 ):

        self.model_storage_directory = os.path.join(BASE_PATH, 'models')
        if model_storage_directory:
            self.model_storage_directory = model_storage_directory
        Path(self.model_storage_directory).mkdir(parents=True, exist_ok=True)

        if gpu is False:
            self.device = 'cpu'
        elif not torch.cuda.is_available():
            self.device = 'cpu'
        elif gpu is True:
            self.device = 'cuda'
        else:
            self.device = gpu

        self.detection_models = detection_models
        self.quantize = quantize
        self.cudnn_benchmark=cudnn_benchmark
        self.detect_network = detect_network

        if self.detect_network == 'craft':
            from .CRAFT.detection import get_detector, get_textbox
        elif self.detect_network in ['dbnet18']:
            from .DB.detection import get_detector, get_textbox
        else:
            raise RuntimeError("Unsupport detector network. Support networks are craft and dbnet18.")
        detector_path = os.path.join(self.model_storage_directory, self.detection_models[self.detect_network]['filename'])

        self.get_textbox = get_textbox
        self.get_detector = get_detector    
        self.detector = self.get_detector(
                                 trained_model=detector_path, 
                                 device = self.device, 
                                 quantize = self.quantize, 
                                 cudnn_benchmark = self.cudnn_benchmark
                                 )
    
    def detect(self, img, min_size = 20, text_threshold = 0.7, low_text = 0.4,\
               link_threshold = 0.4, canvas_size = 2560, mag_ratio = 1.,\
               slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
               width_ths = 0.5, add_margin = 0.1, reformat=True, optimal_num_chars=None,
               threshold = 0.2, bbox_min_score = 0.2, bbox_min_size = 3, max_candidates = 0,\
               group_text_box_contour_based = True
               ):
        if reformat:
            img, img_cv_grey = reformat_input(img)

        heatmap, text_boxes = self.get_textbox(self.detector, 
                                    img, 
                                    canvas_size = canvas_size, 
                                    mag_ratio = mag_ratio,
                                    text_threshold = text_threshold, 
                                    link_threshold = link_threshold, 
                                    low_text = low_text,
                                    poly = False, 
                                    device = self.device, 
                                    optimal_num_chars = optimal_num_chars,
                                    threshold = threshold, 
                                    bbox_min_score = bbox_min_score, 
                                    bbox_min_size = bbox_min_size, 
                                    max_candidates = max_candidates,
                                    )
        if group_text_box_contour_based:
            free_list = group_text_boxes_contour_based(img, heatmap, text_boxes)
            horizontal_list = []

        else:
            horizontal_list, free_list = group_text_box(text_boxes, slope_ths,
                                                        ycenter_ths, height_ths,
                                                        width_ths, add_margin,
                                                        (optimal_num_chars is None))

        if min_size:
            horizontal_list = [i for i in horizontal_list if max(
                i[1] - i[0], i[3] - i[2]) > min_size]
            free_list = [i for i in free_list if max(
                diff([c[0] for c in i]), diff([c[1] for c in i])) > min_size]

        return horizontal_list, free_list
    