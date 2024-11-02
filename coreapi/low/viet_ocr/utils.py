from __future__ import print_function
import numpy as np
import cv2
# import torch
import math
# import yaml
# import json

from ...utility.utils import reformat_input, diff, four_point_transform, calculate_ratio, compute_ratio_and_resize

def get_image_list(horizontal_list, free_list, img, model_height = 64, sort_output = True):
    image_list = []
    maximum_y,maximum_x = img.shape

    max_ratio_hori, max_ratio_free = 1,1
    for box in free_list:
        rect = np.array(box, dtype = "float32")
        transformed_img = four_point_transform(img, rect)
        ratio = calculate_ratio(transformed_img.shape[1],transformed_img.shape[0])
        new_width = int(model_height*ratio)
        if new_width == 0:
            pass
        else:
            crop_img,ratio = compute_ratio_and_resize(transformed_img,transformed_img.shape[1],transformed_img.shape[0],model_height)
            image_list.append( (box,crop_img) ) # box = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            max_ratio_free = max(ratio, max_ratio_free)

    max_ratio_free = math.ceil(max_ratio_free)

    for box in horizontal_list:
        x_min = max(0,box[0])
        x_max = min(box[1],maximum_x)
        y_min = max(0,box[2])
        y_max = min(box[3],maximum_y)
        crop_img = img[y_min : y_max, x_min:x_max]
        width = x_max - x_min
        height = y_max - y_min
        ratio = calculate_ratio(width,height)
        new_width = int(model_height*ratio)
        if new_width == 0:
            pass
        else:
            crop_img,ratio = compute_ratio_and_resize(crop_img,width,height,model_height)
            image_list.append( ( [[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]] ,crop_img) )
            max_ratio_hori = max(ratio, max_ratio_hori)

    max_ratio_hori = math.ceil(max_ratio_hori)
    max_ratio = max(max_ratio_hori, max_ratio_free)
    max_width = math.ceil(max_ratio)*model_height

    if sort_output:
        image_list = sorted(image_list, key=lambda item: item[0][0][1]) # sort by vertical position
    return image_list, max_width

if __name__ == "__main__":
    pass