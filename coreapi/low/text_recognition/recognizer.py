""" File recognizer.py
    Class TextRecognizer includes methods to inference the OCR model
"""
import os, sys
import numpy as np
import cv2
import torch
import math
import yaml
import json

from scipy import ndimage
from bidi.algorithm import get_display

from ...utility.utils import reformat_input, diff, four_point_transform, calculate_ratio, compute_ratio_and_resize
from .config import *
from .CRNN.recognition import get_recognizer, get_text

if sys.version_info[0] == 2:
    from io import open
    from pathlib2 import Path
else:
    from pathlib import Path

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


def get_paragraph(raw_result, x_ths=1, y_ths=0.5, mode = 'ltr'):
    # create basic attributes
    box_group = []
    for box in raw_result:
        all_x = [int(coord[0]) for coord in box[0]]
        all_y = [int(coord[1]) for coord in box[0]]
        min_x = min(all_x)
        max_x = max(all_x)
        min_y = min(all_y)
        max_y = max(all_y)
        height = max_y - min_y
        box_group.append([box[1], min_x, max_x, min_y, max_y, height, 0.5*(min_y+max_y), 0]) # last element indicates group
    # cluster boxes into paragraph
    current_group = 1
    while len([box for box in box_group if box[7]==0]) > 0:
        box_group0 = [box for box in box_group if box[7]==0] # group0 = non-group
        # new group
        if len([box for box in box_group if box[7]==current_group]) == 0:
            box_group0[0][7] = current_group # assign first box to form new group
        # try to add group
        else:
            current_box_group = [box for box in box_group if box[7]==current_group]
            mean_height = np.mean([box[5] for box in current_box_group])
            min_gx = min([box[1] for box in current_box_group]) - x_ths*mean_height
            max_gx = max([box[2] for box in current_box_group]) + x_ths*mean_height
            min_gy = min([box[3] for box in current_box_group]) - y_ths*mean_height
            max_gy = max([box[4] for box in current_box_group]) + y_ths*mean_height
            add_box = False
            for box in box_group0:
                same_horizontal_level = (min_gx<=box[1]<=max_gx) or (min_gx<=box[2]<=max_gx)
                same_vertical_level = (min_gy<=box[3]<=max_gy) or (min_gy<=box[4]<=max_gy)
                if same_horizontal_level and same_vertical_level:
                    box[7] = current_group
                    add_box = True
                    break
            # cannot add more box, go to next group
            if add_box==False:
                current_group += 1
    # arrage order in paragraph
    result = []
    for i in set(box[7] for box in box_group):
        current_box_group = [box for box in box_group if box[7]==i]
        mean_height = np.mean([box[5] for box in current_box_group])
        min_gx = min([box[1] for box in current_box_group])
        max_gx = max([box[2] for box in current_box_group])
        min_gy = min([box[3] for box in current_box_group])
        max_gy = max([box[4] for box in current_box_group])

        text = ''
        while len(current_box_group) > 0:
            highest = min([box[6] for box in current_box_group])
            candidates = [box for box in current_box_group if box[6]<highest+0.4*mean_height]
            # get the far left
            if mode == 'ltr':
                most_left = min([box[1] for box in candidates])
                for box in candidates:
                    if box[1] == most_left: best_box = box
            elif mode == 'rtl':
                most_right = max([box[2] for box in candidates])
                for box in candidates:
                    if box[2] == most_right: best_box = box
            text += ' '+best_box[0]
            current_box_group.remove(best_box)

        result.append([ [[min_gx,min_gy],[max_gx,min_gy],[max_gx,max_gy],[min_gx,max_gy]], text[1:]])

    return result


def make_rotated_img_list(rotationInfo, img_list):

    result_img_list = img_list[:]

    # add rotated images to original image_list
    max_ratio=1
    
    for angle in rotationInfo:
        for img_info in img_list : 
            rotated = ndimage.rotate(img_info[1], angle, reshape=True) 
            height,width = rotated.shape
            ratio = calculate_ratio(width,height)
            max_ratio = max(max_ratio,ratio)
            result_img_list.append((img_info[0], rotated))
    return result_img_list

def set_result_with_confidence(results):
    """ Select highest confidence augmentation for TTA
    Given a list of lists of results (outer list has one list per augmentation,
    inner lists index the images being recognized), choose the best result 
    according to confidence level.
    Each "result" is of the form (box coords, text, confidence)
    A final_result is returned which contains one result for each image
    """
    final_result = []
    for col_ix in range(len(results[0])):
        # Take the row_ix associated with the max confidence
        best_row = max(
            [(row_ix, results[row_ix][col_ix][2]) for row_ix in range(len(results))],
            key=lambda x: x[1])[0]
        final_result.append(results[best_row][col_ix])

    return final_result

class TextRecognizer:
    """ Provide the model selection and methods of text recognition

    Args:
        config (dict): configuration of the selected model
    """
    def __init__(self, lang_list, gpu=True, 
                 model_storage_directory=None, user_network_directory=None,
                 recog_network='standard',
                 quantize=True, cudnn_benchmark=False):

        self.model_storage_directory = os.path.join(BASE_PATH, 'models')
        if model_storage_directory:
            self.model_storage_directory = model_storage_directory
        Path(self.model_storage_directory).mkdir(parents=True, exist_ok=True)
        self.user_network_directory = os.path.join(BASE_PATH, 'user_network')
        if user_network_directory:
            self.user_network_directory = user_network_directory
        #Path(self.user_network_directory).mkdir(parents=True, exist_ok=True)
        sys.path.append(self.user_network_directory)

        if gpu is False:
            self.device = 'cpu'
        elif not torch.cuda.is_available():
            self.device = 'cpu'
        elif gpu is True:
            self.device = 'cuda'
        else:
            self.device = gpu

        self.recognition_models = recognition_models
        self.quantize=quantize
        self.cudnn_benchmark=cudnn_benchmark
        
        # recognition model
        separator_list = {}
        if recog_network in ['standard'] + [model for model in recognition_models['gen1']] + [model for model in recognition_models['gen2']]:
            if recog_network in [model for model in recognition_models['gen1']]:
                model = recognition_models['gen1'][recog_network]
                recog_network = 'generation1'
                self.model_lang = model['model_script']
            elif recog_network in [model for model in recognition_models['gen2']]:
                model = recognition_models['gen2'][recog_network]
                recog_network = 'generation2'
                self.model_lang = model['model_script']
            else: # auto-detect
                unknown_lang = set(lang_list) - set(all_lang_list)
                if unknown_lang != set():
                    raise ValueError(unknown_lang, 'is not supported')
                # choose recognition model
                if lang_list == ['en']:
                    self.setModelLanguage('english', lang_list, ['en'], '["en"]')
                    model = recognition_models['gen2']['english_g2']
                    recog_network = 'generation2'
                elif 'ja' in lang_list:
                    self.setModelLanguage('japanese', lang_list, ['ja','en'], '["ja","en"]')
                    model = recognition_models['gen2']['japanese_g2']
                    recog_network = 'generation2'
                else:
                    self.model_lang = 'latin'
                    model = recognition_models['gen2']['latin_g2']
                    recog_network = 'generation2'
            self.character = model['characters']
            
            model_path = os.path.join(self.model_storage_directory, model['filename'])
            self.setLanguageList(lang_list, model)
        else: # user-defined model
            with open(os.path.join(self.user_network_directory, 'custom_config.json'), encoding='utf8') as file:
                recog_config = json.load(file)

            global imgH # if custom model, save this variable. (from *.yaml)
            if recog_config['imgH']:
                imgH = recog_config['imgH']

            lang = recog_config['lang']
            available_lang = recog_config['lang_list']
            self.setModelLanguage(lang, lang_list, available_lang, str(available_lang))
            #char_file = os.path.join(self.user_network_directory, recog_network+ '.txt')
            self.character = recog_config['character']
            model_file = recog_config['model_file']
            model_path = os.path.join(self.model_storage_directory, model_file)
            self.setLanguageList(lang_list, recog_config)

        dict_list = {}
        for lang in lang_list:
            dict_list[lang] = os.path.join(BASE_PATH, 'dict', lang + ".txt")
        
        if recog_network == 'generation1':
            network_params = {
                'input_channel': 1,
                'output_channel': 512,
                'hidden_size': 512,
                }
        elif recog_network == 'generation2':
            network_params = {
                'input_channel': 1,
                'output_channel': 256,
                'hidden_size': 256,
                }
        else:
            network_params = recog_config

        self.recognizer, self.converter = get_recognizer(recog_network, network_params,\
                                                        self.character, separator_list,\
                                                        dict_list, model_path, device = self.device, quantize=quantize)

    def setModelLanguage(self, language, lang_list, list_lang, list_lang_string):
        self.model_lang = language
        if set(lang_list) - set(list_lang) != set():
            raise ValueError(language.capitalize() + ' is only compatible with English, try lang_list=' + list_lang_string)

    def setLanguageList(self, lang_list, model):
        self.lang_char = []
        for lang in lang_list:
            char_file = os.path.join(BASE_PATH, 'character', lang + "_char.txt")
            with open(char_file, "r", encoding = "utf-8-sig") as input_file:
                char_list =  input_file.read().splitlines()
            self.lang_char += char_list
        if model.get('symbols'):
            symbol = model['symbols']
        elif model.get('character_list'):
            symbol = model['character_list']
        else:
            symbol = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
        self.lang_char = set(self.lang_char).union(set(symbol))
        self.lang_char = ''.join(self.lang_char)

    def recognize(self, img_cv_grey, horizontal_list=None, free_list=None,\
                  decoder = 'greedy', beamWidth= 5, batch_size = 1,\
                  workers = 0, allowlist = None, blocklist = None, detail = 1,\
                  rotation_info = None,paragraph = False,\
                  contrast_ths = 0.1,adjust_contrast = 0.5, filter_ths = 0.003,\
                  y_ths = 0.5, x_ths = 1.0, reformat=True, output_format='standard'):

        if reformat:
            img, img_cv_grey = reformat_input(img_cv_grey)

        if allowlist:
            ignore_char = ''.join(set(self.character)-set(allowlist))
        elif blocklist:
            ignore_char = ''.join(set(blocklist))
        else:
            ignore_char = ''.join(set(self.character)-set(self.lang_char))

        if self.model_lang in ['chinese_tra','chinese_sim']: decoder = 'greedy'

        if (horizontal_list==None) and (free_list==None):
            y_max, x_max = img_cv_grey.shape
            horizontal_list = [[0, x_max, 0, y_max]]
            free_list = []

        # without gpu/parallelization, it is faster to process image one by one
        if ((batch_size == 1) or (self.device == 'cpu')) and not rotation_info:
            result = []
            for bbox in horizontal_list:
                h_list = [bbox]
                f_list = []
                image_list, max_width = get_image_list(h_list, f_list, img_cv_grey, model_height = imgH)
                result0 = get_text(self.character, imgH, int(max_width), self.recognizer, self.converter, image_list,\
                              ignore_char, decoder, beamWidth, batch_size, contrast_ths, adjust_contrast, filter_ths,\
                              workers, self.device)
                result += result0
            for bbox in free_list:
                h_list = []
                f_list = [bbox]
                image_list, max_width = get_image_list(h_list, f_list, img_cv_grey, model_height = imgH)
                result0 = get_text(self.character, imgH, int(max_width), self.recognizer, self.converter, image_list,\
                              ignore_char, decoder, beamWidth, batch_size, contrast_ths, adjust_contrast, filter_ths,\
                              workers, self.device)
                result += result0
        # default mode will try to process multiple boxes at the same time
        else:
            image_list, max_width = get_image_list(horizontal_list, free_list, img_cv_grey, model_height = imgH)
            image_len = len(image_list)
            if rotation_info and image_list:
                image_list = make_rotated_img_list(rotation_info, image_list)
                max_width = max(max_width, imgH)

            result = get_text(self.character, imgH, int(max_width), self.recognizer, self.converter, image_list,\
                          ignore_char, decoder, beamWidth, batch_size, contrast_ths, adjust_contrast, filter_ths,\
                          workers, self.device)

            if rotation_info and (horizontal_list+free_list):
                # Reshape result to be a list of lists, each row being for 
                # one of the rotations (first row being no rotation)
                result = set_result_with_confidence(
                    [result[image_len*i:image_len*(i+1)] for i in range(len(rotation_info) + 1)])

        if self.model_lang == 'arabic':
            direction_mode = 'rtl'
            result = [list(item) for item in result]
            for item in result:
                item[1] = get_display(item[1])
        else:
            direction_mode = 'ltr'

        if paragraph:
            result = get_paragraph(result, x_ths=x_ths, y_ths=y_ths, mode = direction_mode)

        if detail == 0:
            return [item[1] for item in result]
        elif output_format == 'dict':
            return [ {'boxes':item[0],'text':item[1],'confident':item[2]} for item in result]
        elif output_format == 'json':
            return [json.dumps({'boxes':[list(map(int, lst)) for lst in item[0]],'text':item[1],'confident':item[2]}, ensure_ascii=False) for item in result]
        else:
            return result