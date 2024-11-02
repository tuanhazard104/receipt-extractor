""" File reader.py
    The service is to read all texts
"""
import os, sys
import cv2
import numpy as np
import re

from ...low.text_detection import TextDetector
# from ...low.text_recognition import TextRecognizer
# from ...low.strhub import TextRecognizer
from ...low.viet_ocr import TextRecognizer
from ...utility.utils import reformat_input

EXPECTED_H = 32

class TextReader(object):
    def __init__(self, lang_list=["en"], gpu=True, 
                 model_storage_directory=None, user_network_directory=None, 
                 detect_network="craft", recog_network='standard', 
                 quantize=True, cudnn_benchmark=False,
                 recog_model='standard'):
        """Create an Text Reader

        Parameters:
            lang_list (list): Language codes (ISO 639) for languages to be recognized during analysis.
            gpu (bool): Enable GPU support (default)
            model_storage_directory (string): Path to directory for model data. If not specified,
            models will be read from a directory as defined by the environment variable
            user_network_directory (string): Path to directory for custom network architecture.
            If not specified, it is as defined by the environment variable
        """
        self.detector = TextDetector(
                                    gpu=gpu, 
                                    model_storage_directory=model_storage_directory,
                                    detect_network="craft",
                                    quantize=quantize, cudnn_benchmark=cudnn_benchmark
                                    )
        self.recog_model = recog_model
        # self.recognizer = TextRecognizer(
        #                                 lang_list=lang_list,
        #                                 gpu=gpu,
        #                                 model_storage_directory=model_storage_directory,
        #                                 user_network_directory=user_network_directory,
        #                                 recog_network=recog_network,
        #                                 quantize=quantize,
        #                                 cudnn_benchmark=cudnn_benchmark
        #                                 )
        self.recognizer = TextRecognizer()
	        
    def readtext(self, image, decoder = 'greedy', beamWidth= 5, batch_size = 1,\
                 workers = 0, allowlist = None, blocklist = None, detail = 1,\
                 rotation_info = None, paragraph = False, min_size = 20,\
                 contrast_ths = 0.1,adjust_contrast = 0.5, filter_ths = 0.003,\
                 text_threshold = 0.7, low_text = 0.4, link_threshold = 0.4,\
                 canvas_size = 2560, mag_ratio = 1.,\
                 slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
                 width_ths = 0.5, y_ths = 0.5, x_ths = 1.0, add_margin = 0.1, 
                 threshold = 0.2, bbox_min_score = 0.2, bbox_min_size = 3, max_candidates = 0,
                 output_format='standard'):
        '''
        Parameters:
        image: file path or numpy-array or a byte stream object
        '''
        img, img_cv_grey = reformat_input(image)

        horizontal_list, free_list = self.detector.detect(img, 
                                                 min_size = min_size, text_threshold = text_threshold,\
                                                 low_text = low_text, link_threshold = link_threshold,\
                                                 canvas_size = canvas_size, mag_ratio = mag_ratio,\
                                                 slope_ths = slope_ths, ycenter_ths = ycenter_ths,\
                                                 height_ths = height_ths, width_ths= width_ths,\
                                                 add_margin = add_margin, reformat = False,\
                                                 threshold = threshold, bbox_min_score = bbox_min_score,\
                                                 bbox_min_size = bbox_min_size, max_candidates = max_candidates
                                                 )
        # get the 1st result from hor & free list as self.detect returns a list of depth 3
        # # # horizontal_list, free_list = horizontal_list[0], free_list[0] 
        # result = self.recognizer.recognize(img_cv_grey, horizontal_list, free_list,\
        #                         decoder, beamWidth, batch_size,\
        #                         workers, allowlist, blocklist, detail, rotation_info,\
        #                         paragraph, contrast_ths, adjust_contrast,\
        #                         filter_ths, y_ths, x_ths, False, output_format)
        result = self.recognizer.recognize(img_cv_grey, horizontal_list, free_list)
        return result

